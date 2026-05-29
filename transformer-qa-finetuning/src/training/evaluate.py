"""
evaluate.py
===========
Standalone evaluation script for a fine-tuned QA model.

Usage:
    python -m src.training.evaluate \\
        --model_path results/checkpoints/checkpoint-best \\
        --dataset squad \\
        --config configs/default.yaml \\
        --output_dir results/metrics

    # Evaluate on SQuAD v2 with null-answer threshold
    python -m src.training.evaluate \\
        --model_path results/checkpoints/checkpoint-best \\
        --dataset squad_v2 \\
        --squad_v2 \\
        --null_score_diff_threshold 0.5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.collator import build_collator
from src.data.dataset_loader import load_squad_dataset
from src.data.preprocessing import QAPreprocessor, postprocess_qa_predictions
from src.models.qa_model import load_qa_model, load_qa_tokenizer
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_calibration, softmax_confidence, squad_evaluate
from src.utils.reproducibility import get_device, set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned QA model on SQuAD.")
    p.add_argument("--model_path", required=True, help="Path to checkpoint directory.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", default="squad", choices=["squad", "squad_v2"])
    p.add_argument("--squad_v2", action="store_true")
    p.add_argument("--null_score_diff_threshold", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", default="results/metrics")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate_model(
    model_path: str,
    config: dict,
    dataset_name: str = "squad",
    batch_size: int = 32,
    squad_v2: bool = False,
    null_score_diff_threshold: float = 0.0,
    output_dir: str = "results/metrics",
    seed: int = 42,
) -> dict:
    """
    Full evaluation pipeline: load → tokenise → infer → postprocess → score.

    Args:
        model_path: Directory of the fine-tuned checkpoint.
        config: Parsed YAML config dict.
        dataset_name: "squad" or "squad_v2".
        batch_size: Inference batch size.
        squad_v2: Whether dataset has unanswerable questions.
        null_score_diff_threshold: SQuAD v2 threshold.
        output_dir: Where to write evaluation results.
        seed: Random seed.

    Returns:
        Dictionary with EM, F1, calibration, and per-example predictions.
    """
    import torch
    from torch.utils.data import DataLoader

    set_seed(seed)
    device = get_device()

    # Load model and tokenizer
    tokenizer = load_qa_tokenizer(model_path)
    model = load_qa_model(model_path)
    model.to(device)
    model.eval()

    # Load dataset
    data_cfg = config.get("data", {})
    dataset = load_squad_dataset(
        version=dataset_name,
        cache_dir=data_cfg.get("cache_dir", "data/processed"),
    )
    eval_examples = dataset["validation"]

    # Tokenise
    model_cfg = config.get("model", {})
    preprocessor = QAPreprocessor(
        tokenizer=tokenizer,
        max_seq_length=model_cfg.get("max_seq_length", 384),
        doc_stride=model_cfg.get("doc_stride", 128),
        pad_to_max_length=False,
        is_training=False,
    )
    tokenised_eval = eval_examples.map(
        preprocessor,
        batched=True,
        remove_columns=eval_examples.column_names,
        desc="Tokenising for evaluation",
    )
    tokenised_eval.set_format("torch")

    # DataLoader
    collator = build_collator(tokenizer)
    eval_loader = DataLoader(
        tokenised_eval,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=4,
    )

    # Inference
    all_start_logits = []
    all_end_logits = []
    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in {"input_ids", "attention_mask", "token_type_ids"}
                and v is not None
            }
            outputs = model(**input_batch)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)

    # Post-process
    predictions, nbest = postprocess_qa_predictions(
        examples=eval_examples,
        features=tokenised_eval,
        raw_predictions=(start_logits, end_logits),
        tokenizer=tokenizer,
        n_best_size=model_cfg.get("n_best_size", 20),
        max_answer_length=model_cfg.get("max_answer_length", 30),
        null_score_diff_threshold=null_score_diff_threshold,
        squad_v2=squad_v2,
    )

    # Official metrics
    references = {
        ex["id"]: ex["answers"]["text"] for ex in eval_examples
    }
    metrics = squad_evaluate(predictions, references)

    # Calibration
    confidences = []
    for i in range(len(tokenised_eval)):
        conf = softmax_confidence(start_logits[i], end_logits[i])
        confidences.append(conf)

    # Map confidence to example level (first feature per example)
    from src.utils.metrics import compute_exact, compute_f1

    ex_confidences = []
    ex_correct = []
    seen_ids = set()
    for i, feat in enumerate(tokenised_eval):
        eid = feat["example_id"]
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        pred = predictions.get(eid, "")
        golds = references.get(eid, [""])
        correct = max(compute_exact(pred, g) for g in golds)
        ex_confidences.append(confidences[i])
        ex_correct.append(float(correct))

    calibration = compute_calibration(ex_confidences, ex_correct, n_bins=15)
    metrics["ece"] = calibration["ece"]
    metrics["mce"] = calibration["mce"]

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = Path(model_path).name
    results_path = out_dir / f"eval_{dataset_name}_{model_tag}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "calibration": calibration,
                "model_path": model_path,
                "dataset": dataset_name,
            },
            f,
            indent=2,
        )

    # Save predictions
    pred_path = Path("results/predictions") / f"predictions_{dataset_name}_{model_tag}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info("=" * 55)
    logger.info("EVALUATION RESULTS — %s | %s", dataset_name, model_tag)
    logger.info("  Exact Match : %.2f%%", metrics["exact_match"])
    logger.info("  F1 Score    : %.2f%%", metrics["f1"])
    logger.info("  Total       : %d", metrics["total"])
    logger.info("  ECE         : %.4f", metrics["ece"])
    logger.info("=" * 55)
    logger.info("Results saved → %s", results_path)

    return metrics


def main() -> None:
    args = parse_args()
    setup_logging(experiment_name="evaluate")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate_model(
        model_path=args.model_path,
        config=config,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        squad_v2=args.squad_v2,
        null_score_diff_threshold=args.null_score_diff_threshold,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

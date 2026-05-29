"""
predict.py
==========
Production-ready inference interface for the fine-tuned QA model.

Supports:
  - Single question/context pair
  - Batch inference from JSON file
  - Confidence scoring
  - Top-N answer candidates

Usage:
    # Single inference
    python -m src.inference.predict \\
        --model_path results/checkpoints/checkpoint-best \\
        --question "When was the Eiffel Tower built?" \\
        --context "The Eiffel Tower was built in 1889 as the entrance arch..."

    # Batch inference from JSON
    python -m src.inference.predict \\
        --model_path results/checkpoints/checkpoint-best \\
        --input_file data/raw/custom_questions.json \\
        --output_file results/predictions/custom_predictions.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.qa_model import load_qa_model, load_qa_tokenizer
from src.utils.logging_utils import setup_logging
from src.utils.metrics import softmax_confidence
from src.utils.reproducibility import get_device

logger = logging.getLogger(__name__)


class QAPredictor:
    """
    High-level inference class for a fine-tuned extractive QA model.

    Args:
        model_path: Directory containing the fine-tuned checkpoint.
        max_seq_length: Truncation length for context + question.
        n_best_size: Number of answer candidates to return.
        max_answer_length: Maximum span length for a valid answer.
    """

    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 384,
        n_best_size: int = 5,
        max_answer_length: int = 30,
    ) -> None:
        self.device = get_device()
        self.tokenizer = load_qa_tokenizer(model_path)
        self.model = load_qa_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.max_seq_length = max_seq_length
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length

        logger.info(
            "QAPredictor ready — model=%s | device=%s",
            model_path,
            self.device,
        )

    @torch.no_grad()
    def predict(
        self,
        question: str,
        context: str,
    ) -> dict:
        """
        Predict an answer span for a single QA pair.

        Args:
            question: Natural language question string.
            context: Passage that may contain the answer.

        Returns:
            Dictionary with:
              - "answer": Best answer string.
              - "confidence": Model confidence (0–1).
              - "start": Character start position in context.
              - "end": Character end position in context.
              - "n_best": List of top-N candidates.
        """
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].numpy()
        sequence_ids = inputs.sequence_ids(0)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()

        # Collect top-N candidates
        start_idxs = np.argsort(start_logits)[::-1][: self.n_best_size]
        end_idxs = np.argsort(end_logits)[::-1][: self.n_best_size]

        candidates = []
        for si in start_idxs:
            for ei in end_idxs:
                if (
                    sequence_ids[si] != 1
                    or sequence_ids[ei] != 1
                    or ei < si
                    or (ei - si + 1) > self.max_answer_length
                    or offset_mapping[si] is None
                    or offset_mapping[ei] is None
                ):
                    continue
                start_char = int(offset_mapping[si][0])
                end_char = int(offset_mapping[ei][1])
                text = context[start_char:end_char]
                candidates.append(
                    {
                        "text": text,
                        "score": float(start_logits[si] + end_logits[ei]),
                        "start": start_char,
                        "end": end_char,
                        "start_logit": float(start_logits[si]),
                        "end_logit": float(end_logits[ei]),
                    }
                )

        if not candidates:
            return {
                "answer": "",
                "confidence": 0.0,
                "start": -1,
                "end": -1,
                "n_best": [],
            }

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        confidence = softmax_confidence(start_logits, end_logits)

        return {
            "answer": best["text"],
            "confidence": confidence,
            "start": best["start"],
            "end": best["end"],
            "n_best": candidates[: self.n_best_size],
        }

    def predict_batch(
        self,
        examples: list[dict],
    ) -> list[dict]:
        """
        Run inference on a list of {"question": ..., "context": ...} dicts.

        Args:
            examples: List of QA dicts.

        Returns:
            List of prediction dicts, one per input.
        """
        results = []
        for ex in tqdm(examples, desc="Running inference"):
            pred = self.predict(ex["question"], ex["context"])
            pred["id"] = ex.get("id", None)
            pred["question"] = ex["question"]
            results.append(pred)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a fine-tuned QA model.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--context", type=str, default=None)
    p.add_argument("--input_file", type=str, default=None,
                   help="JSON file with list of {id, question, context}.")
    p.add_argument("--output_file", type=str, default="results/predictions/predictions.json")
    p.add_argument("--n_best", type=int, default=5)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--max_answer_length", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(experiment_name="inference")

    predictor = QAPredictor(
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        n_best_size=args.n_best,
        max_answer_length=args.max_answer_length,
    )

    if args.question and args.context:
        # Single prediction
        result = predictor.predict(args.question, args.context)
        print("\n" + "=" * 60)
        print(f"QUESTION  : {args.question}")
        print(f"ANSWER    : {result['answer']}")
        print(f"CONFIDENCE: {result['confidence']:.4f}")
        print(f"SPAN      : chars {result['start']}–{result['end']}")
        print("=" * 60)
        print("\nTop candidates:")
        for i, cand in enumerate(result["n_best"], 1):
            print(f"  [{i}] '{cand['text']}' (score={cand['score']:.3f})")

    elif args.input_file:
        with open(args.input_file) as f:
            examples = json.load(f)

        predictions = predictor.predict_batch(examples)

        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2)
        logger.info("Predictions saved → %s", out_path)

    else:
        print("Provide --question + --context, or --input_file.")
        sys.exit(1)


if __name__ == "__main__":
    main()

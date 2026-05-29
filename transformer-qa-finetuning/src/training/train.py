"""
train.py
========
Entry point for fine-tuning DistilBERT on SQuAD.

Usage:
    python -m src.training.train --config configs/default.yaml
    python -m src.training.train --config configs/cosine_scheduler.yaml
    python -m src.training.train --config configs/default.yaml --fp16 false
    python -m src.training.train --config configs/default.yaml --lr 2e-5 --epochs 2
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset_loader import load_squad_dataset
from src.data.preprocessing import QAPreprocessor
from src.models.qa_model import load_qa_model, load_qa_tokenizer
from src.training.trainer import QATrainer
from src.utils.logging_utils import (
    MetricsLogger,
    get_tensorboard_writer,
    init_wandb,
    setup_logging,
)
from src.utils.reproducibility import log_environment_info, set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for Extractive QA on SQuAD"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override train batch size.")
    parser.add_argument("--fp16", type=str, default=None, choices=["true", "false"],
                        help="Override fp16 flag.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and return YAML config as a nested dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI argument overrides to the loaded config."""
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.epochs is not None:
        config["training"]["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    if args.fp16 is not None:
        config["training"]["fp16"] = args.fp16 == "true"
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = args.output_dir
    if args.no_wandb:
        config["wandb"]["enabled"] = False
    return config


def main() -> None:
    args = parse_args()

    # --- Config ---
    config = load_config(args.config)
    config = apply_overrides(config, args)
    exp_cfg = config["experiment"]
    train_cfg = config["training"]

    # --- Logging ---
    setup_logging(
        log_dir=exp_cfg.get("log_dir", "results/logs"),
        experiment_name=exp_cfg["name"],
    )
    log_environment_info()

    logger.info("Experiment: %s", exp_cfg["name"])
    logger.info("Config: %s", args.config)
    logger.info("Config values:\n%s", yaml.dump(config, default_flow_style=False))

    # --- Reproducibility ---
    set_seed(
        seed=exp_cfg.get("seed", 42),
        deterministic=exp_cfg.get("deterministic", True),
    )

    # --- Tokenizer & Model ---
    model_name = config["model"]["name"]
    tokenizer = load_qa_tokenizer(model_name)
    model = load_qa_model(model_name)

    # --- Dataset ---
    data_cfg = config["data"]
    dataset = load_squad_dataset(
        version=data_cfg.get("dataset_name", "squad"),
        cache_dir=data_cfg.get("cache_dir", "data/processed"),
        overwrite_cache=data_cfg.get("overwrite_cache", False),
    )

    # --- Preprocessing ---
    model_cfg = config["model"]
    train_preprocessor = QAPreprocessor(
        tokenizer=tokenizer,
        max_seq_length=model_cfg["max_seq_length"],
        doc_stride=model_cfg["doc_stride"],
        pad_to_max_length=data_cfg.get("pad_to_max_length", False),
        is_training=True,
    )
    eval_preprocessor = QAPreprocessor(
        tokenizer=tokenizer,
        max_seq_length=model_cfg["max_seq_length"],
        doc_stride=model_cfg["doc_stride"],
        pad_to_max_length=data_cfg.get("pad_to_max_length", False),
        is_training=False,
    )

    logger.info("Tokenising training set...")
    tokenised_train = dataset["train"].map(
        train_preprocessor,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=data_cfg.get("preprocessing_num_workers", 4),
        desc="Tokenising train",
    )
    tokenised_train.set_format("torch")

    logger.info("Tokenising validation set...")
    tokenised_eval = dataset["validation"].map(
        eval_preprocessor,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        num_proc=data_cfg.get("preprocessing_num_workers", 4),
        desc="Tokenising eval",
    )
    tokenised_eval.set_format("torch")

    # --- Experiment Tracking ---
    tb_writer = get_tensorboard_writer(
        log_dir=exp_cfg.get("log_dir", "results/logs"),
        experiment_name=exp_cfg["name"],
    )
    wandb_run = init_wandb(
        config=config,
        enabled=config.get("wandb", {}).get("enabled", False),
    )
    metrics_logger = MetricsLogger(tb_writer=tb_writer, wandb_run=wandb_run)

    # --- Train ---
    trainer = QATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_eval,
        eval_examples=dataset["validation"],
        config=config,
        metrics_logger=metrics_logger,
    )

    results = trainer.train()

    # --- Save final metrics ---
    metrics_path = Path(exp_cfg["output_dir"]) / "metrics" / f"{exp_cfg['name']}_results.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Training complete. Best F1: %.2f", results["best_f1"])
    logger.info("Results saved → %s", metrics_path)
    metrics_logger.close()


if __name__ == "__main__":
    main()

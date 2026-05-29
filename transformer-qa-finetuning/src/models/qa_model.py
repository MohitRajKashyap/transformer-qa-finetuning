"""
qa_model.py
===========
Thin wrapper around HuggingFace AutoModelForQuestionAnswering.

Provides:
  - Clean model loading with error handling and logging
  - Parameter count summary
  - Checkpoint save / load helpers
  - Easy extension to BERT, RoBERTa, ELECTRA, DeBERTa, T5, etc.

Usage:
    model = load_qa_model("distilbert-base-uncased")
    model = load_qa_model("roberta-base")  # drop-in replacement
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

logger = logging.getLogger(__name__)


def load_qa_model(
    model_name_or_path: str = "distilbert-base-uncased",
    from_checkpoint: Optional[str] = None,
) -> PreTrainedModel:
    """
    Load a pre-trained or fine-tuned QA model.

    Args:
        model_name_or_path: HuggingFace Hub model ID or local path.
        from_checkpoint: If provided, load weights from this checkpoint
            directory (overrides model_name_or_path for weights only).

    Returns:
        model ready for training or inference.
    """
    load_path = from_checkpoint or model_name_or_path
    logger.info("Loading model from: %s", load_path)

    model = AutoModelForQuestionAnswering.from_pretrained(load_path)
    _log_model_stats(model, load_path)
    return model


def load_qa_tokenizer(
    model_name_or_path: str = "distilbert-base-uncased",
) -> PreTrainedTokenizerFast:
    """
    Load the tokenizer corresponding to a QA model.

    Enforces use_fast=True so that offset_mapping is available.

    Args:
        model_name_or_path: HuggingFace Hub model ID or local path.

    Returns:
        Fast PreTrainedTokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError(
            f"The tokenizer for '{model_name_or_path}' is not a Fast tokenizer. "
            "QA preprocessing requires offset_mapping, which is only available "
            "in Fast tokenizers."
        )
    logger.info("Tokenizer loaded: %s | vocab_size=%d",
                model_name_or_path, tokenizer.vocab_size)
    return tokenizer


def save_model_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    output_dir: str,
    suffix: str = "",
) -> None:
    """
    Save model weights and tokenizer to a versioned checkpoint directory.

    Args:
        model: The fine-tuned model.
        tokenizer: The associated tokenizer.
        output_dir: Root directory for checkpoints.
        suffix: Optional suffix (e.g. "epoch-1", "best").
    """
    name = f"checkpoint{('-' + suffix) if suffix else ''}"
    ckpt_dir = Path(output_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    logger.info("Checkpoint saved → %s", ckpt_dir)


def count_parameters(model: PreTrainedModel) -> dict[str, int]:
    """
    Count trainable and total parameters.

    Returns:
        {"total": int, "trainable": int, "frozen": int}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def _log_model_stats(model: PreTrainedModel, name: str) -> None:
    """Log model architecture summary."""
    params = count_parameters(model)
    logger.info(
        "Model: %s | params=%s (trainable=%s)",
        name,
        f"{params['total']:,}",
        f"{params['trainable']:,}",
    )
    logger.info("Architecture:\n%s", str(model)[:600] + "...")


def freeze_base_layers(
    model: PreTrainedModel,
    num_layers_to_freeze: int = 4,
) -> None:
    """
    Freeze the bottom N transformer layers for efficient fine-tuning.

    Useful for very small datasets or when computational budget is limited.

    Args:
        model: Model whose layers will be (partially) frozen.
        num_layers_to_freeze: Number of transformer blocks to freeze
            (counted from the embedding layer upward).
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        # Freeze embeddings
        if "embedding" in name:
            param.requires_grad = False
            frozen_count += param.numel()
        # Freeze lower transformer layers (layer index < threshold)
        elif "layer." in name:
            try:
                layer_idx = int(name.split("layer.")[1].split(".")[0])
                if layer_idx < num_layers_to_freeze:
                    param.requires_grad = False
                    frozen_count += param.numel()
            except (IndexError, ValueError):
                pass

    logger.info(
        "Froze %s parameters (bottom %d layers + embeddings)",
        f"{frozen_count:,}",
        num_layers_to_freeze,
    )

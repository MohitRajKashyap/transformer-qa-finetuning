"""
dataset_loader.py
=================
Responsible for loading and caching SQuAD (v1 / v2) from the
Hugging Face Hub.  Returns raw DatasetDict objects ready for
tokenisation downstream.
"""

import logging
from pathlib import Path
from typing import Optional

from datasets import DatasetDict, load_dataset

logger = logging.getLogger(__name__)


def load_squad_dataset(
    version: str = "squad",
    cache_dir: Optional[str] = "data/processed",
    overwrite_cache: bool = False,
) -> DatasetDict:
    """
    Load SQuAD v1.1 or v2.0 from the Hugging Face Hub.

    Args:
        version: "squad" for v1.1, "squad_v2" for v2.0.
        cache_dir: Directory to cache downloaded data.
        overwrite_cache: Force re-download even if cached.

    Returns:
        DatasetDict with "train" and "validation" splits.

    Raises:
        ValueError: If the version string is unrecognised.
    """
    supported = {"squad", "squad_v2"}
    if version not in supported:
        raise ValueError(
            f"Unknown dataset version '{version}'. Choose from {supported}."
        )

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset: %s (cache_dir=%s)", version, cache_dir)

    dataset = load_dataset(
        version,
        cache_dir=cache_dir,
        download_mode="force_redownload" if overwrite_cache else "reuse_dataset_if_exists",
    )

    logger.info(
        "Dataset loaded — train: %d examples | validation: %d examples",
        len(dataset["train"]),
        len(dataset["validation"]),
    )
    return dataset


def inspect_example(dataset: DatasetDict, split: str = "train", idx: int = 0) -> None:
    """Pretty-print a single dataset example for sanity checking."""
    example = dataset[split][idx]
    logger.info("--- Example [%s][%d] ---", split, idx)
    logger.info("ID      : %s", example["id"])
    logger.info("Context : %s...", example["context"][:200])
    logger.info("Question: %s", example["question"])
    answers = example.get("answers", {})
    logger.info("Answers : %s", answers.get("text", []))
    logger.info("Starts  : %s", answers.get("answer_start", []))


def get_dataset_statistics(dataset: DatasetDict) -> dict:
    """
    Compute descriptive statistics for a SQuAD DatasetDict.

    Returns:
        Dictionary with counts, average lengths, etc.
    """
    stats: dict = {}
    for split in dataset:
        ds = dataset[split]
        ctx_lens = [len(ex["context"].split()) for ex in ds]
        q_lens = [len(ex["question"].split()) for ex in ds]
        stats[split] = {
            "n_examples": len(ds),
            "avg_context_words": sum(ctx_lens) / len(ctx_lens),
            "max_context_words": max(ctx_lens),
            "avg_question_words": sum(q_lens) / len(q_lens),
        }
        logger.info(
            "[%s] n=%d | avg_ctx_words=%.1f | avg_q_words=%.1f",
            split,
            stats[split]["n_examples"],
            stats[split]["avg_context_words"],
            stats[split]["avg_question_words"],
        )
    return stats

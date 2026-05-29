"""
metrics.py
==========
SQuAD-style evaluation metrics: Exact Match and Token-level F1.

Implements the official SQuAD evaluation script logic, with additional
helpers for calibration analysis and per-example score logging.

References:
    Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension
    of Text", EMNLP 2016.
"""

import collections
import logging
import re
import string
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-level helpers (official SQuAD normalisation)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """
    Lower-case, strip punctuation, articles, and extra whitespace.

    This is the canonical normalisation used in the official SQuAD eval.
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list[str]:
    """Return normalised token list for a string."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(prediction: str, ground_truth: str) -> int:
    """Return 1 if prediction exactly matches any ground-truth answer."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 between prediction and ground truth.

    Returns:
        F1 score in [0, 1].
    """
    pred_tokens = get_tokens(prediction)
    truth_tokens = get_tokens(ground_truth)

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def get_raw_scores(
    predictions: dict[str, str],
    references: dict[str, list[str]],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute per-example EM and F1 scores.

    Args:
        predictions: {example_id: predicted_answer_string}
        references:  {example_id: [list_of_acceptable_answers]}

    Returns:
        Tuple of (exact_scores, f1_scores) dictionaries.
    """
    exact_scores: dict[str, float] = {}
    f1_scores: dict[str, float] = {}

    for qid, prediction in predictions.items():
        gold_answers = references.get(qid, [""])
        if not gold_answers:
            gold_answers = [""]

        exact_scores[qid] = max(
            compute_exact(prediction, a) for a in gold_answers
        )
        f1_scores[qid] = max(
            compute_f1(prediction, a) for a in gold_answers
        )

    return exact_scores, f1_scores


def aggregate_scores(
    exact_scores: dict[str, float],
    f1_scores: dict[str, float],
) -> dict[str, float]:
    """
    Average per-example scores to produce dataset-level metrics.

    Returns:
        {"exact_match": float, "f1": float, "total": int}
    """
    total = len(exact_scores)
    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0, "total": 0}

    return {
        "exact_match": 100.0 * sum(exact_scores.values()) / total,
        "f1": 100.0 * sum(f1_scores.values()) / total,
        "total": total,
    }


def squad_evaluate(
    predictions: dict[str, str],
    references: dict[str, list[str]],
) -> dict[str, float]:
    """
    Full SQuAD-style evaluation.

    Args:
        predictions: {example_id: answer_string}
        references:  {example_id: [answer_strings]}

    Returns:
        {"exact_match": float, "f1": float, "total": int}
    """
    exact_scores, f1_scores = get_raw_scores(predictions, references)
    results = aggregate_scores(exact_scores, f1_scores)
    logger.info(
        "Evaluation complete — EM: %.2f | F1: %.2f | N=%d",
        results["exact_match"],
        results["f1"],
        results["total"],
    )
    return results


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def compute_calibration(
    confidences: list[float],
    accuracies: list[float],
    n_bins: int = 15,
) -> dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and reliability curve data.

    Args:
        confidences: Per-example model confidence scores.
        accuracies:  Per-example correctness (1.0 = correct, 0.0 = wrong).
        n_bins: Number of equal-width probability bins.

    Returns:
        Dictionary with ECE, MCE, bin statistics for plotting.
    """
    confidences_arr = np.array(confidences, dtype=np.float64)
    accuracies_arr = np.array(accuracies, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]
    bin_mids = 0.5 * (bin_lowers + bin_uppers)

    bin_acc: list[float] = []
    bin_conf: list[float] = []
    bin_count: list[int] = []
    bin_gap: list[float] = []

    for low, up in zip(bin_lowers, bin_uppers):
        mask = (confidences_arr >= low) & (confidences_arr < up)
        count = int(mask.sum())
        if count > 0:
            avg_conf = float(confidences_arr[mask].mean())
            avg_acc = float(accuracies_arr[mask].mean())
        else:
            avg_conf = float(0.5 * (low + up))
            avg_acc = 0.0
        bin_acc.append(avg_acc)
        bin_conf.append(avg_conf)
        bin_count.append(count)
        bin_gap.append(abs(avg_acc - avg_conf))

    counts_arr = np.array(bin_count)
    gaps_arr = np.array(bin_gap)
    total = counts_arr.sum()

    ece = float((counts_arr * gaps_arr).sum() / total) if total > 0 else 0.0
    mce = float(gaps_arr.max()) if len(gaps_arr) > 0 else 0.0

    return {
        "ece": ece,
        "mce": mce,
        "bin_mids": bin_mids.tolist(),
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_count": bin_count,
        "bin_gap": bin_gap,
        "n_bins": n_bins,
    }


def softmax_confidence(start_logits: np.ndarray, end_logits: np.ndarray) -> float:
    """
    Compute a simple answer confidence from start/end logit distributions.

    Uses the product of softmax probabilities for the best span endpoints.

    Args:
        start_logits: Logit vector over token positions for span start.
        end_logits:   Logit vector over token positions for span end.

    Returns:
        Confidence score in (0, 1].
    """
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    p_start = _softmax(start_logits)
    p_end = _softmax(end_logits)
    # max joint probability as confidence proxy
    confidence = float(p_start.max() * p_end.max())
    return min(max(confidence, 1e-9), 1.0)

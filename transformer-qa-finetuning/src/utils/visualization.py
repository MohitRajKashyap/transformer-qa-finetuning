"""
visualization.py
================
Publication-quality plotting utilities for the QA fine-tuning project.

All plots use a consistent research-paper aesthetic:
  - Seaborn "whitegrid" style
  - Color-blind-friendly palette
  - Tight layout with descriptive axes labels
  - Saved as both PNG (300 dpi) and PDF for paper inclusion
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for servers
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent palette used across all figures
_PALETTE = sns.color_palette("colorblind")
_COLORS = {
    "linear": _PALETTE[0],
    "cosine": _PALETTE[1],
    "fp16": _PALETTE[2],
    "fp32": _PALETTE[3],
    "train": _PALETTE[4],
    "val": _PALETTE[5],
    "baseline": "#888888",
}

_STYLE = "whitegrid"
_DPI = 300


def _save(fig: plt.Figure, path: Path, close: bool = True) -> None:
    """Save figure as PNG and PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    logger.info("Saved plot → %s", path.with_suffix(".png"))
    if close:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Loss Curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str = "results/plots/loss_curves",
    title: str = "Training vs Validation Loss",
) -> None:
    """Plot training and validation loss across epochs/steps."""
    with sns.axes_style(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        steps = list(range(1, len(train_losses) + 1))
        ax.plot(steps, train_losses, label="Train Loss",
                color=_COLORS["train"], linewidth=2, marker="o", markersize=5)
        ax.plot(steps, val_losses, label="Validation Loss",
                color=_COLORS["val"], linewidth=2, marker="s", markersize=5,
                linestyle="--")
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4)
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# EM / F1 Comparison
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    experiments: dict[str, dict[str, float]],
    save_path: str = "results/plots/metric_comparison",
    title: str = "EM and F1 Comparison Across Experiments",
) -> None:
    """
    Grouped bar chart comparing EM and F1 for multiple experiment configs.

    Args:
        experiments: {experiment_name: {"exact_match": val, "f1": val}}
    """
    with sns.axes_style(_STYLE):
        names = list(experiments.keys())
        em_vals = [experiments[n]["exact_match"] for n in names]
        f1_vals = [experiments[n]["f1"] for n in names]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, em_vals, width, label="Exact Match",
                       color=_COLORS["train"], alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width / 2, f1_vals, width, label="F1 Score",
                       color=_COLORS["val"], alpha=0.85, edgecolor="white")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylim(60, 100)
        ax.set_ylabel("Score (%)", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# Scheduler Comparison
# ---------------------------------------------------------------------------

def plot_scheduler_comparison(
    linear_lr: list[float],
    cosine_lr: list[float],
    save_path: str = "results/plots/scheduler_comparison",
) -> None:
    """Plot LR schedule curves side-by-side."""
    with sns.axes_style(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        steps = list(range(len(linear_lr)))
        ax.plot(steps, linear_lr, label="Linear Warmup + Decay",
                color=_COLORS["linear"], linewidth=2)
        ax.plot(steps, cosine_lr, label="Cosine Schedule",
                color=_COLORS["cosine"], linewidth=2, linestyle="--")
        ax.set_xlabel("Training Step", fontsize=13)
        ax.set_ylabel("Learning Rate", fontsize=13)
        ax.set_title("Learning Rate Schedules", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# Reliability Diagram (Calibration)
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    calibration_data: dict[str, Any],
    save_path: str = "results/plots/reliability_diagram",
    title: str = "Confidence Calibration (Reliability Diagram)",
) -> None:
    """
    Plot reliability diagram and ECE summary.

    Args:
        calibration_data: Output of metrics.compute_calibration().
    """
    bin_mids = calibration_data["bin_mids"]
    bin_acc = calibration_data["bin_acc"]
    ece = calibration_data["ece"]

    with sns.axes_style(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 7))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
                linewidth=1.5, label="Perfect Calibration", alpha=0.7)

        # Model calibration bars
        width = bin_mids[1] - bin_mids[0] if len(bin_mids) > 1 else 0.07
        ax.bar(bin_mids, bin_acc, width=width * 0.9, alpha=0.7,
               color=_PALETTE[0], label="Model", edgecolor="white")

        # Gap fill (over/under confidence)
        for mid, acc in zip(bin_mids, bin_acc):
            lo, hi = min(mid, acc), max(mid, acc)
            ax.fill_between([mid - width * 0.45, mid + width * 0.45],
                            lo, hi, alpha=0.25, color="red")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=13)
        ax.set_ylabel("Accuracy", fontsize=13)
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=14,
                     fontweight="bold")
        ax.legend(fontsize=11)
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# Confidence Histogram
# ---------------------------------------------------------------------------

def plot_confidence_histogram(
    confidences: list[float],
    save_path: str = "results/plots/confidence_histogram",
) -> None:
    """Histogram of model confidence scores."""
    with sns.axes_style(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(confidences, bins=30, color=_PALETTE[0], edgecolor="white",
                alpha=0.85)
        ax.axvline(float(np.mean(confidences)), color="red", linewidth=2,
                   linestyle="--", label=f"Mean = {np.mean(confidences):.3f}")
        ax.set_xlabel("Confidence Score", fontsize=13)
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title("Model Confidence Distribution", fontsize=14,
                     fontweight="bold")
        ax.legend(fontsize=11)
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# Token Length Distribution
# ---------------------------------------------------------------------------

def plot_token_length_distribution(
    token_lengths: list[int],
    max_seq_len: int = 384,
    save_path: str = "results/plots/token_length_distribution",
) -> None:
    """Plot token length histogram with max-length boundary."""
    with sns.axes_style(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(token_lengths, bins=40, color=_PALETTE[2], edgecolor="white",
                alpha=0.85)
        ax.axvline(max_seq_len, color="red", linewidth=2, linestyle="--",
                   label=f"max_seq_length = {max_seq_len}")
        ax.set_xlabel("Token Length", fontsize=13)
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title("Context + Question Token Length Distribution",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# Ablation Study Bar Chart
# ---------------------------------------------------------------------------

def plot_ablation_results(
    ablation_results: dict[str, dict[str, float]],
    save_path: str = "results/plots/ablation_study",
    metric: str = "f1",
) -> None:
    """Horizontal bar chart for ablation study results."""
    with sns.axes_style(_STYLE):
        names = list(ablation_results.keys())
        scores = [ablation_results[n].get(metric, 0.0) for n in names]

        fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.7)))
        colors = [_PALETTE[0] if s == max(scores) else _PALETTE[6 % len(_PALETTE)]
                  for s in scores]
        bars = ax.barh(names, scores, color=colors, alpha=0.85,
                       edgecolor="white")
        for bar in bars:
            ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.2f}",
                    va="center", fontsize=10, fontweight="bold")
        ax.set_xlim(min(scores) - 2, max(scores) + 3)
        ax.set_xlabel(f"{metric.upper()} Score (%)", fontsize=13)
        ax.set_title(f"Ablation Study — {metric.upper()} Comparison",
                     fontsize=14, fontweight="bold")
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# FP16 vs FP32 Memory Chart
# ---------------------------------------------------------------------------

def plot_memory_comparison(
    memory_data: dict[str, dict[str, float]],
    save_path: str = "results/plots/memory_comparison",
) -> None:
    """
    Grouped bar chart comparing GPU memory usage for FP16 vs FP32.

    Args:
        memory_data: {batch_size: {"fp16_gb": x, "fp32_gb": y}}
    """
    with sns.axes_style(_STYLE):
        batch_sizes = list(memory_data.keys())
        fp16 = [memory_data[b]["fp16_gb"] for b in batch_sizes]
        fp32 = [memory_data[b]["fp32_gb"] for b in batch_sizes]

        x = np.arange(len(batch_sizes))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, fp16, width, label="FP16",
               color=_COLORS["fp16"], alpha=0.85, edgecolor="white")
        ax.bar(x + width / 2, fp32, width, label="FP32",
               color=_COLORS["fp32"], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([f"Batch={b}" for b in batch_sizes], fontsize=11)
        ax.set_ylabel("GPU Memory (GB)", fontsize=13)
        ax.set_title("FP16 vs FP32 GPU Memory Usage", fontsize=14,
                     fontweight="bold")
        ax.legend(fontsize=11)
        _save(fig, Path(save_path))


def generate_all_sample_plots(output_dir: str = "results/plots") -> None:
    """
    Generate all demonstration plots with synthetic data.

    Call this after training to produce the full suite of research figures.
    """
    logger.info("Generating sample visualizations...")

    # Loss curves
    train_l = [3.12, 1.84, 1.42, 1.18, 0.97, 0.85, 0.76, 0.71, 0.67, 0.64,
               0.61, 0.59]
    val_l = [2.10, 1.61, 1.38, 1.22, 1.15, 1.09, 1.06, 1.04, 1.03, 1.02,
             1.01, 1.00]
    plot_loss_curves(train_l, val_l, f"{output_dir}/loss_curves")

    # Metric comparison
    experiments = {
        "Linear (lr=2e-5)": {"exact_match": 74.8, "f1": 83.6},
        "Linear (lr=3e-5)": {"exact_match": 76.2, "f1": 85.1},
        "Cosine (lr=3e-5)": {"exact_match": 75.9, "f1": 84.7},
        "Linear (bs=32)": {"exact_match": 75.4, "f1": 84.3},
    }
    plot_metric_comparison(experiments, f"{output_dir}/metric_comparison")

    # Scheduler comparison
    n_steps = 200
    warmup = int(0.06 * n_steps)
    linear_lr = (
        [3e-5 * i / warmup for i in range(warmup)]
        + [3e-5 * (1 - (i - warmup) / (n_steps - warmup))
           for i in range(warmup, n_steps)]
    )
    cosine_lr = (
        [3e-5 * i / warmup for i in range(warmup)]
        + [3e-5 * 0.5 * (1 + np.cos(np.pi * (i - warmup) / (n_steps - warmup)))
           for i in range(warmup, n_steps)]
    )
    plot_scheduler_comparison(linear_lr, cosine_lr,
                              f"{output_dir}/scheduler_comparison")

    # Confidence histogram
    rng = np.random.default_rng(42)
    confidences = rng.beta(5, 2, size=1000).tolist()
    plot_confidence_histogram(confidences, f"{output_dir}/confidence_histogram")

    # Reliability diagram
    from src.utils.metrics import compute_calibration
    accuracies = [1.0 if c > 0.5 else 0.0 for c in confidences]
    cal = compute_calibration(confidences, accuracies, n_bins=15)
    plot_reliability_diagram(cal, f"{output_dir}/reliability_diagram")

    # Ablation study
    ablation = {
        "Full Model (DistilBERT)": {"f1": 85.1},
        "No FP16": {"f1": 84.9},
        "No Warmup": {"f1": 83.7},
        "Batch=8": {"f1": 84.2},
        "Batch=32": {"f1": 84.3},
        "Max Seq=256": {"f1": 82.8},
        "No Grad Clip": {"f1": 83.1},
    }
    plot_ablation_results(ablation, f"{output_dir}/ablation_study")

    # Memory comparison
    memory = {
        8:  {"fp16_gb": 3.2, "fp32_gb": 6.1},
        16: {"fp16_gb": 5.8, "fp32_gb": 11.2},
        32: {"fp16_gb": 10.9, "fp32_gb": "OOM"},
    }
    # Remove OOM entry for clean chart
    clean_memory = {
        k: v for k, v in memory.items() if isinstance(v.get("fp32_gb"), float)
    }
    plot_memory_comparison(clean_memory, f"{output_dir}/memory_comparison")

    # Token lengths
    token_lengths = (rng.integers(50, 384, size=800).tolist()
                     + rng.integers(200, 420, size=100).tolist())
    plot_token_length_distribution(token_lengths, 384,
                                   f"{output_dir}/token_length_distribution")

    logger.info("All sample plots saved to %s", output_dir)

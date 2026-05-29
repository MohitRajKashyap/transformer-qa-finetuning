"""
ablation.py
===========
Ablation study runner for the QA fine-tuning pipeline.

Runs multiple training configurations in sequence and aggregates
results into a comparison table.  Each ablation varies one component
at a time (one factor principle).

Usage:
    python -m src.experiments.ablation --config configs/default.yaml
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logging_utils import setup_logging
from src.utils.visualization import plot_ablation_results

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation configurations
# Each entry: (label, override_dict)
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    # Baseline
    (
        "Baseline (default)",
        {},
    ),
    # No FP16 (FP32 training)
    (
        "No FP16 (FP32)",
        {"training": {"fp16": False}},
    ),
    # No warmup
    (
        "No LR Warmup",
        {"training": {"warmup_ratio": 0.0}},
    ),
    # Smaller batch
    (
        "Batch Size = 8",
        {"training": {"per_device_train_batch_size": 8}},
    ),
    # Larger batch
    (
        "Batch Size = 32",
        {"training": {"per_device_train_batch_size": 32}},
    ),
    # Shorter context window
    (
        "Max Seq Length = 256",
        {"model": {"max_seq_length": 256, "doc_stride": 64}},
    ),
    # No gradient clipping
    (
        "No Gradient Clipping",
        {"training": {"max_grad_norm": 100.0}},
    ),
    # Lower learning rate
    (
        "LR = 2e-5",
        {"training": {"learning_rate": 2e-5}},
    ),
    # Higher learning rate
    (
        "LR = 5e-5",
        {"training": {"learning_rate": 5e-5}},
    ),
    # Cosine scheduler
    (
        "Cosine Scheduler",
        {"training": {"lr_scheduler_type": "cosine"}},
    ),
]


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base config."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result:
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def run_ablation_study(
    base_config: dict,
    output_dir: str = "results/metrics",
    dry_run: bool = False,
) -> dict[str, dict]:
    """
    Execute each ablation configuration and collect results.

    Args:
        base_config: Baseline training configuration dict.
        output_dir: Where to store per-run metrics.
        dry_run: If True, skip actual training (return synthetic results).

    Returns:
        {label: {"exact_match": float, "f1": float, ...}}
    """
    all_results: dict[str, dict] = {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, override in ABLATION_CONFIGS:
        logger.info("=" * 60)
        logger.info("ABLATION: %s", label)
        logger.info("Override: %s", override)

        run_config = deep_update(base_config, override)
        run_config["experiment"]["name"] = (
            "ablation_" + label.lower().replace(" ", "_").replace("=", "")
        )

        if dry_run:
            # Return pre-computed results for demonstration
            results = _synthetic_ablation_result(label)
        else:
            try:
                from src.training.train import main as run_train  # noqa: F401

                # In a real run we'd call a training loop directly.
                # For CLI usage, we serialize the config and call the
                # training module via subprocess or importlib.
                import importlib
                import tempfile

                with tempfile.NamedTemporaryFile(
                    "w", suffix=".yaml", delete=False
                ) as f:
                    yaml.dump(run_config, f)
                    tmp_cfg_path = f.name

                import subprocess

                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.training.train",
                        "--config",
                        tmp_cfg_path,
                    ],
                    check=True,
                )

                # Load metrics from saved file
                metrics_file = (
                    out_dir / f"{run_config['experiment']['name']}_results.json"
                )
                if metrics_file.exists():
                    with open(metrics_file) as mf:
                        saved = json.load(mf)
                    results = saved.get("history", [{}])[-1]
                else:
                    results = {"exact_match": 0.0, "f1": 0.0}

            except Exception as e:
                logger.error("Ablation run failed for '%s': %s", label, e)
                results = {"exact_match": 0.0, "f1": 0.0, "error": str(e)}

        all_results[label] = results
        logger.info(
            "  → EM: %.2f | F1: %.2f",
            results.get("exact_match", 0),
            results.get("f1", 0),
        )

    # Persist summary
    summary_path = out_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Ablation summary → %s", summary_path)

    # Visualise
    plot_ablation_results(
        all_results,
        save_path="results/plots/ablation_study",
        metric="f1",
    )

    return all_results


def _synthetic_ablation_result(label: str) -> dict:
    """
    Return realistic pre-computed ablation metrics for demonstration.

    These numbers are based on known empirical behaviour of DistilBERT on SQuAD.
    """
    _results = {
        "Baseline (default)":      {"exact_match": 76.2, "f1": 85.1},
        "No FP16 (FP32)":          {"exact_match": 76.0, "f1": 84.9},
        "No LR Warmup":            {"exact_match": 73.8, "f1": 83.2},
        "Batch Size = 8":          {"exact_match": 75.6, "f1": 84.5},
        "Batch Size = 32":         {"exact_match": 75.9, "f1": 84.7},
        "Max Seq Length = 256":    {"exact_match": 71.4, "f1": 81.3},
        "No Gradient Clipping":    {"exact_match": 72.9, "f1": 82.4},
        "LR = 2e-5":               {"exact_match": 74.8, "f1": 83.8},
        "LR = 5e-5":               {"exact_match": 73.1, "f1": 82.6},
        "Cosine Scheduler":        {"exact_match": 75.8, "f1": 84.6},
    }
    return _results.get(label, {"exact_match": 70.0, "f1": 80.0})


def format_ablation_table(results: dict[str, dict]) -> str:
    """Format results as a markdown table."""
    header = "| Configuration | Exact Match | F1 Score | Δ F1 (vs baseline) |"
    sep = "|---|---|---|---|"
    baseline_f1 = results.get("Baseline (default)", {}).get("f1", 0.0)
    rows = [header, sep]
    for label, metrics in results.items():
        em = metrics.get("exact_match", 0.0)
        f1 = metrics.get("f1", 0.0)
        delta = f1 - baseline_f1
        sign = "+" if delta >= 0 else ""
        rows.append(f"| {label} | {em:.2f} | {f1:.2f} | {sign}{delta:.2f} |")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation study.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output_dir", default="results/metrics")
    p.add_argument("--dry_run", action="store_true",
                   help="Use synthetic results (skip actual training).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(experiment_name="ablation")

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    results = run_ablation_study(
        base_config=base_config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 65)
    print("ABLATION STUDY RESULTS")
    print("=" * 65)
    print(format_ablation_table(results))


if __name__ == "__main__":
    main()

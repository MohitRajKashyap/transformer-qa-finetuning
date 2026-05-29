"""
logging_utils.py
================
Structured logging setup for the QA pipeline.

Provides:
  - Console + file logging with configurable levels
  - Rich-formatted console output
  - TensorBoard SummaryWriter factory
  - W&B initialisation helper
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(
    log_dir: str = "results/logs",
    experiment_name: str = "experiment",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Configure root logger with Rich console + optional file handler.

    Args:
        log_dir: Directory where log files are written.
        experiment_name: Prefix for the log file name.
        level: Logging level (default INFO).
        log_to_file: Write logs to a timestamped file. Default True.

    Returns:
        Configured root logger.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
    ]

    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"{experiment_name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(message)s",
        datefmt="[%X]",
    )

    # Silence overly verbose third-party loggers
    for noisy in ("transformers", "datasets", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root = logging.getLogger()
    root.info("Logging initialised → %s", log_dir)
    return root


def get_tensorboard_writer(log_dir: str, experiment_name: str):
    """
    Create a TensorBoard SummaryWriter.

    Args:
        log_dir: Base directory for TensorBoard logs.
        experiment_name: Sub-directory name for this run.

    Returns:
        SummaryWriter instance.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        run_dir = os.path.join(log_dir, experiment_name)
        writer = SummaryWriter(log_dir=run_dir)
        logging.getLogger(__name__).info(
            "TensorBoard: tensorboard --logdir %s", run_dir
        )
        return writer
    except ImportError:
        logging.getLogger(__name__).warning(
            "TensorBoard not available. Install with: pip install tensorboard"
        )
        return None


def init_wandb(config: dict, enabled: bool = False):
    """
    Initialise Weights & Biases run if enabled.

    Args:
        config: Experiment config dict to log.
        enabled: Whether W&B logging is active.

    Returns:
        wandb.Run or None.
    """
    if not enabled:
        return None
    try:
        import wandb  # type: ignore

        run = wandb.init(
            project=config.get("wandb", {}).get("project", "transformer-qa"),
            entity=config.get("wandb", {}).get("entity"),
            name=config.get("experiment", {}).get("name", "run"),
            config=config,
            tags=config.get("wandb", {}).get("tags", []),
            resume="allow",
        )
        logging.getLogger(__name__).info("W&B run: %s", wandb.run.url)
        return run
    except ImportError:
        logging.getLogger(__name__).warning(
            "W&B not available. Install with: pip install wandb"
        )
        return None


class MetricsLogger:
    """
    Lightweight wrapper that logs metrics to both TensorBoard and W&B.
    """

    def __init__(self, tb_writer=None, wandb_run=None):
        self.tb = tb_writer
        self.wb = wandb_run
        self._logger = logging.getLogger(self.__class__.__name__)

    def log(self, metrics: dict, step: int, prefix: str = "") -> None:
        """
        Log a dictionary of scalar metrics.

        Args:
            metrics: {metric_name: value} mapping.
            step: Global step index.
            prefix: Optional prefix added to all metric names.
        """
        tagged = {
            f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()
        }

        for name, value in tagged.items():
            if self.tb is not None:
                self.tb.add_scalar(name, value, global_step=step)

        if self.wb is not None:
            try:
                self.wb.log(tagged, step=step)
            except Exception:
                pass

        self._logger.info(
            "Step %6d | %s",
            step,
            " | ".join(f"{k}={v:.4f}" for k, v in tagged.items()),
        )

    def close(self) -> None:
        """Flush and close all writers."""
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self.wb is not None:
            try:
                self.wb.finish()
            except Exception:
                pass

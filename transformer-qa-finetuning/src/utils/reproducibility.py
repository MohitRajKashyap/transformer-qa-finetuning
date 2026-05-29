"""
reproducibility.py
==================
Utilities for deterministic, reproducible training runs.

Ensures that all random number generators (Python, NumPy, PyTorch, CUDA)
are seeded identically across runs so experiments can be compared fairly.
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed all RNGs for reproducibility.

    Args:
        seed: Integer seed value. Default 42.
        deterministic: If True, force CUDA into deterministic mode
            (may reduce performance but ensures reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8
        try:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        except AttributeError:
            pass

    logger.info(
        "Reproducibility seeded — seed=%d, deterministic=%s",
        seed,
        deterministic,
    )


def get_worker_init_fn(seed: int = 42):
    """
    Return a DataLoader worker init function that seeds each worker.

    Usage:
        DataLoader(..., worker_init_fn=get_worker_init_fn(seed=42))
    """

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Return the best available device.

    Args:
        prefer_gpu: Use GPU if available. Defaults to True.

    Returns:
        torch.device: 'cuda', 'mps', or 'cpu'.
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(
                "Using GPU: %s (VRAM: %.1f GB)",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )
            return device
        if torch.backends.mps.is_available():
            logger.info("Using Apple MPS device.")
            return torch.device("mps")

    logger.info("Using CPU device.")
    return torch.device("cpu")


def log_environment_info(logger_instance: Optional[logging.Logger] = None) -> None:
    """Log Python / PyTorch / CUDA environment details for the run record."""
    import sys
    import platform

    log = logger_instance or logger
    log.info("=" * 60)
    log.info("ENVIRONMENT")
    log.info("  Python    : %s", sys.version.split()[0])
    log.info("  Platform  : %s", platform.platform())
    log.info("  PyTorch   : %s", torch.__version__)
    log.info(
        "  CUDA avail: %s (v%s)",
        torch.cuda.is_available(),
        torch.version.cuda or "N/A",
    )
    if torch.cuda.is_available():
        log.info("  GPU       : %s", torch.cuda.get_device_name(0))
        log.info(
            "  VRAM      : %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    log.info("=" * 60)

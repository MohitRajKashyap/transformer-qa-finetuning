"""
scheduler.py
============
Learning rate scheduler factory with warmup support.

Supports:
  - linear warmup + linear decay
  - linear warmup + cosine decay
  - linear warmup + cosine with hard restarts
  - constant LR with warmup

All schedules are implemented via transformers.get_scheduler() so they
integrate cleanly with Accelerate and the Trainer API.
"""

import logging
import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler

logger = logging.getLogger(__name__)


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """
    Build a learning rate scheduler with optional warmup.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        scheduler_type: One of:
            "linear"              — linear warmup + linear decay
            "cosine"              — linear warmup + cosine decay
            "cosine_with_restarts"— cosine with hard restarts
            "constant_with_warmup"— flat LR after warmup
        num_warmup_steps: Steps during which LR linearly increases to peak.
        num_training_steps: Total training steps (used for decay end point).
        num_cycles: For cosine_with_restarts, number of wave cycles.

    Returns:
        A LambdaLR scheduler instance.
    """
    logger.info(
        "Building LR scheduler: type=%s | warmup=%d | total=%d",
        scheduler_type,
        num_warmup_steps,
        num_training_steps,
    )

    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


def compute_warmup_steps(
    num_training_steps: int,
    warmup_ratio: float = 0.06,
) -> int:
    """
    Compute the number of warmup steps from a ratio.

    Args:
        num_training_steps: Total training steps.
        warmup_ratio: Fraction of training to use for warmup.

    Returns:
        Integer number of warmup steps.
    """
    steps = max(1, math.floor(warmup_ratio * num_training_steps))
    logger.info(
        "Warmup steps: %d (ratio=%.2f, total=%d)",
        steps,
        warmup_ratio,
        num_training_steps,
    )
    return steps


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with separate weight decay groups.

    Weight decay is NOT applied to bias terms or LayerNorm parameters,
    following the BERT paper's implementation.

    Args:
        model: The model to optimise.
        learning_rate: Peak learning rate.
        weight_decay: L2 coefficient (applied to non-bias params).
        adam_epsilon: Numerical stability term.
        adam_beta1: First moment exponential decay rate.
        adam_beta2: Second moment exponential decay rate.

    Returns:
        torch.optim.AdamW optimizer.
    """
    no_decay = {"bias", "LayerNorm.weight"}
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )
    logger.info(
        "AdamW optimizer: lr=%.2e | wd=%.3f | eps=%.1e",
        learning_rate,
        weight_decay,
        adam_epsilon,
    )
    return optimizer


def get_lr_values(scheduler: LambdaLR, n_steps: int) -> list[float]:
    """
    Simulate and return the LR curve for plotting.

    Args:
        scheduler: A LambdaLR scheduler instance.
        n_steps: Number of steps to simulate.

    Returns:
        List of LR values at each step.
    """
    lrs: list[float] = []
    for _ in range(n_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    return lrs

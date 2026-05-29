"""
trainer.py
==========
Custom QA trainer that wraps Hugging Face Trainer with additional:
  - Early stopping on validation F1
  - Structured metric logging
  - Checkpoint versioning
  - Gradient clipping
  - FP16 mixed-precision training
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from src.data.collator import build_collator
from src.data.preprocessing import postprocess_qa_predictions
from src.models.qa_model import save_model_checkpoint
from src.training.scheduler import build_optimizer, build_scheduler, compute_warmup_steps
from src.utils.metrics import squad_evaluate

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Monitor validation F1 and signal when training should stop.

    Args:
        patience: Epochs with no improvement before stopping.
        min_delta: Minimum improvement to reset the counter.
    """

    def __init__(self, patience: int = 2, min_delta: float = 0.01) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Update state with the latest validation score.

        Args:
            score: Current validation F1.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(
                "EarlyStopping: no improvement (%d / %d)",
                self.counter,
                self.patience,
            )
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


class QATrainer:
    """
    Full training loop for extractive QA with mixed precision and early stopping.

    Designed to be framework-agnostic (does not depend on HF Trainer),
    while remaining compatible with Accelerate for multi-GPU extension.

    Args:
        model: Loaded QA model.
        tokenizer: Corresponding tokenizer.
        train_dataset: Tokenised training dataset.
        eval_dataset: Tokenised validation dataset.
        eval_examples: Raw (untokenised) validation examples.
        config: Parsed YAML configuration dictionary.
        metrics_logger: Optional MetricsLogger for TensorBoard / W&B.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        train_dataset,
        eval_dataset,
        eval_examples,
        config: dict,
        metrics_logger=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eval_examples = eval_examples
        self.config = config
        self.metrics_logger = metrics_logger

        self.training_cfg = config["training"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Training hyper-parameters
        self.num_epochs = self.training_cfg["num_train_epochs"]
        self.train_bs = self.training_cfg["per_device_train_batch_size"]
        self.eval_bs = self.training_cfg["per_device_eval_batch_size"]
        self.fp16 = self.training_cfg.get("fp16", True)
        self.max_grad_norm = self.training_cfg.get("max_grad_norm", 1.0)
        self.log_steps = self.training_cfg.get("logging_steps", 100)
        self.output_dir = config["experiment"]["output_dir"]

        self.scaler = GradScaler() if self.fp16 else None
        self.early_stopping = EarlyStopping(
            patience=self.training_cfg.get("early_stopping_patience", 2)
        )
        self.best_f1 = 0.0
        self.global_step = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        collator = build_collator(self.tokenizer)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_bs,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.training_cfg.get("dataloader_num_workers", 4),
            pin_memory=True,
        )
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.eval_bs,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.training_cfg.get("dataloader_num_workers", 4),
            pin_memory=True,
        )
        return train_loader, eval_loader

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """
        Execute the full training run.

        Returns:
            Dictionary with best metrics and training history.
        """
        train_loader, eval_loader = self._build_dataloaders()

        num_update_steps = len(train_loader) * self.num_epochs
        warmup_steps = compute_warmup_steps(
            num_update_steps,
            self.training_cfg.get("warmup_ratio", 0.06),
        )

        optimizer = build_optimizer(
            self.model,
            learning_rate=self.training_cfg["learning_rate"],
            weight_decay=self.training_cfg.get("weight_decay", 0.01),
            adam_epsilon=self.training_cfg.get("adam_epsilon", 1e-8),
        )
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=self.training_cfg.get("lr_scheduler_type", "linear"),
            num_warmup_steps=warmup_steps,
            num_training_steps=num_update_steps,
        )

        logger.info(
            "Starting training — epochs=%d | steps/epoch=%d | total=%d",
            self.num_epochs,
            len(train_loader),
            num_update_steps,
        )

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, epoch
            )
            eval_metrics = self._evaluate_epoch(eval_loader, epoch)

            elapsed = time.time() - t0
            log_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **eval_metrics,
                "elapsed_s": elapsed,
            }
            self.history.append(log_row)
            logger.info(
                "Epoch %d | loss=%.4f | EM=%.2f | F1=%.2f | time=%.0fs",
                epoch,
                train_loss,
                eval_metrics["exact_match"],
                eval_metrics["f1"],
                elapsed,
            )

            if self.metrics_logger:
                self.metrics_logger.log(
                    {"train_loss": train_loss, **eval_metrics},
                    step=epoch,
                    prefix="epoch",
                )

            # Best model tracking
            if eval_metrics["f1"] > self.best_f1:
                self.best_f1 = eval_metrics["f1"]
                save_model_checkpoint(
                    self.model,
                    self.tokenizer,
                    Path(self.output_dir) / "checkpoints",
                    suffix="best",
                )

            # Early stopping
            if self.early_stopping.step(eval_metrics["f1"]):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        return {
            "best_f1": self.best_f1,
            "history": self.history,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
    ) -> float:
        """Run a single training epoch, return mean loss."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.fp16:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            self.global_step += 1
            total_loss += loss.item()

            if self.global_step % self.log_steps == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
                if self.metrics_logger:
                    self.metrics_logger.log(
                        {"loss": loss.item(), "lr": lr},
                        step=self.global_step,
                        prefix="train",
                    )

        return total_loss / len(loader)

    def _evaluate_epoch(
        self, loader: DataLoader, epoch: int
    ) -> dict[str, float]:
        """Run evaluation, return EM and F1."""
        self.model.eval()
        all_start_logits = []
        all_end_logits = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Epoch {epoch} [eval]", leave=False):
                # Remove non-model keys before forward pass
                input_batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k in {"input_ids", "attention_mask", "token_type_ids"}
                    and v is not None
                }
                outputs = self.model(**input_batch)
                all_start_logits.append(outputs.start_logits.cpu().numpy())
                all_end_logits.append(outputs.end_logits.cpu().numpy())

        import numpy as np

        start_logits = np.concatenate(all_start_logits, axis=0)
        end_logits = np.concatenate(all_end_logits, axis=0)

        predictions, _ = postprocess_qa_predictions(
            examples=self.eval_examples,
            features=self.eval_dataset,
            raw_predictions=(start_logits, end_logits),
            tokenizer=self.tokenizer,
            n_best_size=self.config["model"].get("n_best_size", 20),
            max_answer_length=self.config["model"].get("max_answer_length", 30),
        )

        references = {
            ex["id"]: ex["answers"]["text"] for ex in self.eval_examples
        }
        return squad_evaluate(predictions, references)

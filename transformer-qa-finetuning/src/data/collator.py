"""
collator.py
===========
Custom DataCollator for Extractive QA with dynamic padding.

Dynamic padding pads each batch to the longest sequence *in that batch*,
rather than to the global max_seq_length.  This typically gives a 20–40%
speedup in training time on standard QA datasets.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForQA:
    """
    Collates QA examples with dynamic padding.

    Pads input_ids, attention_mask, and (optionally) token_type_ids to
    the maximum length in the current batch.  start_positions and
    end_positions are stacked without padding (they are scalars).

    Args:
        tokenizer: The tokeniser used for padding tokens.
        padding: Padding strategy passed to tokenizer.pad().
        max_length: Cap on padded length (ignored if padding != "max_length").
        pad_to_multiple_of: Round padded length up to nearest multiple
            (useful for Tensor Core alignment on NVIDIA GPUs; use 8).
        return_tensors: "pt" for PyTorch tensors.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    return_tensors: str = "pt"

    def __call__(
        self, features: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """
        Pad a list of feature dicts and convert to tensors.

        Fields not recognised by the tokenizer pad method (e.g.
        start_positions, end_positions) are handled separately.
        """
        # Separate QA labels from tokenizer fields
        label_names = {"start_positions", "end_positions"}
        qa_labels: dict[str, list] = {
            k: [f[k] for f in features if k in f] for k in label_names
        }
        # Remove labels from features so tokenizer.pad() does not choke
        padded_features = [
            {k: v for k, v in f.items() if k not in label_names}
            for f in features
        ]

        batch = self.tokenizer.pad(
            padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Re-attach labels as tensors
        for key, values in qa_labels.items():
            if values:
                batch[key] = torch.tensor(values, dtype=torch.long)

        return batch


def build_collator(
    tokenizer: PreTrainedTokenizerBase,
    pad_to_multiple_of: int = 8,
) -> DataCollatorForQA:
    """
    Factory function that returns a ready-to-use DataCollatorForQA.

    Args:
        tokenizer: Tokeniser for padding.
        pad_to_multiple_of: Round sequence lengths to multiples of this
            value (8 recommended for FP16 / Tensor Core efficiency).

    Returns:
        DataCollatorForQA instance.
    """
    collator = DataCollatorForQA(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    logger.info(
        "DataCollatorForQA built (dynamic padding, pad_to_multiple_of=%d)",
        pad_to_multiple_of,
    )
    return collator

"""
preprocessing.py
================
Tokenisation and feature extraction for SQuAD-style QA.

Handles the non-trivial aspects of extractive QA preprocessing:
  - Sliding window / document-stride for long contexts
  - Offset mapping to align character → token positions
  - Start / end token position computation for training targets
  - Unanswerable questions (SQuAD v2)

Design follows the official Hugging Face QA example but is refactored
into a clean, testable class with full type annotations.
"""

import logging
from typing import Any, Optional

from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class QAPreprocessor:
    """
    Converts raw SQuAD examples into model-ready features.

    Args:
        tokenizer: Fast HuggingFace tokeniser (must support offset_mapping).
        max_seq_length: Maximum total input sequence length (tokens).
        doc_stride: Overlap between consecutive windows of a long context.
        max_query_length: Maximum tokens allocated to the question.
        pad_to_max_length: Pad all examples to max_seq_length. If False,
            dynamic padding is used inside the DataCollator (preferred).
        is_training: Whether to compute answer positions for training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_query_length: int = 64,
        pad_to_max_length: bool = False,
        is_training: bool = True,
    ) -> None:
        if not tokenizer.is_fast:
            raise ValueError(
                "QAPreprocessor requires a Fast tokenizer "
                "(e.g. DistilBertTokenizerFast) for offset_mapping support."
            )
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.pad_to_max_length = pad_to_max_length
        self.is_training = is_training
        self.pad_on_right = tokenizer.padding_side == "right"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, examples: dict[str, Any]) -> dict[str, Any]:
        """
        Tokenise a batch of examples.

        Compatible with datasets.map() — receives a batch dict and returns
        a batch dict of tokenised features.
        """
        if self.is_training:
            return self._prepare_train_features(examples)
        return self._prepare_validation_features(examples)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _prepare_train_features(
        self, examples: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Tokenise training examples and compute token-level start/end positions.

        For questions longer than max_query_length, we truncate.
        For contexts longer than max_seq_length - max_query_length - 3,
        we use a sliding window with stride=doc_stride.
        """
        # Strip leading whitespace from questions
        questions = [q.lstrip() for q in examples["question"]]

        tokenized = self.tokenizer(
            questions if self.pad_on_right else examples["context"],
            examples["context"] if self.pad_on_right else questions,
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.pad_to_max_length else False,
        )

        # Map each overflow window back to its source example
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            sequence_ids = tokenized.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # Unanswerable: point both positions to CLS
            if len(answers["answer_start"]) == 0:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
                continue

            # Take the first (canonical) answer
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find the token window that covers the context
            token_start_index = 0
            while sequence_ids[token_start_index] != (
                1 if self.pad_on_right else 0
            ):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (
                1 if self.pad_on_right else 0
            ):
                token_end_index -= 1

            # Check if the answer is within this window
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                # Answer is outside this window → point to CLS
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                # Walk token indices inward to find exact span
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized["end_positions"].append(token_end_index + 1)

        return tokenized

    # ------------------------------------------------------------------
    # Validation / inference
    # ------------------------------------------------------------------

    def _prepare_validation_features(
        self, examples: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Tokenise validation examples.

        We keep the offset_mapping and example_id so we can recover
        character-level predictions after inference.
        """
        questions = [q.lstrip() for q in examples["question"]]

        tokenized = self.tokenizer(
            questions if self.pad_on_right else examples["context"],
            examples["context"] if self.pad_on_right else questions,
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.pad_to_max_length else False,
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")

        tokenized["example_id"] = []
        for i in range(len(tokenized["input_ids"])):
            sequence_ids = tokenized.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])

            # Mask out non-context tokens in offset_mapping
            tokenized["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]

        return tokenized


def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions: tuple,
    tokenizer: PreTrainedTokenizerFast,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    squad_v2: bool = False,
) -> tuple[dict[str, str], dict[str, list]]:
    """
    Convert model logits to answer strings using the offset mapping.

    This implements the standard SQuAD post-processing:
      1. Collect the top-n_best_size (start, end) combinations per window
      2. Map token positions back to character offsets in the context
      3. For SQuAD v2, apply null_score_diff_threshold

    Args:
        examples: Raw dataset split (with id, context, answers fields).
        features: Tokenised features (with offset_mapping, example_id).
        raw_predictions: Tuple of (start_logits, end_logits) arrays.
        tokenizer: The same tokeniser used during preprocessing.
        n_best_size: Candidate spans per feature.
        max_answer_length: Discard spans longer than this (tokens).
        null_score_diff_threshold: SQuAD v2 null answer threshold.
        squad_v2: Whether dataset has unanswerable questions.

    Returns:
        Tuple of (predictions_dict, nbest_dict):
          - predictions_dict: {example_id: answer_string}
          - nbest_dict: {example_id: [{"text", "score", "start_logit", "end_logit"}]}
    """
    import collections
    import numpy as np

    start_logits, end_logits = raw_predictions

    # Build index: example_id → [feature indices]
    example_id_to_index = {ex["id"]: i for i, ex in enumerate(examples)}
    features_per_example = collections.defaultdict(list)
    for i, feat in enumerate(features):
        features_per_example[example_id_to_index[feat["example_id"]]].append(i)

    predictions: dict[str, str] = {}
    nbest_predictions: dict[str, list] = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score: Optional[float] = None
        valid_answers: list[dict] = []
        context = example["context"]

        for feat_idx in feature_indices:
            start_log = start_logits[feat_idx]
            end_log = end_logits[feat_idx]
            offset_mapping = features[feat_idx]["offset_mapping"]
            feature_null_score = float(start_log[0] + end_log[0])

            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            # Build all valid spans
            start_indexes = np.argsort(start_log)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_log)[-1 : -n_best_size - 1 : -1].tolist()

            for si in start_indexes:
                for ei in end_indexes:
                    if (
                        si >= len(offset_mapping)
                        or ei >= len(offset_mapping)
                        or offset_mapping[si] is None
                        or offset_mapping[ei] is None
                    ):
                        continue
                    if ei < si or (ei - si + 1) > max_answer_length:
                        continue
                    start_char, end_char = (
                        offset_mapping[si][0],
                        offset_mapping[ei][1],
                    )
                    valid_answers.append(
                        {
                            "score": float(start_log[si] + end_log[ei]),
                            "text": context[start_char:end_char],
                            "start_logit": float(start_log[si]),
                            "end_logit": float(end_log[ei]),
                        }
                    )

        best_answer = (
            sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            if valid_answers
            else {"text": "", "score": 0.0}
        )

        if squad_v2:
            if (
                min_null_score is not None
                and min_null_score
                > best_answer["score"] + null_score_diff_threshold
            ):
                predictions[example["id"]] = ""
            else:
                predictions[example["id"]] = best_answer["text"]
        else:
            predictions[example["id"]] = best_answer["text"]

        nbest_predictions[example["id"]] = (
            sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                :n_best_size
            ]
        )

    return predictions, nbest_predictions

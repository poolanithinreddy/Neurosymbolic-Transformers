"""FEVER NLI model: DeBERTa-v3-base for 3-class fact verification.

Architecture:
  - Backbone: DeBERTa-v3-base (or any HF sequence classifier)
  - Input: "[CLS] claim [SEP] evidence [SEP]"
  - Output: 3-class logits (SUPPORTS / REFUTES / NOT ENOUGH INFO)
  - Training: standard cross-entropy + optional NST constraint loss

Why DeBERTa, not T5:
  - Fact verification is sequence classification, NOT generation.
  - DeBERTa's disentangled attention handles entity/position comparisons well.
  - DeBERTa-v3-base is a strong NLI backbone (cf. MNLI, SNLI leaderboards).
  - Generating "Supported"/"Refuted" tokens with T5 is lossy and slow.

Supports:
  - Mixed precision (fp16/bf16) via autocast
  - Gradient accumulation
  - Label smoothing
  - Constraint-augmented forward pass
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from data.fever_dataset import NUM_LABELS, LABEL2ID, ID2LABEL

logger = logging.getLogger("fever_nli")


def build_fever_model(
    model_name: str = "microsoft/deberta-v3-base",
    num_labels: int = NUM_LABELS,
    label_smoothing: float = 0.0,
    dropout: float = 0.1,
) -> tuple[AutoTokenizer, nn.Module]:
    """Build DeBERTa NLI model and tokenizer.

    Args:
        model_name: HuggingFace model name.
        num_labels: number of output classes (3 for FEVER).
        label_smoothing: label smoothing factor (0 = off).
        dropout: classifier dropout.

    Returns:
        (tokenizer, model) tuple.
    """
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=dropout,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    logger.info(f"Model loaded: {model_name} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return tokenizer, model


class FeverNLIWrapper(nn.Module):
    """Wrapper around HuggingFace sequence classifier with NST additions.

    Provides:
    - Standard forward() for cross-entropy training
    - predict() for inference with probabilities
    - constraint_forward() for NST modes (returns label probs + structured facts)
    """

    def __init__(
        self,
        model: nn.Module,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning logits and optional loss.

        Returns dict with:
            logits: [B, 3] raw logits
            probs: [B, 3] softmax probabilities
            loss: scalar cross-entropy loss (if labels provided)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [B, 3]
        probs = F.softmax(logits, dim=-1)

        result = {"logits": logits, "probs": probs}

        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)

        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Inference: return predicted labels and probabilities."""
        self.eval()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)
        pred_labels = [ID2LABEL[i.item()] for i in pred_ids]

        return {
            "logits": logits,
            "probs": probs,
            "pred_ids": pred_ids,
            "pred_labels": pred_labels,
            "confidences": probs.max(dim=-1).values,
        }

    def get_label_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Get per-label probabilities for constraint computation.

        Returns:
            p_supports: [B] probability of SUPPORTS
            p_refutes: [B] probability of REFUTES
            p_nei: [B] probability of NOT ENOUGH INFO
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1)

        return {
            "p_supports": probs[:, LABEL2ID["SUPPORTS"]],
            "p_refutes": probs[:, LABEL2ID["REFUTES"]],
            "p_nei": probs[:, LABEL2ID["NOT ENOUGH INFO"]],
            "logits": outputs.logits,
            "probs": probs,
        }

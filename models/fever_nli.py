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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    DebertaV2Tokenizer,
)

from data.fever_dataset import NUM_LABELS, LABEL2ID, ID2LABEL

logger = logging.getLogger("fever_nli")


def build_fever_model(
    model_name: str = "microsoft/deberta-v3-base",
    num_labels: int = NUM_LABELS,
    label_smoothing: float = 0.0,
    dropout: float = 0.1,
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    gradient_checkpointing: bool = False,
) -> tuple[AutoTokenizer, nn.Module]:
    """Build DeBERTa NLI model and tokenizer.

    Args:
        model_name: HuggingFace model name (supports base + large).
        num_labels: number of output classes (3 for FEVER).
        label_smoothing: label smoothing factor (0 = off).
        dropout: classifier dropout.
        use_lora: whether to apply LoRA adapters.
        lora_rank: LoRA rank (default 16).
        lora_alpha: LoRA alpha scaling (default 32).
        gradient_checkpointing: enable gradient checkpointing for memory savings.

    Returns:
        (tokenizer, model) tuple.
    """
    logger.info(f"Loading model: {model_name}")

    # DeBERTa-v3 has tiktoken/protobuf issues with fast tokenizer in
    # transformers 4.46.x — use the explicit slow tokenizer class.
    if "deberta-v3" in model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    else:
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

    # Gradient checkpointing (saves ~40% memory on large models)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # LoRA: parameter-efficient fine-tuning
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["query_proj", "key_proj", "value_proj", "dense"],
                bias="none",
            )
            model = get_peft_model(model, lora_config)

            total_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                f"LoRA applied: r={lora_rank}, alpha={lora_alpha}, "
                f"trainable={trainable/1e6:.1f}M / {total_params/1e6:.1f}M "
                f"({100*trainable/total_params:.1f}%)"
            )
        except ImportError:
            logger.warning("peft not installed; skipping LoRA. pip install peft")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {model_name} ({n_params/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable)")
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
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.label_smoothing = label_smoothing
        # Class-weighted cross-entropy for FEVER label imbalance
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights,
        )

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

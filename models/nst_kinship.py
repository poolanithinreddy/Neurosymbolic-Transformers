"""Neuro-Symbolic Model for kinship relational reasoning.

Combines:
1. Transformer encoder for text understanding.
2. Classification head for relation prediction.
3. Symbolic constraint loss based on kinship rules.
4. Optional Lagrangian adaptive constraint weighting.

Uses a small Transformer (2 layers, 128 dim) — Colab-friendly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.kinship import (
    NUM_RELATIONS,
    VOCAB_SIZE,
    check_kinship_constraint,
    kinship_constraint_loss,
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class KinshipTransformer(nn.Module):
    """Small Transformer encoder for kinship relation classification.

    Architecture:
        Embedding(VOCAB_SIZE, d_model) + PositionalEncoding
        TransformerEncoder(n_layers, n_heads, d_model, d_ff)
        MeanPool → FC(d_model, NUM_RELATIONS)

    Input: [B, seq_len] integer token IDs (character-level).
    Output: [B, NUM_RELATIONS] logits.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, NUM_RELATIONS)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        chain_lengths: list[int] | None = None,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: [B, seq_len] token IDs.
            labels: [B] ground-truth relation indices (optional).
            chain_lengths: list of chain lengths per sample (optional).

        Returns:
            Dict with logits, probs, and optional loss_task, loss_constraint.
        """
        # Create attention mask (True = ignore)
        pad_mask = input_ids == 0  # [B, seq_len]

        # Embed + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        # Transformer encoder
        # Use mask=attn_mask (additive float) instead of src_key_padding_mask
        # to avoid nested-tensor ops not supported on MPS.
        seq_len = input_ids.size(1)
        attn_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [B, S, S]
        attn_mask = attn_mask.float().masked_fill(attn_mask.bool(), float("-inf"))
        # TransformerEncoder expects [B*nhead, S, S] or broadcastable
        attn_mask = attn_mask.repeat(self.transformer.layers[0].self_attn.num_heads, 1, 1)
        x = self.transformer(x, mask=attn_mask)

        # Mean pool over non-padding positions
        mask_expanded = (~pad_mask).unsqueeze(-1).float()  # [B, seq_len, 1]
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Classify
        logits = self.classifier(x)  # [B, NUM_RELATIONS]
        probs = F.softmax(logits, dim=-1)

        result = {
            "logits": logits,
            "probs": probs,
        }

        if labels is not None:
            loss_task = F.cross_entropy(logits, labels)
            result["loss_task"] = loss_task

        if chain_lengths is not None:
            loss_constraint = kinship_constraint_loss(probs, chain_lengths)
            result["loss_constraint"] = loss_constraint

            csr, viol = check_kinship_constraint(probs, chain_lengths)
            result["csr"] = csr
        else:
            result["loss_constraint"] = torch.tensor(0.0, device=input_ids.device)
            result["csr"] = 0.0

        if labels is not None and chain_lengths is not None:
            result["loss_total"] = result["loss_task"]  # Constraint added externally

        return result

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, chain_lengths: list[int] | None = None) -> dict:
        """Inference-time prediction."""
        self.eval()
        result = self.forward(input_ids, chain_lengths=chain_lengths)
        preds = result["probs"].argmax(dim=-1)

        from data.kinship import IDX_TO_RELATION
        pred_names = [IDX_TO_RELATION[p.item()] for p in preds]

        return {
            "pred_idx": preds,
            "pred_names": pred_names,
            "probs": result["probs"],
            "csr": result.get("csr", 0.0),
        }

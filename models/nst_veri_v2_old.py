"""NST-VERI v2: Learned Multi-Task Neurosymbolic Fact Verification.

Major redesign from v1. Key changes:
  - Replaces heuristic regex-based constraints with LEARNED neural heads
  - Multi-task training: NLI + contradiction detection + evidence relevance
  - R-Drop regularization for variance reduction
  - Learned recalibration network fuses all signals at inference
  - No dependency on ConstraintEngineV2 at training time

Architecture:
    Input: [CLS] claim [SEP] evidence [SEP]
                      │
          ┌───────────┴───────────┐
          │   DeBERTa-v3 Encoder  │ (with gradient checkpointing + LoRA)
          └───────────┬───────────┘
                      │
           ┌──────────┴──────────┐
           │  [CLS] embedding h  │  +  attention pooling over sequence
           └──────────┬──────────┘
                      │
        ┌─────────────┼─────────────────┐
        │             │                 │
   ┌────┴────┐  ┌─────┴──────┐  ┌──────┴──────┐
   │ Main    │  │Contradiction│  │  Evidence   │
   │ NLI     │  │ Detection   │  │  Relevance  │
   │ Head    │  │ Head (bin)  │  │  Head (bin) │
   └────┬────┘  └─────┬──────┘  └──────┬──────┘
        │             │                 │
        └──────┬──────┴────────┬────────┘
               │               │
        ┌──────┴──────┐  ┌─────┴─────┐
        │Recalibration│  │ Contrastive│
        │  Network    │  │  Head      │
        └──────┬──────┘  └───────────┘
               │
        ┌──────┴──────┐
        │  Final      │
        │  Logits     │
        └─────────────┘

Why this works better than v1:
  - Contradiction detection is LEARNED from data, not regex heuristics
  - Evidence relevance catches "NEI" patterns the NLI head misses
  - Recalibration combines all signals adaptively per-sample
  - R-Drop reduces variance across runs
  - Contrastive loss applies to ALL samples, not just 5%
  - No noisy heuristic constraints → cleaner gradients
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.fever_dataset import NUM_LABELS, LABEL2ID, ID2LABEL

logger = logging.getLogger("nst_veri_v2")


class AttentionPool(nn.Module):
    """Attention-weighted pooling over token hidden states.

    Learns which tokens are most important for each auxiliary task,
    rather than relying solely on [CLS].
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, H)
            attention_mask: (B, L)
        Returns:
            pooled: (B, H)
        """
        scores = self.query(hidden_states).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (hidden_states * weights).sum(dim=1)  # (B, H)


class ContradictionHead(nn.Module):
    """Learned contradiction detection head.

    Predicts whether the claim-evidence pair contains a factual contradiction.
    This replaces the heuristic NumericalConstraint, NegationConstraint, etc.

    Trained with silver labels derived from REFUTES labels:
      - REFUTES → contradiction=1 (the evidence contradicts the claim)
      - SUPPORTS → contradiction=0 (evidence supports)
      - NEI → contradiction=0 (no evidence for contradiction)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Returns contradiction logit (B, 1)."""
        return self.net(hidden_state)


class EvidenceRelevanceHead(nn.Module):
    """Learned evidence relevance head.

    Predicts whether the evidence is truly relevant to the claim.
    This replaces the heuristic EntityOverlapConstraint and
    EvidenceSufficiencyConstraint.

    Trained with silver labels:
      - SUPPORTS/REFUTES → relevant=1 (evidence directly addresses claim)
      - NEI → relevant=0 (evidence doesn't address the claim)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Returns relevance logit (B, 1)."""
        return self.net(hidden_state)


class RecalibrationNetwork(nn.Module):
    """Learned recalibration that fuses NLI logits with auxiliary signals.

    Instead of heuristic fusion (alpha-blending with constraint directions),
    this network LEARNS how to combine the NLI head's output with the
    contradiction/relevance signals to produce better final logits.

    Architecture:
        Input: [NLI_logits(3) ; contradiction_logit(1) ; relevance_logit(1) ; CLS_features(H)]
        → Hidden → GELU → LayerNorm → 3-class logit correction

    The correction is scaled by a learned parameter (starts near 0).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        input_dim = 3 + 2 + hidden_dim  # NLI logits + aux logits + cls features
        inner_dim = 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.GELU(),
            nn.LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, 3),
        )
        self.scale = nn.Parameter(torch.tensor(-3.0))

        # Zero-init final layer for stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        nli_logits: torch.Tensor,
        contradiction_logit: torch.Tensor,
        relevance_logit: torch.Tensor,
        cls_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logit correction (B, 3)."""
        x = torch.cat([
            nli_logits,
            contradiction_logit,
            relevance_logit,
            cls_features,
        ], dim=-1)
        correction = self.net(x)
        scale = torch.sigmoid(self.scale)
        return correction * scale, scale


class ContrastiveHead(nn.Module):
    """Contrastive representation learning head.

    Applied to ALL training samples (not just constraint-firing ones as in v1).
    Uses class prototypes with supervised contrastive loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_classes: int = 3,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        proj_dim = min(256, hidden_dim)

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(n_classes, proj_dim) * 0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised contrastive loss over ALL samples."""
        z = F.normalize(self.projector(hidden_states), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        sim = z @ proto.T / self.temperature
        return F.cross_entropy(sim, labels)


class NSTVeriModelV2(nn.Module):
    """NST-VERI v2: Learned multi-task neurosymbolic fact verification.

    Replaces heuristic constraint engine with learned neural heads for:
      - Contradiction detection (replaces NumericalConstraint, NegationConstraint, etc.)
      - Evidence relevance (replaces EntityOverlapConstraint, EvidenceSufficiencyConstraint)
      - Learned recalibration (replaces alpha-blending heuristic fusion)

    Args:
        backbone: HuggingFace sequence classification model.
        hidden_dim: Hidden dimension of the backbone.
        label_smoothing: Label smoothing for NLI loss.
        class_weights: Optional per-class loss weights.
        dropout: Dropout rate for heads.
        contrastive_temperature: Temperature for contrastive loss.
        rdrop_alpha: Weight for R-Drop KL divergence regularization.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 768,
        label_smoothing: float = 0.05,
        class_weights: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
        contrastive_temperature: float = 0.07,
        rdrop_alpha: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.rdrop_alpha = rdrop_alpha

        # Main NLI loss
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights,
        )

        # Auxiliary heads - learned, not heuristic
        self.contradiction_head = ContradictionHead(hidden_dim, dropout)
        self.relevance_head = EvidenceRelevanceHead(hidden_dim, dropout)

        # Recalibration network - learned fusion
        self.recalibration = RecalibrationNetwork(hidden_dim, dropout)

        # Contrastive head - applied to ALL samples
        self.contrastive = ContrastiveHead(
            hidden_dim, NUM_LABELS, contrastive_temperature
        )

        # Attention pooling for auxiliary heads (complement to [CLS])
        self.attn_pool = AttentionPool(hidden_dim)

    def _get_hidden_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract [CLS] hidden, attention-pooled hidden, and main logits.

        Returns:
            (cls_hidden, pooled_hidden, logits)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits  # (B, 3)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]  # (B, L, H)
            cls_hidden = last_hidden[:, 0, :]  # (B, H)
            pooled_hidden = self.attn_pool(last_hidden, attention_mask)  # (B, H)
        else:
            cls_hidden = torch.zeros(
                logits.shape[0], self.hidden_dim,
                device=logits.device, dtype=logits.dtype,
            )
            pooled_hidden = cls_hidden

        return cls_hidden, pooled_hidden, logits

    def _forward_once(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Single forward pass returning all head outputs."""
        cls_hidden, pooled_hidden, nli_logits = self._get_hidden_and_logits(
            input_ids, attention_mask
        )

        # Auxiliary heads operate on pooled representation
        # (combines [CLS] with attention-pooled for richer features)
        combined_hidden = cls_hidden + pooled_hidden  # element-wise sum

        contradiction_logit = self.contradiction_head(combined_hidden)  # (B, 1)
        relevance_logit = self.relevance_head(combined_hidden)  # (B, 1)

        # Recalibration: learned fusion of all signals
        correction, recalib_scale = self.recalibration(
            nli_logits, contradiction_logit, relevance_logit, cls_hidden
        )
        final_logits = nli_logits + correction

        return {
            "nli_logits": nli_logits,
            "final_logits": final_logits,
            "probs": F.softmax(final_logits, dim=-1),
            "cls_hidden": cls_hidden,
            "combined_hidden": combined_hidden,
            "contradiction_logit": contradiction_logit.squeeze(-1),
            "relevance_logit": relevance_logit.squeeze(-1),
            "recalib_scale": recalib_scale,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        beta_contradiction: float = 0.3,
        beta_relevance: float = 0.3,
        gamma_contrastive: float = 0.1,
        use_rdrop: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with multi-task loss computation.

        Args:
            input_ids: (B, L) token ids.
            attention_mask: (B, L) attention mask.
            labels: (B,) ground truth NLI labels.
            beta_contradiction: Weight for contradiction detection loss.
            beta_relevance: Weight for evidence relevance loss.
            gamma_contrastive: Weight for contrastive loss.
            use_rdrop: Whether to apply R-Drop regularization.

        Returns:
            dict with logits, probs, losses, and diagnostics.
        """
        out1 = self._forward_once(input_ids, attention_mask)

        result = {
            "logits": out1["final_logits"],
            "nli_logits": out1["nli_logits"],
            "probs": out1["probs"],
            "contradiction_logit": out1["contradiction_logit"],
            "relevance_logit": out1["relevance_logit"],
            "recalib_scale": out1["recalib_scale"],
        }

        if labels is not None:
            # ── Loss 1: Main NLI (always active) ──
            loss_nli = self.loss_fn(out1["final_logits"], labels)
            result["loss_nli"] = loss_nli

            # ── Loss 2: Contradiction detection ──
            # Silver label: REFUTES=1, SUPPORTS/NEI=0
            contradiction_target = (labels == LABEL2ID["REFUTES"]).float()
            loss_contradiction = F.binary_cross_entropy_with_logits(
                out1["contradiction_logit"],
                contradiction_target,
            )
            result["loss_contradiction"] = loss_contradiction

            # ── Loss 3: Evidence relevance ──
            # Silver label: SUPPORTS/REFUTES=1 (evidence relevant), NEI=0
            relevance_target = (labels != LABEL2ID["NOT ENOUGH INFO"]).float()
            loss_relevance = F.binary_cross_entropy_with_logits(
                out1["relevance_logit"],
                relevance_target,
            )
            result["loss_relevance"] = loss_relevance

            # ── Loss 4: Contrastive (ALL samples) ──
            loss_contrastive = self.contrastive(out1["cls_hidden"], labels)
            result["loss_contrastive"] = loss_contrastive

            # ── Loss 5: R-Drop regularization ──
            loss_rdrop = torch.tensor(0.0, device=labels.device)
            if use_rdrop and self.rdrop_alpha > 0:
                out2 = self._forward_once(input_ids, attention_mask)
                # KL divergence between two forward passes (dropout noise)
                p = F.log_softmax(out1["final_logits"], dim=-1)
                q = F.log_softmax(out2["final_logits"], dim=-1)
                loss_rdrop = 0.5 * (
                    F.kl_div(p, q.exp(), reduction="batchmean") +
                    F.kl_div(q, p.exp(), reduction="batchmean")
                )
                # Also use second pass NLI loss (average both)
                loss_nli_2 = self.loss_fn(out2["final_logits"], labels)
                loss_nli = 0.5 * (loss_nli + loss_nli_2)
                result["loss_nli"] = loss_nli
            result["loss_rdrop"] = loss_rdrop

            # ── Total loss ──
            total_loss = (
                loss_nli
                + beta_contradiction * loss_contradiction
                + beta_relevance * loss_relevance
                + gamma_contrastive * loss_contrastive
                + self.rdrop_alpha * loss_rdrop
            )
            result["loss"] = total_loss

        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        claims: list[str] | None = None,
        evidences: list[str] | None = None,
        use_symbolic_fusion: bool = True,
        constraint_alpha: float = 0.1,
    ) -> dict[str, Any]:
        """Inference: return predictions with learned recalibration.

        The recalibration network automatically adjusts predictions
        based on contradiction/relevance signals. No heuristic fusion needed.

        Optionally applies symbolic constraint fusion (v1 compatibility).
        """
        self.eval()
        out = self._forward_once(input_ids, attention_mask)
        probs = out["probs"]

        # Optional: also apply symbolic constraints for extra signal
        if use_symbolic_fusion and claims is not None and evidences is not None:
            try:
                from symbolic.constraints_v2 import ConstraintEngineV2
                engine = ConstraintEngineV2()
                signals = engine.evaluate_batch(claims, evidences)
                fires = signals["fires"].to(probs.device).float()
                conf = signals["confidence"].to(probs.device)
                direction = signals["direction"].to(probs.device)

                # Only use high-precision constraints with uncertainty gating
                high_precision_k = {0, 3, 4}
                K = fires.shape[1]
                max_prob = probs.max(dim=-1).values
                uncertain = max_prob < 0.8

                total_correction = torch.zeros_like(probs)
                for k in range(K):
                    if k not in high_precision_k:
                        continue
                    mask_k = (fires[:, k] > 0.5) & (conf[:, k] > 0.6) & uncertain
                    if mask_k.any():
                        conf_k = conf[mask_k, k].unsqueeze(-1)
                        dir_k = direction[mask_k, k]
                        total_correction[mask_k] += constraint_alpha * conf_k * (
                            dir_k - probs[mask_k]
                        )

                probs = (probs + total_correction).clamp(min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            except Exception:
                pass  # graceful fallback if constraint engine unavailable

        pred_ids = probs.argmax(dim=-1)

        return {
            "logits": out["final_logits"],
            "probs": probs,
            "pred_ids": pred_ids,
            "pred_labels": [ID2LABEL[i.item()] for i in pred_ids],
            "confidences": probs.max(dim=-1).values,
            "contradiction_score": torch.sigmoid(out["contradiction_logit"]),
            "relevance_score": torch.sigmoid(out["relevance_logit"]),
            "recalib_scale": out["recalib_scale"],
        }

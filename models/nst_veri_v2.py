"""NST-VERI v2: Learned Multi-Task Neurosymbolic Fact Verification.

Major redesign. Key architectural decisions:

  1. AUXILIARY HEADS LEARN *DIFFERENT* FEATURES FROM THE NLI HEAD.
     - ContradictionHead uses its own AttentionPool over hidden states
       (not CLS). Learns complementary entity-level conflict patterns.
     - EvidenceRelevanceHead similarly uses separate attention pooling.

  2. RECALIBRATION IS SIGNAL-ONLY (NO RAW CLS).
     - Input: [NLI_logits(3) + contradiction_prob(1) + relevance_prob(1)
              + max_prob(1) + entropy(1)] = 7-dim
     - This is a TINY correction network (7→32→3), not a second classifier.
     - Forced to be a genuine correction based on structured diagnostics.

  3. REFUTES-FOCUSED TRAINING.
     - Focal loss with per-class gamma: higher for REFUTES.
     - REFUTES is the accuracy bottleneck (79.4% vs 90%+ SUPPORTS).

  4. SYMMETRIC R-DROP.
     - All losses (NLI + aux) averaged across both forward passes.

  5. CONTRASTIVE ON ALL SAMPLES.
     - Supervised contrastive with class prototypes, every batch.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.fever_dataset import NUM_LABELS, LABEL2ID, ID2LABEL

logger = logging.getLogger("nst_veri_v2")


# ── Focal Cross-Entropy ────────────────────────────────────────

class FocalCrossEntropy(nn.Module):
    """Focal loss with per-class gamma and label smoothing.

    Upweights hard examples. Per-class gamma allows stronger focus on
    the REFUTES class (the accuracy bottleneck).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        weight: Optional[torch.Tensor] = None,
        class_gamma: Optional[dict[int, float]] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)
        self.class_gamma = class_gamma or {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[-1]
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(logits, self.label_smoothing / (n_classes - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth = F.one_hot(targets, n_classes).float()

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        p_t = (probs * smooth).sum(dim=-1)

        gamma_t = torch.full_like(p_t, self.gamma)
        for cls_id, g in self.class_gamma.items():
            gamma_t[targets == cls_id] = g

        focal_weight = (1.0 - p_t) ** gamma_t
        loss = -(smooth * log_probs).sum(dim=-1)

        if self.weight is not None:
            w = self.weight[targets]
            loss = loss * w

        loss = loss * focal_weight
        return loss.mean()


# ── Attention Pool ─────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Attention-weighted pooling over token hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        scores = self.query(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


# ── Auxiliary Heads ────────────────────────────────────────────

class ContradictionHead(nn.Module):
    """Learned contradiction detector with SEPARATE attention pooling.

    Uses its own AttentionPool to learn different token weighting from CLS.
    Silver labels: REFUTES → 1, else → 0.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pool = AttentionPool(hidden_dim)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pool(hidden_states, attention_mask)
        return self.net(pooled)


class EvidenceRelevanceHead(nn.Module):
    """Learned evidence relevance detector with SEPARATE attention pooling.

    Silver labels: SUPPORTS/REFUTES → 1, NEI → 0.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pool = AttentionPool(hidden_dim)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pool(hidden_states, attention_mask)
        return self.net(pooled)


# ── Signal-Only Recalibration ──────────────────────────────────

class RecalibrationNetwork(nn.Module):
    """Lightweight recalibration from STRUCTURED SIGNALS ONLY.

    Input: [NLI_logits(3) + p_contra(1) + p_relevant(1) + max_prob(1)
            + entropy(1)] = 7-dim
    Output: 3-class logit correction, scaled by a learned parameter.

    KEY: No raw CLS features → cannot become a second classifier.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )
        self.scale = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        nli_logits: torch.Tensor,
        contradiction_prob: torch.Tensor,
        relevance_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(nli_logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)

        x = torch.cat([
            nli_logits,
            contradiction_prob.unsqueeze(-1) if contradiction_prob.dim() == 1 else contradiction_prob,
            relevance_prob.unsqueeze(-1) if relevance_prob.dim() == 1 else relevance_prob,
            max_prob,
            entropy,
        ], dim=-1)

        correction = self.net(x)
        scale = torch.sigmoid(self.scale)
        return correction * scale, scale


# ── Contrastive Head ───────────────────────────────────────────

class ContrastiveHead(nn.Module):
    """Supervised contrastive via class prototypes. Applied to ALL samples."""

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

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.projector(hidden_states), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        sim = z @ proto.T / self.temperature
        return F.cross_entropy(sim, labels)


# ── Main Model ─────────────────────────────────────────────────

class NSTVeriModelV2(nn.Module):
    """NST-VERI v2: learned multi-task neurosymbolic fact verification.

    Key design changes from v2-old:
      1. Aux heads use SEPARATE attention pooling (complementary features)
      2. RecalibrationNetwork sees only structured signals (7-dim), not CLS
      3. Focal loss with higher gamma for REFUTES (bottleneck class)
      4. Symmetric R-Drop (all losses averaged across both passes)
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
        focal_gamma: float = 2.0,
        refutes_gamma: float = 3.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.rdrop_alpha = rdrop_alpha

        refutes_id = LABEL2ID["REFUTES"]
        self.loss_fn = FocalCrossEntropy(
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
            weight=class_weights,
            class_gamma={refutes_id: refutes_gamma},
        )

        self.contradiction_head = ContradictionHead(hidden_dim, dropout)
        self.relevance_head = EvidenceRelevanceHead(hidden_dim, dropout)
        self.recalibration = RecalibrationNetwork()
        self.contrastive = ContrastiveHead(
            hidden_dim, NUM_LABELS, contrastive_temperature
        )

    def _get_backbone_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run backbone → (last_hidden_states, cls_hidden, nli_logits)."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
            cls_hidden = last_hidden[:, 0, :]
        else:
            last_hidden = torch.zeros(
                logits.shape[0], input_ids.shape[1], self.hidden_dim,
                device=logits.device, dtype=logits.dtype,
            )
            cls_hidden = torch.zeros(
                logits.shape[0], self.hidden_dim,
                device=logits.device, dtype=logits.dtype,
            )

        return last_hidden, cls_hidden, logits

    def _forward_once(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        last_hidden, cls_hidden, nli_logits = self._get_backbone_outputs(
            input_ids, attention_mask
        )

        contradiction_logit = self.contradiction_head(last_hidden, attention_mask)
        relevance_logit = self.relevance_head(last_hidden, attention_mask)

        contra_prob = torch.sigmoid(contradiction_logit.squeeze(-1))
        rel_prob = torch.sigmoid(relevance_logit.squeeze(-1))

        correction, recalib_scale = self.recalibration(
            nli_logits, contra_prob, rel_prob
        )
        final_logits = nli_logits + correction

        return {
            "nli_logits": nli_logits,
            "final_logits": final_logits,
            "probs": F.softmax(final_logits, dim=-1),
            "cls_hidden": cls_hidden,
            "last_hidden": last_hidden,
            "contradiction_logit": contradiction_logit.squeeze(-1),
            "relevance_logit": relevance_logit.squeeze(-1),
            "contradiction_prob": contra_prob,
            "relevance_prob": rel_prob,
            "recalib_scale": recalib_scale,
        }

    def _compute_losses(
        self,
        out: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all task losses for a single forward pass output."""
        loss_nli = self.loss_fn(out["final_logits"], labels)

        contra_target = (labels == LABEL2ID["REFUTES"]).float()
        loss_contra = F.binary_cross_entropy_with_logits(
            out["contradiction_logit"], contra_target,
        )

        rel_target = (labels != LABEL2ID["NOT ENOUGH INFO"]).float()
        loss_rel = F.binary_cross_entropy_with_logits(
            out["relevance_logit"], rel_target,
        )

        loss_contrastive = self.contrastive(out["cls_hidden"], labels)

        return {
            "loss_nli": loss_nli,
            "loss_contradiction": loss_contra,
            "loss_relevance": loss_rel,
            "loss_contrastive": loss_contrastive,
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
        """Forward pass with multi-task loss.

        When use_rdrop=True, ALL losses are averaged across two passes
        for symmetric regularization.
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
            losses1 = self._compute_losses(out1, labels)

            loss_rdrop = torch.tensor(0.0, device=labels.device)
            if use_rdrop and self.rdrop_alpha > 0:
                out2 = self._forward_once(input_ids, attention_mask)
                losses2 = self._compute_losses(out2, labels)

                # Average ALL losses symmetrically
                for key in losses1:
                    losses1[key] = 0.5 * (losses1[key] + losses2[key])

                # Symmetric KL divergence
                p = F.log_softmax(out1["final_logits"], dim=-1)
                q = F.log_softmax(out2["final_logits"], dim=-1)
                loss_rdrop = 0.5 * (
                    F.kl_div(p, q.exp(), reduction="batchmean") +
                    F.kl_div(q, p.exp(), reduction="batchmean")
                )

            result.update(losses1)
            result["loss_rdrop"] = loss_rdrop

            total_loss = (
                losses1["loss_nli"]
                + beta_contradiction * losses1["loss_contradiction"]
                + beta_relevance * losses1["loss_relevance"]
                + gamma_contrastive * losses1["loss_contrastive"]
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
        """Inference with learned recalibration + optional symbolic fusion."""
        self.eval()
        out = self._forward_once(input_ids, attention_mask)
        probs = out["probs"]

        if use_symbolic_fusion and claims is not None and evidences is not None:
            try:
                from symbolic.constraints_v2 import ConstraintEngineV2
                engine = ConstraintEngineV2()
                signals = engine.evaluate_batch(claims, evidences)
                fires = signals["fires"].to(probs.device).float()
                conf = signals["confidence"].to(probs.device)
                direction = signals["direction"].to(probs.device)

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
            except Exception as e:
                logger.warning(f"Symbolic fusion failed: {e}")

        pred_ids = probs.argmax(dim=-1)

        return {
            "logits": out["final_logits"],
            "probs": probs,
            "pred_ids": pred_ids,
            "pred_labels": [ID2LABEL[i.item()] for i in pred_ids],
            "confidences": probs.max(dim=-1).values,
            "contradiction_score": out["contradiction_prob"],
            "relevance_score": out["relevance_prob"],
            "recalib_scale": out["recalib_scale"],
        }

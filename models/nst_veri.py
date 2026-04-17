"""NST-VERI: Verification-Enhanced Reasoning Integration for Fact Verification.

The flagship neurosymbolic model of this project. Combines:
  1. DeBERTa-v3 backbone (base or large) with optional LoRA
  2. Auxiliary verification heads (bridge neural→symbolic)
  3. Verification-conditioned residual correction (zero-init)
  4. Per-sample adaptive lambda via uncertainty-aware gating
  5. Supervised contrastive loss on high-confidence constraint examples

Architecture:
    Input: [CLS] claim [SEP] evidence [SEP]
                      │
          ┌───────────┴───────────┐
          │   DeBERTa-v3 Encoder  │ (with gradient checkpointing + LoRA)
          └───────────┬───────────┘
                      │
           ┌──────────┴──────────┐
           │  [CLS] embedding h  │
           └──────────┬──────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────┴────┐  ┌─────┴─────┐  ┌───┴────┐
   │ Main    │  │ Verif.    │  │Contras.│
   │ NLI     │  │ Heads     │  │ Head   │
   │ Head    │  │ (K aux)   │  │        │
   └────┬────┘  └─────┬─────┘  └───┬────┘
        │             │             │
        │      ┌──────┴──────┐      │
        │      │  Residual   │      │
        │      │  Correction │      │
        │      └──────┬──────┘      │
        │             │             │
        └──────┬──────┘             │
               │                    │
        ┌──────┴──────┐      ┌─────┴─────┐
        │  Final      │      │ Prototype │
        │  Logits     │      │ Loss      │
        └─────────────┘      └───────────┘

Training phases:
  Phase 1 (Epoch 0):           β=1.0, γ=0.0, λ_max=0.0  [Pure NLI + aux]
  Phase 2 (Epoch 1):           β=1.0, γ=0.1, λ_max=0.0  [Add contrastive]
  Phase 3 (Epochs 2-N):        β=1.0, γ=0.1, λ_max→0.3  [Warmup constraints]
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.fever_dataset import NUM_LABELS, LABEL2ID, ID2LABEL
from models.verification_heads import VerificationHeads
from models.contrastive_head import ContrastiveConstraintHead

logger = logging.getLogger("nst_veri")


class NSTVeriModel(nn.Module):
    """NST-VERI: full neurosymbolic fact verification model.

    This wraps a HuggingFace sequence classifier and adds:
      - Verification heads (auxiliary task)
      - Residual correction from verification signals
      - Contrastive head for representation shaping
      - Clean interfaces for the multi-phase training loop

    Args:
        backbone: HuggingFace model (AutoModelForSequenceClassification).
        hidden_dim: Backbone's hidden dimension.
        n_constraints: Number of constraint / verification heads.
        label_smoothing: Label smoothing for main NLI loss.
        class_weights: Per-class loss weights for imbalanced FEVER.
        dropout: Dropout for verification/contrastive heads.
        contrastive_temperature: Temperature for contrastive loss.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 768,
        n_constraints: int = 6,
        label_smoothing: float = 0.05,
        class_weights: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.n_constraints = n_constraints

        # Main NLI loss
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights,
        )

        # Verification heads (auxiliary + residual correction)
        self.verification = VerificationHeads(
            hidden_dim=hidden_dim,
            n_constraints=n_constraints,
            dropout=dropout,
        )

        # Contrastive head for representation shaping
        self.contrastive = ContrastiveConstraintHead(
            hidden_dim=hidden_dim,
            n_classes=NUM_LABELS,
            temperature=contrastive_temperature,
        )

    def _get_cls_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract [CLS] hidden state and main logits from backbone.

        Returns:
            (hidden_state, logits) where hidden_state is (B, H)
            and logits is (B, num_labels).
        """
        # Access the underlying transformer (handles PEFT wrapping)
        model = self.backbone
        # For PEFT models, get base model's transformer
        if hasattr(model, 'base_model'):
            # PEFT-wrapped
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        logits = outputs.logits  # (B, 3)

        # Get [CLS] hidden state from the last hidden layer
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_state = outputs.hidden_states[-1][:, 0, :]  # (B, H)
        else:
            # Fallback: use outputs directly (some models expose differently)
            logger.warning("Could not get hidden_states; falling back to pooler or logits")
            hidden_state = torch.zeros(
                logits.shape[0], self.hidden_dim,
                device=logits.device, dtype=logits.dtype,
            )

        return hidden_state, logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        constraint_signals: Optional[dict[str, torch.Tensor]] = None,
        phase: int = 1,
        beta: float = 1.0,
        gamma: float = 0.0,
        adaptive_lambda: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for NST-VERI.

        Args:
            input_ids: Tokenised input (B, L).
            attention_mask: Attention mask (B, L).
            labels: Ground truth labels (B,) or None.
            constraint_signals: Output of ConstraintEngineV2.evaluate_batch().
            phase: Training phase (1=NLI+aux, 2=+contrastive, 3=+constraints).
            beta: Weight for auxiliary verification loss.
            gamma: Weight for contrastive loss.
            adaptive_lambda: Output of AdaptiveLambdaModule (lambda_per_sample, gate_weights).

        Returns:
            dict with: logits, probs, loss, loss_nli, loss_aux, loss_contrastive,
                       loss_constraint, verification_probs, hidden_state
        """
        # Get [CLS] hidden state and main logits
        hidden_state, main_logits = self._get_cls_hidden(input_ids, attention_mask)

        # Verification heads: auxiliary predictions + residual correction
        verif_out = self.verification(hidden_state)
        logit_correction = verif_out["logit_correction"]  # (B, 3)

        # Apply residual correction to main logits
        final_logits = main_logits + logit_correction
        probs = F.softmax(final_logits, dim=-1)

        result = {
            "logits": final_logits,
            "main_logits": main_logits,
            "probs": probs,
            "hidden_state": hidden_state,
            "verification_probs": verif_out["verification_probs"],
            "verification_logits": verif_out["verification_logits"],
            "residual_scale": verif_out["residual_scale"],
        }

        if labels is not None:
            # ── Loss 1: Main NLI (always active) ──
            loss_nli = self.loss_fn(final_logits, labels)
            result["loss_nli"] = loss_nli

            # ── Loss 2: Auxiliary verification (phases 1+) ──
            loss_aux = torch.tensor(0.0, device=labels.device)
            if constraint_signals is not None and beta > 0:
                fires = constraint_signals["fires"].to(labels.device)
                confidence = constraint_signals["confidence"].to(labels.device)
                loss_aux = self.verification.verification_loss(
                    verif_out["verification_logits"],
                    targets=confidence,  # Soft targets from constraint confidence
                    mask=fires,
                )
            result["loss_aux"] = loss_aux

            # ── Loss 3: Contrastive (phases 2+) ──
            loss_contrastive = torch.tensor(0.0, device=labels.device)
            if gamma > 0 and phase >= 2:
                # Only apply contrastive loss on high-confidence constraint examples
                if constraint_signals is not None:
                    conf = constraint_signals["confidence"].to(labels.device)
                    # Mask: at least one constraint fires with confidence > 0.3
                    high_conf_mask = (conf.max(dim=-1).values > 0.3).float()
                else:
                    high_conf_mask = None
                loss_contrastive = self.contrastive(
                    hidden_state, labels, mask=high_conf_mask
                )
            result["loss_contrastive"] = loss_contrastive

            # ── Loss 4: Constraint (phase 3) ──
            loss_constraint = torch.tensor(0.0, device=labels.device)
            if phase >= 3 and constraint_signals is not None and adaptive_lambda is not None:
                from symbolic.constraints_v2 import ConstraintEngineV2
                # Use the engine's compute_constraint_loss
                # We need to pass through gate_weights from adaptive lambda
                gate_weights = adaptive_lambda.get("gate_weights", None)
                lambda_per_sample = adaptive_lambda.get("lambda_per_sample", None)

                # Per-constraint KL loss
                fires = constraint_signals["fires"].to(labels.device).float()
                confidence_t = constraint_signals["confidence"].to(labels.device)
                direction = constraint_signals["direction"].to(labels.device)

                B, K = fires.shape
                if gate_weights is None:
                    gate_weights = torch.ones(B, K, device=labels.device)

                # KL(direction || probs) per constraint
                log_probs_exp = (probs + 1e-8).log().unsqueeze(1).expand(-1, K, -1)
                log_dir = (direction + 1e-8).log()
                kl = (direction * (log_dir - log_probs_exp)).sum(dim=-1)  # (B, K)
                kl = kl.clamp(max=10.0)  # Prevent huge gradients from bad constraints

                weighted_kl = kl * fires * confidence_t * gate_weights  # (B, K)
                per_sample_constraint = weighted_kl.sum(dim=-1)  # (B,)

                if lambda_per_sample is not None:
                    loss_constraint = (lambda_per_sample * per_sample_constraint).mean()
                else:
                    loss_constraint = per_sample_constraint.mean()

            result["loss_constraint"] = loss_constraint

            # ── Total loss ──
            total_loss = loss_nli + beta * loss_aux + gamma * loss_contrastive + loss_constraint
            result["loss"] = total_loss

        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Inference: return predicted labels and probabilities."""
        self.eval()
        hidden_state, main_logits = self._get_cls_hidden(input_ids, attention_mask)
        verif_out = self.verification(hidden_state)
        final_logits = main_logits + verif_out["logit_correction"]
        probs = F.softmax(final_logits, dim=-1)
        pred_ids = final_logits.argmax(dim=-1)

        return {
            "logits": final_logits,
            "probs": probs,
            "pred_ids": pred_ids,
            "pred_labels": [ID2LABEL[i.item()] for i in pred_ids],
            "confidences": probs.max(dim=-1).values,
            "verification_probs": verif_out["verification_probs"],
            "residual_scale": verif_out["residual_scale"],
        }

    def get_label_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Get per-label probabilities for constraint computation."""
        hidden_state, main_logits = self._get_cls_hidden(input_ids, attention_mask)
        verif_out = self.verification(hidden_state)
        final_logits = main_logits + verif_out["logit_correction"]
        probs = F.softmax(final_logits, dim=-1)

        return {
            "p_supports": probs[:, LABEL2ID["SUPPORTS"]],
            "p_refutes": probs[:, LABEL2ID["REFUTES"]],
            "p_nei": probs[:, LABEL2ID["NOT ENOUGH INFO"]],
            "logits": final_logits,
            "probs": probs,
        }

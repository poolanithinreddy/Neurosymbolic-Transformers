"""Neuro-Symbolic Model for digit addition.

Combines:
1. Neural perception (DigitCNN) — predicts digit classes from images.
2. Symbolic constraint (sum_constraint_soft) — enforces a + b = c via
   differentiable t-norm penalty.
3. Optional hard-constraint repair (Z3) at inference time.

Ablation modes:
- "neural":  pure neural — digit classification + MLP sum head, no constraints.
- "soft":    digit classification + differentiable sum constraint loss.
- "hard":    soft-trained model + Z3 hard-constraint repair at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.perception import DigitCNN, DigitEncoder, SumHead
from symbolic.constraint_solver import (
    constraint_satisfaction_rate,
    hard_constraint_batch,
    sum_constraint_soft,
)


class NSTDigitAddModel(nn.Module):
    """Neuro-symbolic digit addition model.

    Contains a shared CNN encoder for both digits and either:
    - A separate MLP sum head (pure neural mode), or
    - A symbolic constraint layer (soft/hard mode).

    In soft mode, the model predicts digit probabilities and uses discrete
    convolution to compute the expected sum distribution, with a KL-divergence
    loss to enforce the arithmetic constraint.
    """

    def __init__(self, mode: str = "soft", num_digits: int = 10, num_sums: int = 19):
        """Initialise the model.

        Args:
            mode: one of "neural", "soft", "hard", "lagrangian".
            num_digits: number of digit classes (default 10 for 0-9).
            num_sums: number of sum classes (default 19 for 0-18).
        """
        super().__init__()
        assert mode in ("neural", "soft", "hard", "lagrangian"), f"Unknown mode: {mode}"
        self.mode = mode
        self.num_digits = num_digits
        self.num_sums = num_sums

        # Shared perception encoder
        self.digit_cnn = DigitCNN(num_classes=num_digits)

        # Feature extractor (for neural mode's sum head)
        self.digit_encoder = DigitEncoder()

        # Neural-only sum head (used in 'neural' mode)
        if mode == "neural":
            self.sum_head = SumHead(in_dim=128, num_classes=num_sums)
        else:
            # In soft/hard/lagrangian mode, sum is computed symbolically
            self.sum_head = None

    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        digit_a: torch.Tensor | None = None,
        digit_b: torch.Tensor | None = None,
        sum_target: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass.

        Args:
            img_a: [B, 1, 28, 28] image of digit a.
            img_b: [B, 1, 28, 28] image of digit b.
            digit_a: [B] ground-truth digit a labels (optional, for loss).
            digit_b: [B] ground-truth digit b labels (optional, for loss).
            sum_target: [B] ground-truth sum labels (optional, for loss).

        Returns:
            Dict with keys:
                logits_a: [B, 10] digit a logits.
                logits_b: [B, 10] digit b logits.
                probs_a: [B, 10] digit a probabilities.
                probs_b: [B, 10] digit b probabilities.
                logits_sum: [B, 19] sum logits/probs.
                loss_digit: scalar digit classification loss (if targets given).
                loss_constraint: scalar constraint loss (soft/hard mode).
                loss_total: scalar total loss.
                csr: constraint satisfaction rate.
        """
        # Digit classification
        logits_a = self.digit_cnn(img_a)  # [B, 10]
        logits_b = self.digit_cnn(img_b)  # [B, 10]
        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)

        result = {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "probs_a": probs_a,
            "probs_b": probs_b,
        }

        # Compute sum predictions based on mode
        if self.mode == "neural":
            # Pure neural: use feature encoder + MLP
            feat_a = self.digit_encoder(img_a)
            feat_b = self.digit_encoder(img_b)
            features = torch.cat([feat_a, feat_b], dim=-1)  # [B, 128]
            logits_sum = self.sum_head(features)
            probs_sum = F.softmax(logits_sum, dim=-1)
            result["logits_sum"] = logits_sum
            result["probs_sum"] = probs_sum
        else:
            # Soft/Hard: compute expected sum via discrete convolution
            _, p_c_expected = sum_constraint_soft(
                probs_a, probs_b,
                probs_a.new_ones(probs_a.size(0), self.num_sums) / self.num_sums,  # dummy
                max_val=self.num_sums - 1,
            )
            result["logits_sum"] = torch.log(p_c_expected.clamp(min=1e-8))
            result["probs_sum"] = p_c_expected

        # Compute losses if targets are provided
        if digit_a is not None and digit_b is not None:
            loss_a = F.cross_entropy(logits_a, digit_a)
            loss_b = F.cross_entropy(logits_b, digit_b)
            loss_digit = loss_a + loss_b
            result["loss_digit"] = loss_digit

            if self.mode == "neural" and sum_target is not None:
                loss_sum = F.cross_entropy(logits_sum, sum_target)
                result["loss_sum"] = loss_sum
                result["loss_constraint"] = torch.tensor(0.0, device=img_a.device)
                result["loss_total"] = loss_digit + loss_sum
            elif self.mode in ("soft", "hard", "lagrangian"):
                # Constraint loss: KL between expected sum dist and current probs
                loss_constraint, p_c_expected = sum_constraint_soft(
                    probs_a, probs_b, result["probs_sum"],
                    max_val=self.num_sums - 1,
                )
                result["loss_constraint"] = loss_constraint
                result["loss_total"] = loss_digit  # constraint added externally with λ
            else:
                result["loss_constraint"] = torch.tensor(0.0, device=img_a.device)
                result["loss_total"] = loss_digit

        # Constraint satisfaction rate
        csr = constraint_satisfaction_rate(probs_a, probs_b, result["probs_sum"])
        result["csr"] = csr

        return result

    @torch.no_grad()
    def predict(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        use_hard_constraints: bool = False,
    ) -> dict:
        """Inference-time prediction.

        Args:
            img_a: [B, 1, 28, 28] digit a images.
            img_b: [B, 1, 28, 28] digit b images.
            use_hard_constraints: if True, apply Z3 verification/repair.

        Returns:
            Dict with pred_a, pred_b, pred_sum, csr, repair_rate.
        """
        self.eval()
        result = self.forward(img_a, img_b)

        pred_a = result["probs_a"].argmax(dim=-1)
        pred_b = result["probs_b"].argmax(dim=-1)
        pred_sum = result["probs_sum"].argmax(dim=-1)

        out = {
            "pred_a": pred_a,
            "pred_b": pred_b,
            "pred_sum": pred_sum,
            "csr": result["csr"],
            "repair_rate": 0.0,
        }

        if use_hard_constraints or self.mode == "hard":
            c_repaired, repair_rate = hard_constraint_batch(
                result["probs_a"], result["probs_b"], result["probs_sum"]
            )
            out["pred_sum"] = c_repaired
            out["repair_rate"] = repair_rate
            # Recompute CSR after repair
            out["csr"] = (pred_a + pred_b == c_repaired).float().mean().item()

        return out

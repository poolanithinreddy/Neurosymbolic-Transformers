"""Multi-digit addition model: CNN perception + carry-aware symbolic constraints.

Architecture:
    Shared DigitCNN encodes each 28×28 sub-image → 10-class logits.
    For a 2-digit number, we split the [1, 28, 56] image into left/right halves.
    Symbolic layer: carry-propagation constraints produce expected sum digits.
    Neural baseline: concatenated features → MLP for each output digit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.perception import DigitCNN, DigitEncoder
from symbolic.multi_digit_constraints import carry_constraint_soft, verify_multi_digit


class MultiDigitModel(nn.Module):
    """Neuro-symbolic model for 2-digit + 2-digit addition.

    Modes:
        - "neural": pure MLP output heads, no constraint.
        - "soft"/"lagrangian": CNN digit recognition + differentiable carry constraint.
    """

    def __init__(self, mode: str = "soft"):
        super().__init__()
        assert mode in ("neural", "soft", "lagrangian"), f"Unknown mode: {mode}"
        self.mode = mode

        # Shared digit CNN (processes each 28×28 sub-image)
        self.digit_cnn = DigitCNN(num_classes=10)

        if mode == "neural":
            # Neural heads for each output digit
            self.digit_encoder = DigitEncoder()
            # 4 × 64 features from 4 input digits
            self.sum_ones_head = nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 10)
            )
            self.sum_tens_head = nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 10)
            )
            self.sum_hund_head = nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 10)
            )

    def _split_digits(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a [B, 1, 28, 56] number image into two [B, 1, 28, 28] digit images."""
        w = img.size(-1) // 2
        return img[..., :w], img[..., w:]

    def forward(
        self,
        img_a: torch.Tensor,  # [B, 1, 28, 56]
        img_b: torch.Tensor,  # [B, 1, 28, 56]
        a_tens: torch.Tensor | None = None,
        a_ones: torch.Tensor | None = None,
        b_tens: torch.Tensor | None = None,
        b_ones: torch.Tensor | None = None,
        sum_ones: torch.Tensor | None = None,
        sum_tens: torch.Tensor | None = None,
        sum_hundreds: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass."""
        # Split each number image into tens and ones digits
        img_a_tens, img_a_ones = self._split_digits(img_a)
        img_b_tens, img_b_ones = self._split_digits(img_b)

        # Classify each digit
        logits_a_tens = self.digit_cnn(img_a_tens)
        logits_a_ones = self.digit_cnn(img_a_ones)
        logits_b_tens = self.digit_cnn(img_b_tens)
        logits_b_ones = self.digit_cnn(img_b_ones)

        probs_a_tens = F.softmax(logits_a_tens, dim=-1)
        probs_a_ones = F.softmax(logits_a_ones, dim=-1)
        probs_b_tens = F.softmax(logits_b_tens, dim=-1)
        probs_b_ones = F.softmax(logits_b_ones, dim=-1)

        result = {
            "logits_a_tens": logits_a_tens,
            "logits_a_ones": logits_a_ones,
            "logits_b_tens": logits_b_tens,
            "logits_b_ones": logits_b_ones,
            "probs_a_tens": probs_a_tens,
            "probs_a_ones": probs_a_ones,
            "probs_b_tens": probs_b_tens,
            "probs_b_ones": probs_b_ones,
        }

        if self.mode == "neural":
            # Extract features and predict sum digits via MLP
            feat_at = self.digit_encoder(img_a_tens)
            feat_ao = self.digit_encoder(img_a_ones)
            feat_bt = self.digit_encoder(img_b_tens)
            feat_bo = self.digit_encoder(img_b_ones)
            features = torch.cat([feat_at, feat_ao, feat_bt, feat_bo], dim=-1)

            logits_s_ones = self.sum_ones_head(features)
            logits_s_tens = self.sum_tens_head(features)
            logits_s_hund = self.sum_hund_head(features)

            result["probs_s_ones"] = F.softmax(logits_s_ones, dim=-1)
            result["probs_s_tens"] = F.softmax(logits_s_tens, dim=-1)
            result["probs_s_hund"] = F.softmax(logits_s_hund, dim=-1)
        else:
            # Symbolic: derive sum digits from digit distributions via constraint
            # Use uniform initialisation for predicted sum distributions
            B = img_a.size(0)
            device = img_a.device
            p_s_ones = torch.ones(B, 10, device=device) / 10
            p_s_tens = torch.ones(B, 10, device=device) / 10
            p_s_hund = torch.ones(B, 10, device=device) / 10

            # The carry constraint computes expected sum distributions
            _, info = carry_constraint_soft(
                probs_a_ones, probs_a_tens,
                probs_b_ones, probs_b_tens,
                p_s_ones, p_s_tens, p_s_hund,
            )

            # Use expected distributions as output
            # Recompute them properly
            ones_sum = torch.zeros(B, 19, device=device)
            for k in range(19):
                for i in range(max(0, k - 9), min(9, k) + 1):
                    ones_sum[:, k] += probs_a_ones[:, i] * probs_b_ones[:, k - i]

            p_carry = ones_sum[:, 10:].sum(dim=-1)

            p_s_ones_exp = torch.zeros(B, 10, device=device)
            for d in range(10):
                p_s_ones_exp[:, d] = ones_sum[:, d] + (ones_sum[:, d + 10] if d + 10 < 19 else 0)
            p_s_ones_exp = p_s_ones_exp / p_s_ones_exp.sum(-1, keepdim=True).clamp(min=1e-8)

            tens_sum = torch.zeros(B, 20, device=device)
            for k in range(20):
                for carry in (0, 1):
                    pc = p_carry if carry == 1 else (1 - p_carry)
                    rem = k - carry
                    if rem < 0 or rem > 18:
                        continue
                    for i in range(max(0, rem - 9), min(9, rem) + 1):
                        tens_sum[:, k] += probs_a_tens[:, i] * probs_b_tens[:, rem - i] * pc

            p_s_tens_exp = torch.zeros(B, 10, device=device)
            for d in range(10):
                p_s_tens_exp[:, d] += tens_sum[:, d]
                if d + 10 < 20:
                    p_s_tens_exp[:, d] += tens_sum[:, d + 10]
            p_s_tens_exp = p_s_tens_exp / p_s_tens_exp.sum(-1, keepdim=True).clamp(min=1e-8)

            p_carry2 = tens_sum[:, 10:].sum(dim=-1)
            p_s_hund_exp = torch.zeros(B, 10, device=device)
            p_s_hund_exp[:, 0] = 1 - p_carry2
            p_s_hund_exp[:, 1] = p_carry2

            result["probs_s_ones"] = p_s_ones_exp
            result["probs_s_tens"] = p_s_tens_exp
            result["probs_s_hund"] = p_s_hund_exp

        # Compute losses
        if a_tens is not None:
            loss_digits = (
                F.cross_entropy(logits_a_tens, a_tens) +
                F.cross_entropy(logits_a_ones, a_ones) +
                F.cross_entropy(logits_b_tens, b_tens) +
                F.cross_entropy(logits_b_ones, b_ones)
            )
            result["loss_digit"] = loss_digits

            if self.mode == "neural" and sum_ones is not None:
                # NOTE: probs_s_* are softmax outputs (probabilities).  Use
                # F.nll_loss(log(p), target) — NOT F.cross_entropy(log(p)),
                # which would apply log_softmax *again* → double-log → NaN.
                log_p_s_ones = torch.log(result["probs_s_ones"].clamp(min=1e-8))
                log_p_s_tens = torch.log(result["probs_s_tens"].clamp(min=1e-8))
                log_p_s_hund = torch.log(result["probs_s_hund"].clamp(min=1e-8))
                loss_sum = (
                    F.nll_loss(log_p_s_ones, sum_ones) +
                    F.nll_loss(log_p_s_tens, sum_tens) +
                    F.nll_loss(log_p_s_hund, sum_hundreds)
                )
                result["loss_sum"] = loss_sum
                result["loss_constraint"] = torch.tensor(0.0, device=img_a.device)
                result["loss_total"] = loss_digits + loss_sum
            elif self.mode in ("soft", "lagrangian") and sum_ones is not None:
                loss_constraint, _ = carry_constraint_soft(
                    probs_a_ones, probs_a_tens, probs_b_ones, probs_b_tens,
                    result["probs_s_ones"], result["probs_s_tens"], result["probs_s_hund"],
                )
                result["loss_constraint"] = loss_constraint
                result["loss_total"] = loss_digits  # constraint added externally

        # CSR
        violations, csr = verify_multi_digit(
            probs_a_tens.argmax(-1), probs_a_ones.argmax(-1),
            probs_b_tens.argmax(-1), probs_b_ones.argmax(-1),
            result["probs_s_ones"].argmax(-1),
            result["probs_s_tens"].argmax(-1),
            result["probs_s_hund"].argmax(-1),
        )
        result["csr"] = csr

        return result

    @torch.no_grad()
    def predict(self, img_a: torch.Tensor, img_b: torch.Tensor) -> dict:
        """Inference-time prediction."""
        self.eval()
        result = self.forward(img_a, img_b)
        return {
            "pred_a_tens": result["probs_a_tens"].argmax(-1),
            "pred_a_ones": result["probs_a_ones"].argmax(-1),
            "pred_b_tens": result["probs_b_tens"].argmax(-1),
            "pred_b_ones": result["probs_b_ones"].argmax(-1),
            "pred_s_ones": result["probs_s_ones"].argmax(-1),
            "pred_s_tens": result["probs_s_tens"].argmax(-1),
            "pred_s_hund": result["probs_s_hund"].argmax(-1),
            "csr": result["csr"],
        }

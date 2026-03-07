"""NST-VERI Verification Heads: learned bridge between neural and symbolic.

Instead of applying heuristic constraints as external penalties, verification
heads are auxiliary binary classifiers that predict verifiable properties
from the transformer's hidden state:
  - h_num:  "Does the claim involve a numerical comparison with evidence?"
  - h_neg:  "Does the evidence semantically negate the claim?"
  - h_ent:  "Do claim and evidence share key entities?"
  - h_suf:  "Is the evidence sufficient to determine the claim?"
  - h_temp: "Is there temporal consistency/inconsistency?"
  - h_hedge: "Does the claim contain hedging language?"

These create a learned bottleneck that:
  1. Forces the model to develop representations aware of symbolic properties
  2. Provides calibrated soft signals for downstream gating
  3. Produces a residual logit correction conditioned on verification outputs

Key architectural choice: the residual correction starts at ZERO (via
zero-initialized scale parameter), so the model begins as pure neural
and only adds verification influence as training progresses.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VerificationHeads(nn.Module):
    """Auxiliary verification heads that predict symbolic properties
    from the transformer's hidden state.

    Architecture per head:
        Dropout → Linear(H→H/4) → GELU → Linear(H/4→1)

    Plus a shared residual correction network:
        Linear(H+K → H/2) → GELU → Dropout → Linear(H/2 → 3)
    scaled by a learned parameter initialised to 0.

    Args:
        hidden_dim: Transformer hidden dimension (768 for base, 1024 for large).
        n_constraints: Number of verification heads (matches constraint engine).
        dropout: Dropout rate for regularisation.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_constraints: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_constraints = n_constraints

        # Per-constraint verification head
        head_inner = max(32, hidden_dim // 4)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, head_inner),
                nn.GELU(),
                nn.Linear(head_inner, 1),
            )
            for _ in range(n_constraints)
        ])

        # Residual correction network:
        # Takes [CLS] embedding + verification outputs → 3-class logit correction
        residual_inner = max(64, hidden_dim // 2)
        self.residual_net = nn.Sequential(
            nn.Linear(hidden_dim + n_constraints, residual_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_inner, 3),  # 3-class correction
        )

        # Learned scale: initialised to 0 → starts as pure neural
        # The sigmoid maps this to ≈0.5 when bias=-3 → effective scale ≈0.047
        # As training proceeds, this can grow to allow more correction
        self.residual_scale = nn.Parameter(torch.tensor(-3.0))

        self._init_weights()

    def _init_weights(self):
        """Careful initialisation to ensure stability."""
        for head in self.heads:
            # Small init for the final linear — predictions start near 0.5
            nn.init.xavier_uniform_(head[-1].weight, gain=0.1)
            nn.init.zeros_(head[-1].bias)

        # Residual net final layer: zero init → no correction at start
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def forward(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_state: [CLS] embedding, shape (B, H)

        Returns:
            verification_logits: (B, K) — raw logits for each verification head
            verification_probs:  (B, K) — sigmoid probabilities
            logit_correction:    (B, 3) — residual correction to add to main logits
        """
        # Get verification predictions
        v_logits = torch.cat(
            [head(hidden_state) for head in self.heads], dim=-1
        )  # (B, K)
        v_probs = torch.sigmoid(v_logits)

        # Compute residual correction
        combined = torch.cat([hidden_state, v_probs], dim=-1)
        correction = self.residual_net(combined)  # (B, 3)

        # Scale by learned parameter (starts near 0)
        scale = torch.sigmoid(self.residual_scale)
        scaled_correction = correction * scale

        return {
            "verification_logits": v_logits,
            "verification_probs": v_probs,
            "logit_correction": scaled_correction,
            "residual_scale": scale.item(),
        }

    def verification_loss(
        self,
        v_logits: torch.Tensor,    # (B, K)
        targets: torch.Tensor,     # (B, K) soft labels from constraint engine
        mask: torch.Tensor,        # (B, K) which constraints fired (valid targets)
    ) -> torch.Tensor:
        """Binary cross-entropy loss for verification heads.

        Only computed on examples where the constraint fires (mask=1),
        using soft targets from the constraint engine's confidence.

        Args:
            v_logits: Raw logits from verification heads.
            targets: Soft target values [0,1] from constraint confidence.
            mask: Binary mask — 1 where constraint fires, 0 otherwise.

        Returns:
            Scalar average verification loss.
        """
        # BCE with logits for numerical stability
        loss = nn.functional.binary_cross_entropy_with_logits(
            v_logits, targets, reduction='none'
        )
        # Mask: only count where constraints fire
        masked_loss = (loss * mask.float()).sum()
        n_valid = mask.float().sum().clamp(min=1.0)
        return masked_loss / n_valid

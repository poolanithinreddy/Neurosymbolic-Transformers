"""NST-VERI Adaptive Lambda: per-sample constraint weighting.

Instead of a single global λ that weights all constraints equally for
all examples, this module learns to estimate how much to trust constraints
for each individual example based on:
  1. The constraint signals themselves (fires, confidence)
  2. The model's current prediction uncertainty (entropy)
  3. Constraint agreement (do constraints agree with each other?)

Key design decisions:
  - Initialised to output near-zero λ (start unconstrained)
  - Bounded by lambda_max (prevents runaway constraint domination)
  - Differentiable end-to-end (learns jointly with the model)
  - Can be warmed up via an external schedule multiplier
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math


class AdaptiveLambdaModule(nn.Module):
    """Per-sample adaptive constraint weighting.

    Input features:
      - constraint_fires:      (B, K) binary — which constraints fire
      - constraint_confidence:  (B, K) float  — constraint self-reported confidence
      - model_probs:            (B, 3) float  — model's current class probabilities

    Output:
      - lambda_per_sample:      (B,) float    — how much to weight constraints

    The module also outputs per-constraint weights (B, K) that can be
    used by the constraint engine for fine-grained control.

    Args:
        n_constraints: Number of constraints (K).
        hidden_dim: Hidden layer size.
        lambda_max: Upper bound on per-sample lambda.
    """

    def __init__(
        self,
        n_constraints: int = 6,
        hidden_dim: int = 64,
        lambda_max: float = 0.5,
    ):
        super().__init__()
        self.n_constraints = n_constraints
        self.lambda_max = lambda_max

        # Input: fires(K) + confidence(K) + entropy(1) + max_confidence(1)
        #       + constraint_agreement(1)
        input_dim = n_constraints * 2 + 3

        self.lambda_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Per-constraint gate weights
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_constraints),
            nn.Sigmoid(),
        )

        # Initialise to output near-zero (start unconstrained)
        nn.init.constant_(self.lambda_net[-2].bias, -3.0)
        # Gate starts at ~0.5 (neutral)
        nn.init.constant_(self.gate_net[-2].bias, 0.0)

    def forward(
        self,
        constraint_fires: torch.Tensor,      # (B, K)
        constraint_confidence: torch.Tensor,  # (B, K)
        model_probs: torch.Tensor,            # (B, 3)
        schedule_multiplier: float = 1.0,     # External warmup schedule
    ) -> dict[str, torch.Tensor]:
        """Compute per-sample lambda and per-constraint gate weights.

        Args:
            constraint_fires: Binary tensor of which constraints fire.
            constraint_confidence: Constraint confidence values.
            model_probs: Model's current output probabilities.
            schedule_multiplier: External multiplier (0→1 during warmup).

        Returns:
            dict with:
                lambda_per_sample: (B,) per-sample constraint weight
                gate_weights: (B, K) per-sample per-constraint weights
        """
        # Compute features
        # Model uncertainty: normalised entropy of predictions
        entropy = -(model_probs * (model_probs + 1e-8).log()).sum(dim=-1, keepdim=True)
        max_entropy = math.log(3.0)
        norm_entropy = entropy / max_entropy  # (B, 1), in [0, 1]

        # Model confidence: max predicted probability
        max_conf = model_probs.max(dim=-1, keepdim=True).values  # (B, 1)

        # Constraint agreement: std of confidence across constraints
        # High agreement (low std) → more trustworthy overall
        if self.n_constraints > 1:
            conf_std = constraint_confidence.std(dim=-1, keepdim=True)
        else:
            conf_std = torch.zeros(constraint_fires.shape[0], 1, device=model_probs.device)

        features = torch.cat([
            constraint_fires.float(),
            constraint_confidence,
            norm_entropy,
            max_conf,
            conf_std,
        ], dim=-1)

        # Compute outputs
        lambda_raw = self.lambda_net(features).squeeze(-1)  # (B,)
        gate_weights = self.gate_net(features)               # (B, K)

        # Apply bounds and schedule
        lambda_per_sample = lambda_raw * self.lambda_max * schedule_multiplier

        return {
            "lambda_per_sample": lambda_per_sample,
            "gate_weights": gate_weights,
        }

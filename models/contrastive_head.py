"""NST-VERI Contrastive Constraint Head.

For high-confidence constraint examples, uses a supervised contrastive loss
that pushes the [CLS] representation toward the correct class prototype
and away from incorrect class prototypes.

This provides representation-level guidance (not just logit-level penalties),
which shapes the feature space to be more separable for constrained examples.

Architecture:
    - 3 learnable class prototypes in the representation space
    - Supervised contrastive loss on high-confidence constraint examples
    - Temperature-scaled cosine similarity

Key insight: standard constraint penalties only adjust output logits.
Contrastive loss shapes the INTERNAL representation space, making the
model's features inherently more aligned with constraint structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveConstraintHead(nn.Module):
    """Contrastive learning head for constraint-aware representation shaping.

    Maintains learnable prototypes for each class and computes a
    supervised contrastive loss that pulls representations toward
    their correct class prototype.

    Args:
        hidden_dim: Dimension of the [CLS] representation.
        n_classes: Number of classes (3 for FEVER).
        temperature: Temperature for cosine similarity scaling.
        projection_dim: Dimension of the projection space (None = use hidden_dim).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_classes: int = 3,
        temperature: float = 0.07,
        projection_dim: int | None = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.temperature = temperature

        proj_dim = projection_dim or min(256, hidden_dim)

        # Projection head: maps [CLS] to a lower-dimensional space
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Learnable class prototypes in the projection space
        self.prototypes = nn.Parameter(
            torch.randn(n_classes, proj_dim) * 0.01
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, H) — [CLS] embeddings
        labels: torch.Tensor,          # (B,) — ground truth labels
        mask: torch.Tensor | None = None,  # (B,) — which samples to include
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            hidden_states: [CLS] embeddings from the transformer.
            labels: Ground truth class labels.
            mask: Binary mask — 1 for high-confidence constraint examples.
                  If None, uses all examples.

        Returns:
            Scalar contrastive loss.
        """
        if mask is not None and mask.sum() == 0:
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

        # Project to contrastive space
        z = self.projector(hidden_states)  # (B, proj_dim)
        z = F.normalize(z, dim=-1)         # L2 normalise

        # Normalise prototypes
        proto = F.normalize(self.prototypes, dim=-1)  # (C, proj_dim)

        # Cosine similarity
        sim = z @ proto.T / self.temperature  # (B, C)

        # Apply mask if provided
        if mask is not None:
            z = z[mask.bool()]
            sim = sim[mask.bool()]
            labels = labels[mask.bool()]

        if len(labels) == 0:
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

        # Cross-entropy over prototype similarities
        loss = F.cross_entropy(sim, labels)

        return loss

    def get_prototype_distances(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Get cosine distances from each class prototype.

        Useful for analysis and visualisation.

        Returns:
            (B, C) tensor of cosine similarities.
        """
        z = self.projector(hidden_states)
        z = F.normalize(z, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        return z @ proto.T

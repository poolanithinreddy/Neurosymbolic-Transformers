"""Lagrangian dual-variable optimiser for neuro-symbolic constraint integration.

Instead of a fixed λ hyperparameter, this module learns λ as a dual variable
of the augmented Lagrangian:

    min_θ max_{λ≥0}  L_task(θ) + λ·(L_logic(θ) − ε) + ρ/2·[max(0, L_logic(θ)−ε)]²

The dual variable λ is updated after each epoch (or step) via:

    λ ← max(0, λ + α·(L_logic − ε))

Key properties:
1. λ increases automatically when constraints are violated.
2. λ decreases when constraints are satisfied beyond the tolerance ε.
3. At convergence, λ* is the "price of logic" — the marginal task-loss
   cost per unit of constraint tightening (shadow price).
4. Setting α = 0, ε = 0 recovers the fixed-λ baseline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import torch


@dataclass
class LagrangianState:
    """Tracks the state of the Lagrangian dual variable across training.

    Attributes:
        lam: current dual variable value (λ).
        epsilon: constraint tolerance (ε) — acceptable violation level.
        alpha: dual step size for λ updates.
        rho: quadratic penalty coefficient for augmented Lagrangian.
        lam_max: upper bound on λ to prevent divergence.
        history: list of (step, λ, L_logic, L_task) tuples for analysis.
    """

    lam: float = 0.0
    epsilon: float = 0.05
    alpha: float = 0.01
    rho: float = 1.0
    lam_max: float = 10.0
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lam": self.lam,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "rho": self.rho,
            "lam_max": self.lam_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LagrangianState":
        return cls(
            lam=d.get("lam", 0.0),
            epsilon=d.get("epsilon", 0.05),
            alpha=d.get("alpha", 0.01),
            rho=d.get("rho", 1.0),
            lam_max=d.get("lam_max", 10.0),
        )


def lagrangian_loss(
    loss_task: torch.Tensor,
    loss_logic: torch.Tensor,
    state: LagrangianState,
) -> torch.Tensor:
    """Compute the augmented Lagrangian total loss.

    L = L_task + λ·(L_logic − ε) + ρ/2·[max(0, L_logic − ε)]²

    The quadratic penalty term smooths the optimisation landscape near the
    constraint boundary and helps prevent oscillation.

    Args:
        loss_task: scalar task loss (cross-entropy, etc.).
        loss_logic: scalar constraint violation loss.
        state: current Lagrangian dual variable state.

    Returns:
        total_loss: scalar augmented Lagrangian loss for backprop.
    """
    constraint_slack = loss_logic - state.epsilon

    # Linear term: λ · (L_logic − ε)
    linear_term = state.lam * constraint_slack

    # Quadratic penalty: ρ/2 · [max(0, L_logic − ε)]²
    penalty = (state.rho / 2.0) * torch.clamp(constraint_slack, min=0.0) ** 2

    total = loss_task + linear_term + penalty

    return total


def update_dual_variable(
    state: LagrangianState,
    loss_logic: float,
    step: int | None = None,
    loss_task: float | None = None,
) -> float:
    """Update the dual variable λ after an epoch/step.

    λ ← max(0, λ + α · (L_logic − ε))

    When L_logic > ε (constraint violated), λ increases → more constraint weight.
    When L_logic < ε (constraint satisfied), λ decreases → less constraint weight.

    Args:
        state: mutable LagrangianState to update in-place.
        loss_logic: current constraint loss value.
        step: optional step number for logging.
        loss_task: optional task loss for logging.

    Returns:
        new_lambda: the updated λ value.
    """
    constraint_slack = loss_logic - state.epsilon
    new_lam = max(0.0, state.lam + state.alpha * constraint_slack)
    new_lam = min(new_lam, state.lam_max)  # Clamp to prevent divergence

    state.lam = new_lam

    # Log history
    entry = {
        "step": step,
        "lambda": round(new_lam, 6),
        "loss_logic": round(loss_logic, 6),
        "constraint_slack": round(constraint_slack, 6),
    }
    if loss_task is not None:
        entry["loss_task"] = round(loss_task, 6)
    state.history.append(entry)

    return new_lam


def price_of_logic(state: LagrangianState) -> float:
    """Return the converged dual variable λ* — the 'price of logic'.

    This is the marginal task-loss cost per unit of constraint tightening.
    A high λ* means the constraint is expensive (conflicts with the task).
    A low λ* means the constraint is cheap (aligned with the task).

    Returns:
        λ* (the current dual variable value).
    """
    return state.lam


def save_lambda_trajectory(state: LagrangianState, path: str) -> None:
    """Save the λ trajectory to a JSON file for analysis and plotting."""
    with open(path, "w") as f:
        json.dump(
            {
                "final_lambda": state.lam,
                "epsilon": state.epsilon,
                "alpha": state.alpha,
                "rho": state.rho,
                "trajectory": state.history,
            },
            f,
            indent=2,
        )


def load_lambda_trajectory(path: str) -> dict:
    """Load a saved λ trajectory for plotting."""
    with open(path) as f:
        return json.load(f)


class MultiConstraintLagrangian:
    """Manage multiple constraints, each with its own dual variable.

    Useful when a model has multiple types of constraints (e.g., arithmetic
    + transitivity + symmetry), each requiring independent weighting.
    """

    def __init__(
        self,
        constraint_names: list[str],
        epsilon: float = 0.05,
        alpha: float = 0.01,
        rho: float = 1.0,
        lam_max: float = 10.0,
    ):
        self.states: dict[str, LagrangianState] = {}
        for name in constraint_names:
            self.states[name] = LagrangianState(
                lam=0.0,
                epsilon=epsilon,
                alpha=alpha,
                rho=rho,
                lam_max=lam_max,
            )

    def compute_loss(
        self,
        loss_task: torch.Tensor,
        constraint_losses: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute combined augmented Lagrangian over all constraints.

        L = L_task + Σ_i [λ_i · (L_i − ε_i) + ρ_i/2 · max(0, L_i − ε_i)²]
        """
        total = loss_task
        for name, loss_logic in constraint_losses.items():
            if name in self.states:
                state = self.states[name]
                slack = loss_logic - state.epsilon
                total = total + state.lam * slack
                total = total + (state.rho / 2.0) * torch.clamp(slack, min=0.0) ** 2
        return total

    def update_all(
        self,
        constraint_losses: dict[str, float],
        step: int | None = None,
    ) -> dict[str, float]:
        """Update all dual variables. Returns dict of new λ values."""
        result = {}
        for name, loss_val in constraint_losses.items():
            if name in self.states:
                new_lam = update_dual_variable(
                    self.states[name], loss_val, step=step
                )
                result[name] = new_lam
        return result

    def get_lambdas(self) -> dict[str, float]:
        return {name: s.lam for name, s in self.states.items()}

    def to_dict(self) -> dict:
        return {name: s.to_dict() for name, s in self.states.items()}

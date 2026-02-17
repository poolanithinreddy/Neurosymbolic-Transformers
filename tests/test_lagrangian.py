"""Tests for the Lagrangian dual-variable constraint optimiser."""

import os
import sys
import tempfile

import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from symbolic.lagrangian import (
    LagrangianState,
    MultiConstraintLagrangian,
    lagrangian_loss,
    load_lambda_trajectory,
    price_of_logic,
    save_lambda_trajectory,
    update_dual_variable,
)


class TestLagrangianLoss:
    def test_loss_increases_with_violation(self):
        """When constraint is violated (L_logic > ε), total loss should be higher."""
        state = LagrangianState(lam=1.0, epsilon=0.05, rho=1.0)
        loss_task = torch.tensor(1.0)

        # Satisfied: L_logic < ε
        loss_satisfied = lagrangian_loss(loss_task, torch.tensor(0.01), state)
        # Violated: L_logic > ε
        loss_violated = lagrangian_loss(loss_task, torch.tensor(0.5), state)

        assert loss_violated.item() > loss_satisfied.item()

    def test_loss_equals_task_when_lambda_zero(self):
        """With λ=0 and L_logic ≤ ε, loss should be close to task loss."""
        state = LagrangianState(lam=0.0, epsilon=1.0, rho=0.0)
        loss_task = torch.tensor(2.5)
        loss_logic = torch.tensor(0.1)

        total = lagrangian_loss(loss_task, loss_logic, state)
        # With lam=0, epsilon=1.0, rho=0: total = 2.5 + 0*(0.1-1.0) + 0 = 2.5
        assert abs(total.item() - 2.5) < 1e-4

    def test_quadratic_penalty(self):
        """Quadratic penalty kicks in when L_logic > ε."""
        state = LagrangianState(lam=0.0, epsilon=0.1, rho=2.0)
        loss_task = torch.tensor(1.0)
        loss_logic = torch.tensor(0.6)

        total = lagrangian_loss(loss_task, loss_logic, state)
        # Expected: 1.0 + 0*(0.6 - 0.1) + 2.0/2 * (0.5)^2 = 1.0 + 0.25 = 1.25
        assert abs(total.item() - 1.25) < 1e-4

    def test_no_penalty_when_satisfied(self):
        """No quadratic penalty when L_logic ≤ ε."""
        state = LagrangianState(lam=0.5, epsilon=0.5, rho=5.0)
        loss_task = torch.tensor(1.0)
        loss_logic = torch.tensor(0.3)  # < ε

        total = lagrangian_loss(loss_task, loss_logic, state)
        # Expected: 1.0 + 0.5*(0.3 - 0.5) + 5.0/2 * max(0, -0.2)^2
        # = 1.0 + 0.5*(-0.2) + 0 = 1.0 - 0.1 = 0.9
        assert abs(total.item() - 0.9) < 1e-4

    def test_differentiable(self):
        """Loss should be differentiable w.r.t. model parameters."""
        state = LagrangianState(lam=1.0, epsilon=0.05, rho=1.0)
        loss_task = torch.tensor(1.0, requires_grad=True)
        loss_logic = torch.tensor(0.3, requires_grad=True)

        total = lagrangian_loss(loss_task, loss_logic, state)
        total.backward()
        assert loss_task.grad is not None
        assert loss_logic.grad is not None


class TestDualUpdate:
    def test_lambda_increases_on_violation(self):
        """λ should increase when L_logic > ε."""
        state = LagrangianState(lam=0.5, epsilon=0.1, alpha=0.1)
        update_dual_variable(state, loss_logic=0.5, step=1)
        assert state.lam > 0.5

    def test_lambda_decreases_on_satisfaction(self):
        """λ should decrease when L_logic < ε."""
        state = LagrangianState(lam=0.5, epsilon=0.5, alpha=0.1)
        update_dual_variable(state, loss_logic=0.1, step=1)
        assert state.lam < 0.5

    def test_lambda_non_negative(self):
        """λ should never go below 0."""
        state = LagrangianState(lam=0.01, epsilon=1.0, alpha=1.0)
        update_dual_variable(state, loss_logic=0.0, step=1)
        assert state.lam >= 0.0

    def test_lambda_capped_at_max(self):
        """λ should not exceed lam_max."""
        state = LagrangianState(lam=9.0, epsilon=0.0, alpha=100.0, lam_max=10.0)
        update_dual_variable(state, loss_logic=5.0, step=1)
        assert state.lam <= 10.0

    def test_history_recorded(self):
        """Each update should be recorded in history."""
        state = LagrangianState()
        for i in range(5):
            update_dual_variable(state, loss_logic=0.1 * (i + 1), step=i)
        assert len(state.history) == 5
        assert state.history[0]["step"] == 0

    def test_fixed_lambda_special_case(self):
        """With α=0, λ should not change (recovers fixed-λ baseline)."""
        state = LagrangianState(lam=0.5, epsilon=0.0, alpha=0.0)
        update_dual_variable(state, loss_logic=1.0, step=1)
        assert state.lam == 0.5


class TestPriceOfLogic:
    def test_returns_current_lambda(self):
        state = LagrangianState(lam=2.5)
        assert price_of_logic(state) == 2.5


class TestSaveLoad:
    def test_save_and_load_trajectory(self):
        state = LagrangianState(lam=0.0, epsilon=0.1, alpha=0.05)
        for i in range(10):
            update_dual_variable(state, loss_logic=0.2 - 0.01 * i, step=i)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_lambda_trajectory(state, path)
            data = load_lambda_trajectory(path)
            assert "trajectory" in data
            assert len(data["trajectory"]) == 10
            assert "final_lambda" in data
        finally:
            os.unlink(path)


class TestMultiConstraint:
    def test_multi_constraint_basic(self):
        """Multiple constraints should each have their own λ."""
        mc = MultiConstraintLagrangian(
            ["arithmetic", "symmetry"],
            epsilon=0.1, alpha=0.05,
        )
        loss_task = torch.tensor(1.0)
        constraints = {
            "arithmetic": torch.tensor(0.3),
            "symmetry": torch.tensor(0.05),
        }
        total = mc.compute_loss(loss_task, constraints)
        assert total.item() > 0

    def test_multi_constraint_update(self):
        mc = MultiConstraintLagrangian(
            ["a", "b"], epsilon=0.1, alpha=0.1,
        )
        new_lambdas = mc.update_all({"a": 0.5, "b": 0.01}, step=1)
        # "a" violated (0.5 > 0.1) → λ_a should increase
        assert new_lambdas["a"] > 0
        # "b" satisfied (0.01 < 0.1) → λ_b should stay 0 (can't go below 0)
        assert new_lambdas["b"] == 0.0

    def test_get_lambdas(self):
        mc = MultiConstraintLagrangian(["x", "y"])
        lambdas = mc.get_lambdas()
        assert "x" in lambdas
        assert "y" in lambdas
        assert all(v == 0.0 for v in lambdas.values())


class TestLagrangianState:
    def test_serialization(self):
        state = LagrangianState(lam=1.5, epsilon=0.1, alpha=0.05, rho=2.0)
        d = state.to_dict()
        restored = LagrangianState.from_dict(d)
        assert restored.lam == 1.5
        assert restored.epsilon == 0.1
        assert restored.alpha == 0.05
        assert restored.rho == 2.0

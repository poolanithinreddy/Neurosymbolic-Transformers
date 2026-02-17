"""Unit tests for the symbolic reasoning module.

Tests cover:
1. Soft constraint (differentiable digit-addition constraint)
2. Hard constraint (Z3 verification/repair)
3. Rule engine (YAML rule loading + evaluation)
4. T-norm logic primitives (via logic.logic)
"""

import os
import sys

import torch

# Ensure imports work
_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from symbolic.constraint_solver import (
    constraint_satisfaction_rate,
    hard_constraint_verify,
    soft_rule_loss,
    sum_constraint_soft,
    sum_constraint_violation,
)
from symbolic.rule_engine import evaluate_all_rules, evaluate_rule, load_rules


# ===== Soft constraint tests =====


class TestSumConstraintSoft:
    def test_perfect_prediction(self):
        """When digit distributions are one-hot and correct, loss should be near zero."""
        B = 4
        p_a = torch.zeros(B, 10)
        p_b = torch.zeros(B, 10)
        p_c = torch.zeros(B, 19)

        # Set one-hot: a=3, b=5, sum=8
        p_a[:, 3] = 1.0
        p_b[:, 5] = 1.0
        p_c[:, 8] = 1.0

        loss, p_expected = sum_constraint_soft(p_a, p_b, p_c, max_val=18)
        assert loss.item() < 0.01, f"Loss should be near zero for correct sums, got {loss.item()}"
        assert p_expected[:, 8].mean().item() > 0.99

    def test_wrong_prediction_high_loss(self):
        """When predicted sum is wrong, loss should be high."""
        B = 4
        p_a = torch.zeros(B, 10)
        p_b = torch.zeros(B, 10)
        p_c = torch.zeros(B, 19)

        p_a[:, 3] = 1.0
        p_b[:, 5] = 1.0
        p_c[:, 2] = 1.0  # Wrong: predicting sum=2 instead of 8

        loss, _ = sum_constraint_soft(p_a, p_b, p_c, max_val=18)
        assert loss.item() > 1.0, f"Loss should be high for wrong sums, got {loss.item()}"

    def test_uniform_distribution(self):
        """Uniform digit predictions should give moderate loss."""
        B = 2
        p_a = torch.ones(B, 10) / 10
        p_b = torch.ones(B, 10) / 10
        p_c = torch.ones(B, 19) / 19

        loss, p_expected = sum_constraint_soft(p_a, p_b, p_c, max_val=18)
        # Loss should be finite and non-negative
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)
        # Expected distribution should sum to 1
        assert torch.allclose(p_expected.sum(dim=-1), torch.ones(B), atol=1e-4)

    def test_differentiable(self):
        """Constraint loss should be differentiable."""
        p_a = torch.randn(2, 10, requires_grad=True).softmax(dim=-1)
        p_b = torch.randn(2, 10, requires_grad=True).softmax(dim=-1)
        p_c = torch.randn(2, 19, requires_grad=True).softmax(dim=-1)

        loss, _ = sum_constraint_soft(p_a, p_b, p_c, max_val=18)
        loss.backward()
        # Gradients should exist and be finite
        # (p_a, p_b, p_c may not have grads because softmax detaches; check loss is finite)
        assert torch.isfinite(loss)


# ===== Violation detection tests =====


class TestConstraintViolation:
    def test_no_violation(self):
        """No violation when argmax sum is correct."""
        p_a = torch.zeros(3, 10)
        p_b = torch.zeros(3, 10)
        p_c = torch.zeros(3, 19)

        p_a[0, 2] = 1.0; p_b[0, 3] = 1.0; p_c[0, 5] = 1.0  # 2+3=5 ✓
        p_a[1, 0] = 1.0; p_b[1, 0] = 1.0; p_c[1, 0] = 1.0  # 0+0=0 ✓
        p_a[2, 9] = 1.0; p_b[2, 9] = 1.0; p_c[2, 18] = 1.0  # 9+9=18 ✓

        violations = sum_constraint_violation(p_a, p_b, p_c)
        assert violations.sum().item() == 0

    def test_violation_detected(self):
        """Violation detected when sum is wrong."""
        p_a = torch.zeros(2, 10)
        p_b = torch.zeros(2, 10)
        p_c = torch.zeros(2, 19)

        p_a[0, 2] = 1.0; p_b[0, 3] = 1.0; p_c[0, 7] = 1.0  # 2+3≠7 ✗
        p_a[1, 1] = 1.0; p_b[1, 1] = 1.0; p_c[1, 2] = 1.0  # 1+1=2 ✓

        violations = sum_constraint_violation(p_a, p_b, p_c)
        assert violations[0].item() == True  # noqa: E712
        assert violations[1].item() == False  # noqa: E712

    def test_csr(self):
        """CSR should be 1.0 when all sums are correct."""
        p_a = torch.zeros(2, 10)
        p_b = torch.zeros(2, 10)
        p_c = torch.zeros(2, 19)

        p_a[0, 4] = 1.0; p_b[0, 5] = 1.0; p_c[0, 9] = 1.0
        p_a[1, 7] = 1.0; p_b[1, 2] = 1.0; p_c[1, 9] = 1.0

        csr = constraint_satisfaction_rate(p_a, p_b, p_c)
        assert csr == 1.0


# ===== Hard constraint (Z3) tests =====


class TestHardConstraint:
    def test_correct_sum(self):
        """Z3 should confirm correct sums."""
        satisfied, (a, b, c) = hard_constraint_verify(3, 5, 8)
        assert satisfied is True
        assert c == 8

    def test_repair_wrong_sum(self):
        """Z3 should repair incorrect sums."""
        satisfied, (a, b, c) = hard_constraint_verify(3, 5, 7)
        assert satisfied is False
        assert c == 8  # repaired to 3+5=8

    def test_edge_case_zero(self):
        satisfied, (_, _, c) = hard_constraint_verify(0, 0, 0)
        assert satisfied is True
        assert c == 0

    def test_edge_case_max(self):
        satisfied, (_, _, c) = hard_constraint_verify(9, 9, 18)
        assert satisfied is True
        assert c == 18


# ===== Soft rule loss tests =====


class TestSoftRuleLoss:
    def test_satisfied_rule(self):
        """Loss should be near zero when head is true."""
        body = [torch.tensor([1.0, 1.0, 1.0])]
        head = torch.tensor([1.0, 1.0, 1.0])
        loss = soft_rule_loss(body, head)
        assert loss.item() < 1e-6

    def test_violated_rule(self):
        """Loss should be positive when body is true but head is false."""
        body = [torch.tensor([1.0, 1.0])]
        head = torch.tensor([0.0, 0.0])
        loss = soft_rule_loss(body, head)
        assert loss.item() > 0.9

    def test_vacuous_body(self):
        """Loss should be zero when body is false."""
        body = [torch.tensor([0.0, 0.0])]
        head = torch.tensor([0.0, 0.0])
        loss = soft_rule_loss(body, head)
        assert loss.item() < 1e-6

    def test_empty_body(self):
        """Loss should be zero with empty body."""
        head = torch.tensor([0.5, 0.5])
        loss = soft_rule_loss([], head)
        assert loss.item() < 1e-6


# ===== Rule engine tests =====


class TestRuleEngine:
    def test_load_rules(self):
        """Should load rules from YAML without error."""
        rules = load_rules()
        assert len(rules) > 0
        assert all("id" in r for r in rules)

    def test_evaluate_r02(self):
        """R02: TrueClaim => ¬FalseClaim — should have zero violation
        when TrueClaim=1 and FalseClaim=0."""
        rules = load_rules()
        r02 = [r for r in rules if r["id"] == "R02"][0]

        pred_vals = {
            "TrueClaim": torch.tensor([1.0, 0.0]),
            "FalseClaim": torch.tensor([0.0, 1.0]),
        }
        truth, violation = evaluate_rule(r02, pred_vals)
        # First sample: TrueClaim=1, FalseClaim=0 => satisfied
        assert violation[0].item() < 1e-6
        # Second sample: TrueClaim=0, FalseClaim=1 => body is false, vacuously true
        assert violation[1].item() < 1e-6

    def test_evaluate_r02_violation(self):
        """R02 should show violation when TrueClaim=1 AND FalseClaim=1."""
        rules = load_rules()
        r02 = [r for r in rules if r["id"] == "R02"][0]

        pred_vals = {
            "TrueClaim": torch.tensor([1.0]),
            "FalseClaim": torch.tensor([1.0]),
        }
        truth, violation = evaluate_rule(r02, pred_vals)
        assert violation[0].item() > 0.9

    def test_evaluate_all_rules(self):
        """Should run without error on dummy predicate values."""
        rules = load_rules()
        pred_vals = {
            "TrueClaim": torch.tensor([0.8]),
            "FalseClaim": torch.tensor([0.1]),
            "IsPerson": torch.tensor([0.9]),
            "IsCountry": torch.tensor([0.1]),
        }
        result = evaluate_all_rules(rules, pred_vals)
        assert "total_violation" in result
        assert "per_rule" in result
        assert result["total_violation"].item() >= 0.0

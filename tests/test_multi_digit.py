"""Tests for multi-digit addition dataset, constraints, and model."""

import pytest
import torch
import random

from data.multi_digit_addition import (
    MultiDigitAdditionDataset,
    multi_digit_collate,
    has_carry,
    count_carries,
    decompose_sum,
    render_number,
)
from symbolic.multi_digit_constraints import (
    carry_constraint_soft,
    verify_multi_digit,
)


# ─── Dataset tests ──────────────────────────────────────────

class TestCarryLogic:
    def test_no_carry(self):
        ones, tens = has_carry(11, 22)
        assert ones == 0 and tens == 0

    def test_ones_carry_only(self):
        ones, tens = has_carry(17, 15)
        assert ones == 1 and tens == 0  # 7+5=12 → carry, 1+1+1=3 no carry

    def test_both_carries(self):
        ones, tens = has_carry(99, 99)
        assert ones == 1 and tens == 1  # 9+9=18 carry, 9+9+1=19 carry

    def test_count_carries(self):
        assert count_carries(11, 22) == 0
        assert count_carries(17, 15) == 1
        assert count_carries(99, 99) == 2

    def test_decompose_sum(self):
        h, t, o = decompose_sum(198)
        assert h == 1 and t == 9 and o == 8

    def test_decompose_single_digit(self):
        h, t, o = decompose_sum(5)
        assert h == 0 and t == 0 and o == 5


class TestRenderNumber:
    def test_output_shape(self):
        img = render_number(42, img_size=28)
        assert img.shape == (28, 56)  # Two 28×28 digits concatenated, raw numpy

    def test_different_numbers_differ(self):
        img1 = render_number(42, img_size=28)
        img2 = render_number(37, img_size=28)
        assert not (img1 == img2).all()


class TestMultiDigitDataset:
    def test_train_split_no_carries(self):
        ds = MultiDigitAdditionDataset(split="train", n_samples=200, seed=42)
        for i in range(len(ds)):
            sample = ds[i]
            a = sample["a_tens"].item() * 10 + sample["a_ones"].item()
            b = sample["b_tens"].item() * 10 + sample["b_ones"].item()
            assert count_carries(a, b) == 0, f"Train sample has carry: {a} + {b}"

    def test_comp_split_has_carries(self):
        ds = MultiDigitAdditionDataset(split="comp_test", n_samples=200, seed=42)
        carry_count = 0
        for i in range(len(ds)):
            sample = ds[i]
            a = sample["a_tens"].item() * 10 + sample["a_ones"].item()
            b = sample["b_tens"].item() * 10 + sample["b_ones"].item()
            if count_carries(a, b) >= 1:
                carry_count += 1
        assert carry_count == len(ds), "All comp_test should have carries"

    def test_hard_split_two_carries(self):
        ds = MultiDigitAdditionDataset(split="hard_test", n_samples=200, seed=42)
        for i in range(len(ds)):
            sample = ds[i]
            a = sample["a_tens"].item() * 10 + sample["a_ones"].item()
            b = sample["b_tens"].item() * 10 + sample["b_ones"].item()
            assert count_carries(a, b) == 2, f"Hard sample should have 2 carries: {a} + {b}"

    def test_sum_correctness(self):
        ds = MultiDigitAdditionDataset(split="comp_test", n_samples=100, seed=42)
        for i in range(len(ds)):
            sample = ds[i]
            a = sample["a_tens"].item() * 10 + sample["a_ones"].item()
            b = sample["b_tens"].item() * 10 + sample["b_ones"].item()
            expected = a + b
            got = (
                sample["sum_hundreds"].item() * 100 +
                sample["sum_tens"].item() * 10 +
                sample["sum_ones"].item()
            )
            assert expected == got, f"{a} + {b} = {expected}, got {got}"

    def test_collate(self):
        ds = MultiDigitAdditionDataset(split="train", n_samples=32, seed=42)
        batch = multi_digit_collate([ds[i] for i in range(4)])
        assert batch["img_a"].shape == (4, 1, 28, 56)
        assert batch["img_b"].shape == (4, 1, 28, 56)
        assert batch["a_tens"].shape == (4,)

    def test_image_shapes(self):
        ds = MultiDigitAdditionDataset(split="train", n_samples=10, seed=42)
        sample = ds[0]
        assert sample["img_a"].shape == (1, 28, 56)
        assert sample["img_b"].shape == (1, 28, 56)


# ─── Constraint tests ──────────────────────────────────────

class TestCarryConstraint:
    def test_correct_predictions_low_loss(self):
        """If input probs are one-hot for correct digits, loss should be low."""
        B = 4
        # 23 + 45 = 68 (no carry)
        p_a_ones = torch.zeros(B, 10)
        p_a_ones[:, 3] = 1.0  # ones digit = 3
        p_a_tens = torch.zeros(B, 10)
        p_a_tens[:, 2] = 1.0  # tens digit = 2
        p_b_ones = torch.zeros(B, 10)
        p_b_ones[:, 5] = 1.0
        p_b_tens = torch.zeros(B, 10)
        p_b_tens[:, 4] = 1.0

        p_s_ones = torch.zeros(B, 10)
        p_s_ones[:, 8] = 1.0  # 3+5=8
        p_s_tens = torch.zeros(B, 10)
        p_s_tens[:, 6] = 1.0  # 2+4=6
        p_s_hund = torch.zeros(B, 10)
        p_s_hund[:, 0] = 1.0  # no hundreds

        loss, info = carry_constraint_soft(
            p_a_ones, p_a_tens, p_b_ones, p_b_tens,
            p_s_ones, p_s_tens, p_s_hund,
        )
        assert loss.item() < 0.1

    def test_wrong_sum_high_loss(self):
        """If predicted sum is wrong, loss should be higher."""
        B = 4
        p_a_ones = torch.zeros(B, 10)
        p_a_ones[:, 3] = 1.0
        p_a_tens = torch.zeros(B, 10)
        p_a_tens[:, 2] = 1.0
        p_b_ones = torch.zeros(B, 10)
        p_b_ones[:, 5] = 1.0
        p_b_tens = torch.zeros(B, 10)
        p_b_tens[:, 4] = 1.0

        # Wrong: sum says 99 instead of 68
        p_s_ones = torch.zeros(B, 10)
        p_s_ones[:, 9] = 1.0
        p_s_tens = torch.zeros(B, 10)
        p_s_tens[:, 9] = 1.0
        p_s_hund = torch.zeros(B, 10)
        p_s_hund[:, 0] = 1.0

        loss, _ = carry_constraint_soft(
            p_a_ones, p_a_tens, p_b_ones, p_b_tens,
            p_s_ones, p_s_tens, p_s_hund,
        )
        assert loss.item() > 0.5


class TestVerifyMultiDigit:
    def test_correct_no_violations(self):
        # 23 + 45 = 068
        violations, csr = verify_multi_digit(
            torch.tensor([2]), torch.tensor([3]),
            torch.tensor([4]), torch.tensor([5]),
            torch.tensor([8]), torch.tensor([6]), torch.tensor([0]),
        )
        assert csr == 1.0
        assert not violations[0]

    def test_wrong_sum_violation(self):
        # 23 + 45 = 099 (wrong)
        violations, csr = verify_multi_digit(
            torch.tensor([2]), torch.tensor([3]),
            torch.tensor([4]), torch.tensor([5]),
            torch.tensor([9]), torch.tensor([9]), torch.tensor([0]),
        )
        assert csr == 0.0
        assert violations[0]

    def test_carry_case(self):
        # 37 + 15 = 052
        violations, csr = verify_multi_digit(
            torch.tensor([3]), torch.tensor([7]),
            torch.tensor([1]), torch.tensor([5]),
            torch.tensor([2]), torch.tensor([5]), torch.tensor([0]),
        )
        assert csr == 1.0


# ─── Model tests ──────────────────────────────────────────

class TestMultiDigitModel:
    def test_forward_neural(self):
        from models.nst_multi_digit import MultiDigitModel
        model = MultiDigitModel(mode="neural")
        img_a = torch.randn(2, 1, 28, 56)
        img_b = torch.randn(2, 1, 28, 56)
        result = model(img_a, img_b)
        assert "probs_s_ones" in result
        assert result["probs_s_ones"].shape == (2, 10)

    def test_forward_soft(self):
        from models.nst_multi_digit import MultiDigitModel
        model = MultiDigitModel(mode="soft")
        img_a = torch.randn(2, 1, 28, 56)
        img_b = torch.randn(2, 1, 28, 56)
        result = model(img_a, img_b)
        assert "probs_s_ones" in result
        assert result["probs_s_ones"].shape == (2, 10)

    def test_forward_with_labels(self):
        from models.nst_multi_digit import MultiDigitModel
        model = MultiDigitModel(mode="soft")
        img_a = torch.randn(2, 1, 28, 56)
        img_b = torch.randn(2, 1, 28, 56)
        result = model(
            img_a, img_b,
            a_tens=torch.tensor([2, 3]),
            a_ones=torch.tensor([3, 4]),
            b_tens=torch.tensor([4, 5]),
            b_ones=torch.tensor([5, 6]),
            sum_ones=torch.tensor([8, 0]),
            sum_tens=torch.tensor([6, 0]),
            sum_hundreds=torch.tensor([0, 1]),
        )
        assert "loss_digit" in result
        assert "loss_constraint" in result
        assert "csr" in result

    def test_predict(self):
        from models.nst_multi_digit import MultiDigitModel
        model = MultiDigitModel(mode="soft")
        img_a = torch.randn(2, 1, 28, 56)
        img_b = torch.randn(2, 1, 28, 56)
        preds = model.predict(img_a, img_b)
        assert "pred_s_ones" in preds
        assert preds["pred_s_ones"].shape == (2,)

"""Tests for config validation, NaN safety, and CSR correctness.

Covers:
  - YAML numeric casting (1e-3 string → float)
  - Range validation
  - NaN/Inf rejection
  - CSR computation under NaN contamination
  - NaN fail-fast in training loop
"""

import math
import pytest
import torch

from training.config_validation import (
    cast_value,
    validate_section,
    load_and_validate_config,
    ConfigValidationError,
)
from symbolic.multi_digit_constraints import verify_multi_digit


# ─── Config Validation ──────────────────────────────────────

class TestCastValue:
    """Test numeric casting of YAML string values."""

    def test_float_string(self):
        """YAML 1e-3 is parsed as string '1e-3'; must become 0.001."""
        assert cast_value("lr", "1e-3") == pytest.approx(0.001)

    def test_float_already_float(self):
        """Already-float values pass through."""
        assert cast_value("lr", 0.001) == pytest.approx(0.001)

    def test_int_from_string(self):
        """Integer fields from string."""
        assert cast_value("epochs", "30") == 30
        assert isinstance(cast_value("epochs", "30"), int)

    def test_int_from_sci_notation(self):
        """1e2 → 100 (int)."""
        assert cast_value("epochs", "1e2") == 100

    def test_int_already_int(self):
        assert cast_value("batch_size", 64) == 64

    def test_none_returns_default(self):
        assert cast_value("epochs", None) == 30  # default in spec

    def test_unknown_key_passthrough(self):
        """Unknown keys are returned unchanged."""
        assert cast_value("my_custom_key", "hello") == "hello"

    def test_invalid_string_raises(self):
        """Unparseable strings should raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="cannot cast"):
            cast_value("lr", "not_a_number")

    def test_below_minimum_raises(self):
        with pytest.raises(ConfigValidationError, match="below minimum"):
            cast_value("epochs", 0)

    def test_above_maximum_raises(self):
        with pytest.raises(ConfigValidationError, match="above maximum"):
            cast_value("lr", 100.0)

    def test_nan_raises(self):
        with pytest.raises(ConfigValidationError, match="NaN or Inf"):
            cast_value("lr", float("nan"))

    def test_inf_raises(self):
        with pytest.raises(ConfigValidationError):
            cast_value("lr", float("inf"))


class TestValidateSection:
    """Test section-level validation."""

    def test_mixed_section(self):
        section = {"lr": "1e-3", "epochs": 30, "mode": "neural", "device": "auto"}
        result = validate_section(section, "training")
        assert result["lr"] == pytest.approx(0.001)
        assert result["epochs"] == 30
        assert result["mode"] == "neural"  # non-numeric passthrough
        assert result["device"] == "auto"  # non-numeric passthrough

    def test_nested_section(self):
        cfg = {"training": {"lr": "1e-3", "epochs": "50"}, "data": {"n_train": "5000"}}
        result = validate_section(cfg, "root")
        assert result["training"]["lr"] == pytest.approx(0.001)
        assert result["training"]["epochs"] == 50
        assert result["data"]["n_train"] == 5000

    def test_invalid_nested_raises(self):
        cfg = {"training": {"lr": "bad"}}
        with pytest.raises(ConfigValidationError):
            validate_section(cfg, "root")


class TestLoadAndValidateConfig:
    """Test end-to-end config loading."""

    def test_valid_config(self, tmp_path):
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(
            "training:\n  lr: 1e-3\n  epochs: 30\n  batch_size: 64\n"
            "data:\n  n_train: 5000\n"
        )
        cfg = load_and_validate_config(str(cfg_path))
        assert cfg["training"]["lr"] == pytest.approx(0.001)
        assert cfg["training"]["epochs"] == 30
        assert cfg["data"]["n_train"] == 5000

    def test_all_string_numerics(self, tmp_path):
        """Simulate worst-case: ALL numeric values are strings (YAML 1e-3 bug)."""
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(
            "cegis:\n"
            "  max_rounds: '10'\n"
            "  inner_epochs: '15'\n"
            "  lr: '1e-3'\n"
            "  batch_size: '64'\n"
            "  lagrangian_epsilon: '0.05'\n"
            "  lagrangian_alpha: '1e-2'\n"
        )
        cfg = load_and_validate_config(str(cfg_path))
        assert cfg["cegis"]["lr"] == pytest.approx(0.001)
        assert cfg["cegis"]["lagrangian_alpha"] == pytest.approx(0.01)
        assert cfg["cegis"]["max_rounds"] == 10
        assert cfg["cegis"]["batch_size"] == 64

    def test_empty_config_raises(self, tmp_path):
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")
        with pytest.raises(ConfigValidationError, match="empty"):
            load_and_validate_config(str(cfg_path))


# ─── NaN-safe CSR ────────────────────────────────────────────

class TestNaNSafeCSR:
    """CSR must never silently become 1.0 due to NaN contamination."""

    def test_correct_predictions_csr_one(self):
        """Normal correct predictions → CSR = 1.0."""
        # Sample 0: 23 + 45 = 68 → s_ones=8, s_tens=6, s_hund=0
        # Sample 1: 34 + 56 = 90 → s_ones=0, s_tens=9, s_hund=0
        violations, csr = verify_multi_digit(
            torch.tensor([2, 3]), torch.tensor([3, 4]),
            torch.tensor([4, 5]), torch.tensor([5, 6]),
            torch.tensor([8, 0]), torch.tensor([6, 9]),
            torch.tensor([0, 0]),
        )
        assert csr == 1.0
        assert not violations.any()

    def test_nan_predictions_are_violations(self):
        """NaN in predictions must be treated as constraint violations."""
        violations, csr = verify_multi_digit(
            torch.tensor([2, float("nan")]),
            torch.tensor([3, 4]),
            torch.tensor([4, 5]),
            torch.tensor([5, 6]),
            torch.tensor([8, 0]),
            torch.tensor([6, 0]),
            torch.tensor([0, 1]),
        )
        # Sample 1 is correct, sample 2 has NaN → violation
        assert violations[1].item() is True
        assert csr < 1.0  # CSR must NOT be 1.0

    def test_all_nan_predictions(self):
        """All NaN predictions → CSR = 0.0."""
        nan = float("nan")
        violations, csr = verify_multi_digit(
            torch.tensor([nan, nan]),
            torch.tensor([nan, nan]),
            torch.tensor([nan, nan]),
            torch.tensor([nan, nan]),
            torch.tensor([nan, nan]),
            torch.tensor([nan, nan]),
            torch.tensor([nan, nan]),
        )
        assert csr == 0.0
        assert violations.all()

    def test_wrong_predictions_csr_zero(self):
        """Wrong predictions → CSR = 0.0 (sanity)."""
        violations, csr = verify_multi_digit(
            torch.tensor([2]), torch.tensor([3]),
            torch.tensor([4]), torch.tensor([5]),
            torch.tensor([9]), torch.tensor([9]),  # wrong
            torch.tensor([9]),  # wrong
        )
        assert csr == 0.0


# ─── Neural Model NaN Stability ──────────────────────────────

class TestNeuralModelNaN:
    """Verify the double-log NaN fix in MultiDigitModel neural mode."""

    def test_neural_forward_no_nan(self):
        """Neural mode forward pass should not produce NaN loss."""
        from models.nst_multi_digit import MultiDigitModel

        torch.manual_seed(42)
        model = MultiDigitModel(mode="neural")
        img_a = torch.randn(4, 1, 28, 56)
        img_b = torch.randn(4, 1, 28, 56)

        result = model(
            img_a, img_b,
            a_tens=torch.tensor([2, 3, 1, 4]),
            a_ones=torch.tensor([3, 4, 5, 6]),
            b_tens=torch.tensor([4, 5, 6, 7]),
            b_ones=torch.tensor([5, 6, 7, 8]),
            sum_ones=torch.tensor([8, 0, 2, 4]),
            sum_tens=torch.tensor([6, 0, 2, 4]),
            sum_hundreds=torch.tensor([0, 1, 0, 1]),
        )

        assert "loss_total" in result
        assert not torch.isnan(result["loss_total"]).any(), \
            f"Neural loss_total is NaN: {result['loss_total']}"
        assert not torch.isinf(result["loss_total"]).any(), \
            f"Neural loss_total is Inf: {result['loss_total']}"

    def test_neural_loss_is_finite_after_backward(self):
        """Gradient flow should not produce NaN."""
        from models.nst_multi_digit import MultiDigitModel

        torch.manual_seed(42)
        model = MultiDigitModel(mode="neural")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for step in range(5):
            img_a = torch.randn(4, 1, 28, 56)
            img_b = torch.randn(4, 1, 28, 56)
            result = model(
                img_a, img_b,
                a_tens=torch.randint(0, 10, (4,)),
                a_ones=torch.randint(0, 10, (4,)),
                b_tens=torch.randint(0, 10, (4,)),
                b_ones=torch.randint(0, 10, (4,)),
                sum_ones=torch.randint(0, 10, (4,)),
                sum_tens=torch.randint(0, 10, (4,)),
                sum_hundreds=torch.randint(0, 2, (4,)),
            )
            loss = result["loss_total"]
            assert not torch.isnan(loss), f"NaN at step {step}"
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

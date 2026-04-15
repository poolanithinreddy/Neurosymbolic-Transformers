"""Tests for ECCG, temperature scaling, class weights, config parsing.

Covers:
  - ConstraintGate shapes and properties
  - gated_fever_constraint_loss computation
  - facts_to_gate_features extraction
  - Temperature scaling (learn + apply)
  - Class weight computation from label distribution
  - Config parsing alignment (nested + flat fallbacks)
  - Gated mode integration (end-to-end smoke)
"""

from __future__ import annotations

import os
import sys
import math
import pytest
import tempfile

import torch

THIS_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


# ── ConstraintGate ─────────────────────────────────────────────

class TestConstraintGate:
    """ECCG ConstraintGate must produce correct shapes and be trainable."""

    def test_gate_output_shape(self):
        from symbolic.constraint_gating import ConstraintGate, _N_SIGNALS, _N_CONSTRAINTS
        gate = ConstraintGate()
        x = torch.randn(8, _N_SIGNALS)
        out = gate(x)
        assert out.shape == (8, _N_CONSTRAINTS)

    def test_gate_output_in_01(self):
        from symbolic.constraint_gating import ConstraintGate, _N_SIGNALS
        gate = ConstraintGate()
        x = torch.randn(32, _N_SIGNALS)
        out = gate(x)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_gate_init_bias_affects_output(self):
        from symbolic.constraint_gating import ConstraintGate, _N_SIGNALS
        gate_low = ConstraintGate(init_bias=-2.0)
        gate_high = ConstraintGate(init_bias=2.0)
        x = torch.zeros(1, _N_SIGNALS)  # zero input isolates bias effect
        out_low = gate_low(x)
        out_high = gate_high(x)
        # High bias → higher gate values
        assert out_high.mean() > out_low.mean()

    def test_gate_param_count_small(self):
        from symbolic.constraint_gating import ConstraintGate
        gate = ConstraintGate(hidden_dim=16)
        n_params = sum(p.numel() for p in gate.parameters())
        assert n_params < 1000, f"Gate should be small, got {n_params} params"

    def test_gate_gradients_flow(self):
        from symbolic.constraint_gating import ConstraintGate, _N_SIGNALS
        gate = ConstraintGate()
        x = torch.randn(4, _N_SIGNALS, requires_grad=False)
        out = gate(x)
        loss = out.mean()
        loss.backward()
        # Check that gradients exist on gate parameters
        for p in gate.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()

    def test_gate_single_sample(self):
        from symbolic.constraint_gating import ConstraintGate, _N_SIGNALS
        gate = ConstraintGate()
        x = torch.randn(1, _N_SIGNALS)
        out = gate(x)
        assert out.shape == (1, 5)

    def test_gate_dropout_in_eval_mode(self):
        from symbolic.constraint_gating import ConstraintGate, _N_SIGNALS
        gate = ConstraintGate(dropout=0.5)
        x = torch.randn(4, _N_SIGNALS)
        gate.eval()
        out1 = gate(x)
        out2 = gate(x)
        # In eval mode, dropout is off → deterministic
        assert torch.allclose(out1, out2)


# ── facts_to_gate_features ─────────────────────────────────────

class TestGateFeatures:
    """Feature extraction for gate network."""

    def test_feature_shape(self):
        from symbolic.constraint_gating import facts_to_gate_features, _N_SIGNALS
        from symbolic.fever_constraints import extract_structured_facts

        facts = [
            extract_structured_facts("Population is 500", "Population was 800"),
            extract_structured_facts("He won", "He won"),
        ]
        features = facts_to_gate_features(facts, torch.device("cpu"))
        assert features.shape == (2, _N_SIGNALS)

    def test_features_bounded_01(self):
        from symbolic.constraint_gating import facts_to_gate_features
        from symbolic.fever_constraints import extract_structured_facts

        facts = [
            extract_structured_facts("500 items", "800 items"),
            extract_structured_facts("March 2001", "June 2005"),
            extract_structured_facts("He never went", "He went"),
        ]
        features = facts_to_gate_features(facts, torch.device("cpu"))
        assert (features >= 0).all()
        assert (features <= 1).all()

    def test_contradiction_signals(self):
        from symbolic.constraint_gating import facts_to_gate_features
        from symbolic.fever_constraints import extract_structured_facts

        facts = [
            extract_structured_facts("Population is 500", "Population was 800"),
        ]
        features = facts_to_gate_features(facts, torch.device("cpu"))
        # number_contradiction should be 1
        assert features[0, 1] == 1.0  # index 1 = num_contra


# ── gated_fever_constraint_loss ────────────────────────────────

class TestGatedConstraintLoss:
    """ECCG gated constraint loss computation."""

    def test_gated_loss_shape(self):
        from symbolic.constraint_gating import ConstraintGate, gated_fever_constraint_loss
        from symbolic.fever_constraints import extract_structured_facts

        gate = ConstraintGate()
        facts = [
            extract_structured_facts("Population is 500", "Population was 800"),
            extract_structured_facts("He won", "He won the award"),
        ]

        p_s = torch.tensor([0.8, 0.9])
        p_r = torch.tensor([0.1, 0.05])
        p_n = torch.tensor([0.1, 0.05])

        loss, info = gated_fever_constraint_loss(p_s, p_r, p_n, facts, gate=gate)
        assert loss.shape == ()
        assert loss.item() >= 0
        assert "constraint_loss_total" in info
        assert "gate_mean_c1" in info

    def test_gated_loss_without_gate_fallback(self):
        from symbolic.constraint_gating import gated_fever_constraint_loss
        from symbolic.fever_constraints import extract_structured_facts

        facts = [
            extract_structured_facts("Population is 500", "Population was 800"),
        ]
        p_s = torch.tensor([0.8])
        p_r = torch.tensor([0.1])
        p_n = torch.tensor([0.1])

        loss, info = gated_fever_constraint_loss(p_s, p_r, p_n, facts, gate=None)
        assert loss.shape == ()
        assert "gate_mean_c1" not in info  # no gate info when gate is None

    def test_gated_loss_empty_facts(self):
        from symbolic.constraint_gating import gated_fever_constraint_loss
        loss, info = gated_fever_constraint_loss(
            torch.tensor([0.8]), torch.tensor([0.1]), torch.tensor([0.1]),
            facts_batch=[], gate=None,
        )
        assert loss.item() == 0.0

    def test_gated_loss_gradients_flow_to_gate(self):
        from symbolic.constraint_gating import ConstraintGate, gated_fever_constraint_loss
        from symbolic.fever_constraints import extract_structured_facts

        gate = ConstraintGate()
        facts = [
            extract_structured_facts("500 items", "800 items"),
            extract_structured_facts("He never went", "He went"),
        ]
        p_s = torch.tensor([0.8, 0.7], requires_grad=True)
        p_r = torch.tensor([0.1, 0.2], requires_grad=True)
        p_n = torch.tensor([0.1, 0.1], requires_grad=True)

        loss, _ = gated_fever_constraint_loss(p_s, p_r, p_n, facts, gate=gate)
        loss.backward()
        # Gate should have gradients
        for p in gate.parameters():
            assert p.grad is not None


# ── Temperature Scaling ────────────────────────────────────────

class TestTemperatureScaling:
    """Post-hoc temperature scaling calibration."""

    def test_temperature_scaler_no_change_at_T1(self):
        from eval.temperature_scaling import TemperatureScaler
        scaler = TemperatureScaler()
        scaler.temperature.data.fill_(1.0)
        logits = torch.randn(4, 3)
        scaled = scaler(logits)
        assert torch.allclose(logits, scaled)

    def test_temperature_scaler_reduces_confidence(self):
        from eval.temperature_scaling import TemperatureScaler
        import torch.nn.functional as F

        scaler = TemperatureScaler()
        scaler.temperature.data.fill_(2.0)  # T > 1 → softer probs
        logits = torch.tensor([[5.0, 1.0, 0.0]])
        original_probs = F.softmax(logits, dim=-1)
        scaled_probs = F.softmax(scaler(logits), dim=-1)
        # Max prob should be lower with higher temperature
        assert scaled_probs.max() < original_probs.max()

    def test_apply_temperature(self):
        from eval.temperature_scaling import apply_temperature
        logits = torch.tensor([[3.0, 1.0, 0.5]])
        probs = apply_temperature(logits, temperature=1.0)
        assert probs.shape == (1, 3)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_apply_temperature_clamp(self):
        from eval.temperature_scaling import apply_temperature
        logits = torch.tensor([[3.0, 1.0, 0.5]])
        # Very small T should be clamped, not crash
        probs = apply_temperature(logits, temperature=0.001)
        assert not torch.isnan(probs).any()


# ── Class Weights ──────────────────────────────────────────────

class TestClassWeights:
    """Class weight computation for imbalanced labels."""

    def test_class_weights_balanced(self):
        """Balanced labels → equal weights."""
        from collections import Counter

        labels = ["SUPPORTS"] * 100 + ["REFUTES"] * 100 + ["NOT ENOUGH INFO"] * 100
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        n_classes = 3
        FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

        weights = torch.tensor([
            total / (n_classes * max(1, label_counts.get(FEVER_LABELS[i], 1)))
            for i in range(n_classes)
        ])
        # All weights should be equal (1.0) for balanced data
        assert torch.allclose(weights, torch.ones(3))

    def test_class_weights_imbalanced(self):
        """Imbalanced labels → higher weight for minority class."""
        from collections import Counter

        labels = ["SUPPORTS"] * 800 + ["REFUTES"] * 100 + ["NOT ENOUGH INFO"] * 100
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        n_classes = 3
        FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

        weights = torch.tensor([
            total / (n_classes * max(1, label_counts.get(FEVER_LABELS[i], 1)))
            for i in range(n_classes)
        ])
        # SUPPORTS weight < REFUTES/NEI weight (majority class → lower weight)
        assert weights[0] < weights[1]
        assert weights[0] < weights[2]

    def test_fever_nli_wrapper_accepts_class_weights(self):
        from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
        from models.fever_nli import FeverNLIWrapper

        config = DebertaV2Config(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            num_labels=3,
            max_position_embeddings=64,
            relative_attention=False,
        )
        base = DebertaV2ForSequenceClassification(config)
        weights = torch.tensor([1.0, 2.0, 1.5])
        wrapper = FeverNLIWrapper(base, class_weights=weights)

        input_ids = torch.randint(0, 100, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)
        labels = torch.tensor([0, 1])
        out = wrapper(input_ids, mask, labels=labels)
        assert "loss" in out


# ── Config Parsing ─────────────────────────────────────────────

class TestConfigParsing:
    """Config parsing must handle both nested and flat YAML keys."""

    def test_nested_config_structure(self):
        """Nested YAML is correctly parsed."""
        import yaml

        cfg_text = """
mode: "gated"
seed: 42
device: "cpu"
model:
  name: "test-model"
  label_smoothing: 0.05
  max_length: 128
train:
  epochs: 2
  batch_size: 8
  lr: 1.0e-5
logic:
  lambda: 0.05
io:
  out_dir: "test_out"
gate:
  hidden_dim: 16
  lr_multiplier: 10.0
"""
        cfg = yaml.safe_load(cfg_text)
        assert cfg["model"]["name"] == "test-model"
        assert cfg["train"]["epochs"] == 2
        assert cfg["logic"]["lambda"] == 0.05
        assert cfg["gate"]["hidden_dim"] == 16

    def test_config_nested_with_fallback(self):
        """The training code's fallback pattern works."""
        cfg = {
            "model": {"name": "test-model"},
            "model_name": "old-flat-name",  # flat fallback
            "train": {"epochs": 5},
            "epochs": 3,  # flat fallback
        }
        # Nested takes priority
        model_cfg = cfg.get("model", {})
        name = model_cfg.get("name", cfg.get("model_name", "default"))
        assert name == "test-model"

        train_cfg = cfg.get("train", {})
        epochs = int(train_cfg.get("epochs", cfg.get("epochs", 1)))
        assert epochs == 5

    def test_gated_config_loads(self):
        """The gated config file is valid YAML."""
        import yaml

        config_path = os.path.join(PROJ_ROOT, "configs", "fever_gold_nst_gated.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            assert cfg["mode"] == "gated"
            assert "gate" in cfg
            assert cfg["gate"]["hidden_dim"] == 16
            assert cfg["gate"]["lr_multiplier"] == 10.0


# ── Integration Smoke ──────────────────────────────────────────

class TestIntegrationSmoke:
    """Quick smoke tests to verify components wire together."""

    def test_eccg_pipeline_no_crash(self):
        """Full ECCG pipeline: extract facts → gate features → gated loss."""
        from symbolic.fever_constraints import extract_batch_facts
        from symbolic.constraint_gating import (
            ConstraintGate, gated_fever_constraint_loss,
        )

        claims = ["Population is 500", "He never visited France"]
        evidences = ["Population was 800", "He visited France in 2010"]

        facts = extract_batch_facts(claims, evidences)
        gate = ConstraintGate()

        p_s = torch.tensor([0.8, 0.7])
        p_r = torch.tensor([0.1, 0.2])
        p_n = torch.tensor([0.1, 0.1])

        loss, info = gated_fever_constraint_loss(p_s, p_r, p_n, facts, gate=gate)
        assert loss.shape == ()
        assert not torch.isnan(loss)
        # Should have per-constraint gate means
        for i in range(5):
            assert f"gate_mean_c{i+1}" in info

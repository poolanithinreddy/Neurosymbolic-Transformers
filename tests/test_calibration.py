"""Tests for calibration metrics (ECE, Brier, reliability diagrams)."""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from eval.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)


class TestECE:
    def test_perfect_calibration(self):
        """Perfectly calibrated predictions → ECE = 0."""
        # 100% confidence, 100% accuracy
        probs = torch.zeros(10, 3)
        probs[:, 0] = 1.0
        labels = torch.zeros(10, dtype=torch.long)

        ece, _ = expected_calibration_error(probs, labels, n_bins=10)
        assert ece < 0.01

    def test_overconfident_has_positive_ece(self):
        """Overconfident wrong predictions → positive ECE."""
        probs = torch.zeros(10, 3)
        probs[:, 0] = 0.99  # Very confident
        probs[:, 1] = 0.005
        probs[:, 2] = 0.005
        labels = torch.ones(10, dtype=torch.long)  # All wrong

        ece, _ = expected_calibration_error(probs, labels, n_bins=10)
        assert ece > 0.5

    def test_ece_bounded(self):
        """ECE should be between 0 and 1."""
        probs = torch.randn(50, 5).softmax(dim=-1)
        labels = torch.randint(0, 5, (50,))

        ece, _ = expected_calibration_error(probs, labels)
        assert 0 <= ece <= 1.0

    def test_bin_data_structure(self):
        """Bin data should have correct keys."""
        probs = torch.randn(20, 3).softmax(dim=-1)
        labels = torch.randint(0, 3, (20,))

        _, bin_data = expected_calibration_error(probs, labels, n_bins=5)
        assert len(bin_data) == 5
        for b in bin_data:
            assert "bin_lo" in b
            assert "bin_hi" in b
            assert "count" in b
            assert "accuracy" in b
            assert "confidence" in b


class TestBrierScore:
    def test_perfect_prediction(self):
        """Perfect one-hot predictions → Brier score = 0."""
        probs = torch.zeros(5, 3)
        labels = torch.tensor([0, 1, 2, 0, 1])
        for i, l in enumerate(labels):
            probs[i, l] = 1.0

        bs = brier_score(probs, labels)
        assert bs < 0.01

    def test_uniform_prediction(self):
        """Uniform predictions → Brier score > 0."""
        probs = torch.ones(10, 4) / 4
        labels = torch.zeros(10, dtype=torch.long)

        bs = brier_score(probs, labels)
        assert bs > 0

    def test_brier_bounded(self):
        """Brier score should be non-negative."""
        probs = torch.randn(30, 5).softmax(dim=-1)
        labels = torch.randint(0, 5, (30,))

        bs = brier_score(probs, labels)
        assert bs >= 0


class TestReliabilityDiagram:
    def test_output_structure(self):
        probs = torch.randn(50, 4).softmax(dim=-1)
        labels = torch.randint(0, 4, (50,))

        data = reliability_diagram_data(probs, labels, n_bins=10)
        assert "midpoints" in data
        assert "accuracies" in data
        assert "confidences" in data
        assert "counts" in data
        assert "ece" in data
        assert len(data["midpoints"]) == 10

    def test_counts_sum_to_total(self):
        N = 100
        probs = torch.randn(N, 3).softmax(dim=-1)
        labels = torch.randint(0, 3, (N,))

        data = reliability_diagram_data(probs, labels, n_bins=10)
        assert sum(data["counts"]) == N

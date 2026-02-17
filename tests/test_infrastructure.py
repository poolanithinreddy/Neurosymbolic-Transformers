"""Tests for multi-seed runner, kinship CEGIS, calibration, rulecheck, results API."""

import json
import os
import sys

import pytest
import torch

THIS_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


# ---------------------------------------------------------------------------
# eval/rulecheck.py — no longer a stub
# ---------------------------------------------------------------------------

class TestRulecheck:
    def test_import_no_print(self, capsys):
        """Importing rulecheck should NOT print anything."""
        import importlib
        import eval.rulecheck
        importlib.reload(eval.rulecheck)
        captured = capsys.readouterr()
        assert "stub" not in captured.out

    def test_digit_add_rule_report(self):
        from eval.rulecheck import rule_satisfaction_report
        probs = torch.eye(5)  # 5 samples, perfect diagonal
        labels = torch.arange(5)
        report = rule_satisfaction_report(probs, labels, task="digit_add")
        assert report["csr"] == 1.0
        assert report["task"] == "digit_add"
        assert len(report["rules"]) == 1
        assert report["rules"][0]["rate"] == 1.0

    def test_kinship_rule_report(self):
        from eval.rulecheck import rule_satisfaction_report
        # 3 samples: depth 1 → must be parent(0)/child(1),
        #            depth 2 → grandparent(2)/grandchild(3)/sibling(4),
        #            depth 3 → ancestor(5)/descendant(6)/sibling(4)
        probs = torch.zeros(3, 8)
        probs[0, 0] = 1.0  # parent — valid for depth 1
        probs[1, 2] = 1.0  # grandparent — valid for depth 2
        probs[2, 5] = 1.0  # ancestor — valid for depth 3
        labels = torch.tensor([0, 2, 5])
        chain_lengths = [1, 2, 3]
        report = rule_satisfaction_report(probs, labels, chain_lengths, task="kinship")
        assert report["csr"] == 1.0
        assert report["task"] == "kinship"


# ---------------------------------------------------------------------------
# eval/calibration.py — no longer a stub
# ---------------------------------------------------------------------------

class TestCalibrationEval:
    def test_evaluate_calibration(self):
        from eval.calibration import evaluate_calibration
        probs = torch.softmax(torch.randn(100, 5), dim=-1)
        labels = torch.randint(0, 5, (100,))
        report = evaluate_calibration(probs, labels, n_bins=10)
        assert "ece" in report
        assert "brier" in report
        assert 0 <= report["ece"] <= 1.0
        assert report["n_samples"] == 100
        assert len(report["bin_data"]) == 10


# ---------------------------------------------------------------------------
# eval/cogs.py — no duplicate function
# ---------------------------------------------------------------------------

class TestCogsNoDuplicate:
    def test_no_duplicate_main(self):
        """cogs.py should have exactly one 'def main()' and one evaluate."""
        cogs_path = os.path.join(PROJ_ROOT, "eval", "cogs.py")
        with open(cogs_path) as f:
            source = f.read()
        assert source.count("def main()") == 1
        assert source.count("def evaluate(") == 1


# ---------------------------------------------------------------------------
# results/__init__.py — API compatibility
# ---------------------------------------------------------------------------

class TestResultsAPI:
    def test_load_reports_accepts_list(self, tmp_path):
        """load_reports should accept a list of directories."""
        from results import load_reports
        d1 = tmp_path / "outputs_a"
        d1.mkdir()
        (d1 / "final_report.json").write_text(json.dumps({"mode": "soft", "metrics": {}}))
        d2 = tmp_path / "outputs_b"
        d2.mkdir()
        (d2 / "final_report.json").write_text(json.dumps({"mode": "hard", "metrics": {}}))
        reports = load_reports([str(d1), str(d2)])
        assert len(reports) == 2
        assert "outputs_a" in reports
        assert "outputs_b" in reports

    def test_load_reports_accepts_string(self, tmp_path):
        """load_reports should accept a single string directory."""
        from results import load_reports
        parent = tmp_path / "all_outputs"
        parent.mkdir()
        sub = parent / "exp1"
        sub.mkdir()
        (sub / "final_report.json").write_text(json.dumps({"mode": "lag"}))
        reports = load_reports(str(parent))
        assert "exp1" in reports

    def test_render_results_accepts_dict(self):
        """render_results should accept a dict of reports directly."""
        from results import render_results
        reports = {
            "NST-Soft": {
                "metrics": {
                    "iid": {"sum_acc": 0.95, "csr": 0.99, "digit_acc_a": 0.97, "digit_acc_b": 0.96, "ece": 0.02},
                    "comp": {"sum_acc": 0.80, "csr": 0.90, "ece": 0.05},
                    "compositional_gap": 0.15,
                }
            }
        }
        md = render_results(reports, fmt="markdown")
        assert "NST-Soft" in md
        assert "0.950" in md

    def test_render_results_fmt_alias(self):
        """render_results should accept fmt= as alias for output_format=."""
        from results import render_results
        reports = {"A": {"metrics": {"iid": {"sum_acc": 0.5, "digit_acc_a": 0, "digit_acc_b": 0, "csr": 0, "ece": 0}, "comp": {"sum_acc": 0.4, "csr": 0, "ece": 0}, "compositional_gap": 0.1}}}
        latex = render_results(reports, fmt="latex")
        assert "\\begin{table}" in latex


# ---------------------------------------------------------------------------
# results aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_aggregate_results(self):
        from results import aggregate_results
        r1 = {"metrics": {"iid": {"sum_acc": 0.9, "csr": 0.95}, "comp": {"sum_acc": 0.7, "csr": 0.8}, "compositional_gap": 0.2}}
        r2 = {"metrics": {"iid": {"sum_acc": 0.92, "csr": 0.96}, "comp": {"sum_acc": 0.72, "csr": 0.82}, "compositional_gap": 0.2}}
        agg = aggregate_results([r1, r2])
        assert "iid/sum_acc" in agg
        assert agg["iid/sum_acc"]["n"] == 2
        assert abs(agg["iid/sum_acc"]["mean"] - 0.91) < 0.01


# ---------------------------------------------------------------------------
# training/multi_seed.py
# ---------------------------------------------------------------------------

class TestMultiSeed:
    def test_parse_seeds(self):
        from training.multi_seed import parse_seeds
        assert parse_seeds("42,43,44") == [42, 43, 44]
        assert parse_seeds("1") == [1]
        assert parse_seeds("10, 20, 30") == [10, 20, 30]

    def test_resolve_train_fn(self):
        from training.multi_seed import _resolve_train_fn
        fn = _resolve_train_fn("train")
        assert callable(fn)
        fn2 = _resolve_train_fn("kinship_cegis")
        assert callable(fn2)
        with pytest.raises(ValueError):
            _resolve_train_fn("unknown_task")


# ---------------------------------------------------------------------------
# Kinship CEGIS — verify_fn and CE dataset
# ---------------------------------------------------------------------------

class TestKinshipCEGIS:
    def test_kinship_ce_to_dataset(self):
        from training.cegis import kinship_ce_to_dataset
        ces = [
            {"input_ids": torch.zeros(10, dtype=torch.long), "label": torch.tensor(0), "chain_length": 1},
            {"input_ids": torch.ones(10, dtype=torch.long), "label": torch.tensor(2), "chain_length": 2},
        ]
        ds = kinship_ce_to_dataset(ces, "cpu")
        assert len(ds) == 2
        item = ds[0]
        assert "input_ids" in item
        assert "label" in item

    def test_kinship_verify_fn_finds_violations(self):
        """Verify fn should find counterexamples from a random model."""
        from training.cegis import kinship_verify_fn
        from models.nst_kinship import KinshipTransformer
        from data.kinship import KinshipDataset, kinship_collate_fn
        from torch.utils.data import DataLoader

        model = KinshipTransformer(d_model=32, n_heads=2, n_layers=1, d_ff=64, max_seq_len=128)
        ds = KinshipDataset("train", n_samples=50, max_seq_len=128, seed=42, balanced_sampling=False)
        loader = DataLoader(ds, batch_size=16, collate_fn=kinship_collate_fn)
        ces = kinship_verify_fn(model, loader, "cpu", max_ce=100)
        # Random model should produce many violations
        assert len(ces) > 0
        assert "input_ids" in ces[0]
        assert "label" in ces[0]


# ---------------------------------------------------------------------------
# Calibration metrics integration
# ---------------------------------------------------------------------------

class TestCalibrationMetrics:
    def test_ece_perfect(self):
        from eval.calibration_metrics import expected_calibration_error
        probs = torch.eye(3)  # Perfect predictions
        labels = torch.arange(3)
        ece, bins = expected_calibration_error(probs, labels, n_bins=5)
        assert ece < 0.01  # Nearly perfect calibration

    def test_brier_score_range(self):
        from eval.calibration_metrics import brier_score
        probs = torch.softmax(torch.randn(50, 5), dim=-1)
        labels = torch.randint(0, 5, (50,))
        bs = brier_score(probs, labels)
        assert 0 <= bs <= 2.0  # Brier score bounded

    def test_reliability_diagram(self):
        from eval.calibration_metrics import reliability_diagram_data
        probs = torch.softmax(torch.randn(100, 5), dim=-1)
        labels = torch.randint(0, 5, (100,))
        data = reliability_diagram_data(probs, labels, n_bins=10)
        assert "midpoints" in data
        assert "ece" in data
        assert len(data["midpoints"]) == 10

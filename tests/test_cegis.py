"""Tests for the CEGIS training loop."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from training.cegis import (
    CEGISConfig,
    CEGISLog,
    CEGISTrainer,
)


# ─── Mock model and dataset for unit testing ──────────────

class _MockModel(nn.Module):
    """Tiny model that returns structured dict like our real models."""

    def __init__(self, always_violate: bool = False):
        super().__init__()
        self.linear = nn.Linear(4, 10)
        self.always_violate = always_violate
        self._call_count = 0

    def forward(self, **kwargs):
        x = kwargs.get("x", torch.randn(1, 4))
        logits = self.linear(x)
        probs = torch.softmax(logits, dim=-1)
        B = x.size(0)
        return {
            "probs_s_ones": probs,
            "probs_s_tens": probs,
            "probs_s_hund": probs,
            "probs_a_tens": probs,
            "probs_a_ones": probs,
            "probs_b_tens": probs,
            "probs_b_ones": probs,
            "loss_digit": torch.tensor(1.0, requires_grad=True),
            "loss_constraint": torch.tensor(0.5, requires_grad=True),
            "csr": 0.5,
        }


class _MockDataset(Dataset):
    def __init__(self, n: int = 100):
        self.n = n
        self.data = [{"x": torch.randn(4), "label": torch.randint(0, 10, ())} for _ in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx]


def _mock_verify(model, dataloader, device, max_ce=100):
    """Returns some counterexamples on first call, none after."""
    model._call_count += 1
    if model._call_count <= 1:
        return [{"x": torch.randn(4), "label": torch.randint(0, 10, ())} for _ in range(5)]
    return []


def _mock_ce_to_dataset(ce_list, device):
    return _MockDataset(n=len(ce_list))


def _mock_collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


# ─── Tests ──────────────────────────────────────────────────

class TestCEGISConfig:
    def test_defaults(self):
        cfg = CEGISConfig()
        assert cfg.max_rounds == 10
        assert cfg.inner_epochs == 10
        assert cfg.ce_oversample == 3

    def test_from_dict(self):
        cfg = CEGISConfig(max_rounds=5, lr=0.01)
        assert cfg.max_rounds == 5
        assert cfg.lr == 0.01


class TestCEGISLog:
    def test_add_round(self):
        log = CEGISLog()
        log.add_round(
            round_num=1, n_counterexamples=10,
            train_loss=0.5, constraint_loss=0.3,
            lam=0.1, csr=0.8, elapsed=5.0,
        )
        assert len(log.rounds) == 1
        assert log.total_counterexamples == 10

    def test_multiple_rounds(self):
        log = CEGISLog()
        log.add_round(1, 10, 0.5, 0.3, 0.1, 0.8, 5.0)
        log.add_round(2, 5, 0.3, 0.1, 0.2, 0.9, 4.0)
        log.add_round(3, 0, 0.1, 0.01, 0.3, 0.99, 3.0)
        assert log.total_counterexamples == 15
        assert len(log.rounds) == 3


class TestCEGISTrainer:
    def test_convergence(self, tmp_path):
        """CEGIS should converge when verifier returns empty CE list."""
        model = _MockModel()
        train_ds = _MockDataset(50)
        verify_ds = _MockDataset(50)

        config = CEGISConfig(
            max_rounds=5,
            inner_epochs=2,
            batch_size=16,
            device="cpu",
            outdir=str(tmp_path / "out"),
        )

        trainer = CEGISTrainer(
            model=model,
            train_dataset=train_ds,
            verify_dataset=verify_ds,
            verify_fn=_mock_verify,
            ce_to_dataset_fn=_mock_ce_to_dataset,
            collate_fn=_mock_collate,
            config=config,
        )

        report = trainer.run()

        assert report["converged"] is True
        assert report["convergence_round"] == 2  # First round: 5 CEs; second: 0
        assert report["total_rounds"] == 2
        assert report["total_counterexamples"] == 5

    def test_ce_buffer_grows(self, tmp_path):
        """CE buffer should accumulate across rounds."""
        model = _MockModel()
        model._call_count = -1  # Will always return CEs

        def always_violate(model, dl, device, max_ce=100):
            return [{"x": torch.randn(4), "label": torch.randint(0, 10, ())} for _ in range(3)]

        config = CEGISConfig(
            max_rounds=3,
            inner_epochs=1,
            batch_size=16,
            device="cpu",
            outdir=str(tmp_path / "out2"),
        )

        trainer = CEGISTrainer(
            model=model,
            train_dataset=_MockDataset(50),
            verify_dataset=_MockDataset(50),
            verify_fn=always_violate,
            ce_to_dataset_fn=_mock_ce_to_dataset,
            collate_fn=_mock_collate,
            config=config,
        )

        report = trainer.run()

        assert not report["converged"]
        assert report["total_counterexamples"] == 9  # 3 per round × 3 rounds
        assert len(trainer.ce_buffer) == 9

    def test_lagrangian_updates(self, tmp_path):
        """Lambda should change across rounds."""
        model = _MockModel()
        config = CEGISConfig(
            max_rounds=3,
            inner_epochs=3,
            batch_size=16,
            device="cpu",
            outdir=str(tmp_path / "out3"),
        )

        trainer = CEGISTrainer(
            model=model,
            train_dataset=_MockDataset(50),
            verify_dataset=_MockDataset(50),
            verify_fn=_mock_verify,
            ce_to_dataset_fn=_mock_ce_to_dataset,
            collate_fn=_mock_collate,
            config=config,
        )

        report = trainer.run()
        assert report["final_lambda"] >= 0.0

    def test_saves_artifacts(self, tmp_path):
        """Run should save model, log, and config."""
        import os

        model = _MockModel()
        outdir = str(tmp_path / "out4")
        config = CEGISConfig(
            max_rounds=2,
            inner_epochs=1,
            batch_size=16,
            device="cpu",
            outdir=outdir,
        )

        trainer = CEGISTrainer(
            model=model,
            train_dataset=_MockDataset(50),
            verify_dataset=_MockDataset(50),
            verify_fn=_mock_verify,
            ce_to_dataset_fn=_mock_ce_to_dataset,
            collate_fn=_mock_collate,
            config=config,
        )

        trainer.run()

        assert os.path.exists(os.path.join(outdir, "ckpt", "model.pt"))
        assert os.path.exists(os.path.join(outdir, "cegis_log.json"))
        assert os.path.exists(os.path.join(outdir, "lambda_trajectory.json"))
        assert os.path.exists(os.path.join(outdir, "config.yaml"))

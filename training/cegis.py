"""Neural CEGIS — Counterexample-Guided Inductive Synthesis for neural networks.

This is the core algorithmic contribution. Training alternates between:
  1. LEARNER: train the model (Lagrangian inner loop) for N epochs.
  2. VERIFIER: find inputs where the model violates constraints.
  3. AUGMENT: add counterexamples to the training buffer.
  4. REPEAT until no counterexamples remain or max rounds exhausted.

Algorithm 1 (Neural CEGIS):
─────────────────────────────────
    Input: model M, dataset D_train, constraint C, max_rounds K
    CE_buffer ← ∅
    for round r = 1 … K:
        D_r ← D_train ∪ CE_buffer
        M ← TrainLagrangian(M, D_r, C, epochs=N)
        CE ← Verify(M, D_holdout, C)
        if |CE| == 0: break
        CE_buffer ← CE_buffer ∪ CE
    return M, CE_log

The key insight: pure Lagrangian training adjusts λ globally, but cannot
target *specific* failure modes. CEGIS finds the exact inputs where the
model fails and forces the model to learn them.

Convergence metric: counterexample count → 0 across rounds.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from symbolic.lagrangian import (
    LagrangianState,
    lagrangian_loss,
    update_dual_variable,
    save_lambda_trajectory,
)

logger = logging.getLogger("cegis")


@dataclass
class CEGISConfig:
    """Configuration for the CEGIS outer loop."""

    max_rounds: int = 10
    inner_epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 64
    max_counterexamples: int = 500
    ce_oversample: int = 3  # repeat each CE this many times in buffer
    device: str = "auto"
    seed: int = 42
    outdir: str = "outputs_cegis"

    # Lagrangian config
    lagrangian_epsilon: float = 0.05
    lagrangian_alpha: float = 0.01
    lagrangian_rho: float = 1.0
    lagrangian_lam_max: float = 10.0

    @classmethod
    def from_yaml(cls, path: str) -> "CEGISConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        cegis_d = d.get("cegis", d)
        return cls(**{k: v for k, v in cegis_d.items() if k in cls.__dataclass_fields__})


@dataclass
class CEGISLog:
    """Log of the CEGIS training process."""

    rounds: list[dict] = field(default_factory=list)
    total_counterexamples: int = 0
    converged: bool = False
    convergence_round: int | None = None

    def add_round(
        self,
        round_num: int,
        n_counterexamples: int,
        train_loss: float,
        constraint_loss: float,
        lam: float,
        csr: float,
        elapsed: float,
        extra: dict | None = None,
    ):
        entry = {
            "round": round_num,
            "n_counterexamples": n_counterexamples,
            "train_loss": round(train_loss, 6),
            "constraint_loss": round(constraint_loss, 6),
            "lambda": round(lam, 6),
            "csr": round(csr, 4),
            "elapsed_s": round(elapsed, 2),
        }
        if extra:
            entry.update(extra)
        self.rounds.append(entry)
        self.total_counterexamples += n_counterexamples

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "converged": self.converged,
                    "convergence_round": self.convergence_round,
                    "total_counterexamples": self.total_counterexamples,
                    "rounds": self.rounds,
                },
                f,
                indent=2,
            )


def _auto_device(preferred: str | None = None) -> str:
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred


class CEGISTrainer:
    """Neural CEGIS training loop.

    Generic over model and domain. Requires:
        - model: nn.Module with forward() returning dict with loss_digit, loss_constraint, csr
        - train_dataset: Dataset
        - verify_fn: Callable(model, dataloader, device) → list[dict] of counterexamples
        - ce_to_dataset_fn: Callable(list[dict], device) → Dataset of counterexample samples
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        verify_dataset: Dataset,
        verify_fn: Callable,
        ce_to_dataset_fn: Callable,
        collate_fn: Callable | None = None,
        config: CEGISConfig | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.verify_dataset = verify_dataset
        self.verify_fn = verify_fn
        self.ce_to_dataset_fn = ce_to_dataset_fn
        self.collate_fn = collate_fn
        self.config = config or CEGISConfig()
        self.device = _auto_device(self.config.device)
        self.ce_buffer: list[dict] = []
        self.log = CEGISLog()

        # Lagrangian state
        self.lag_state = LagrangianState(
            lam=0.0,
            epsilon=self.config.lagrangian_epsilon,
            alpha=self.config.lagrangian_alpha,
            rho=self.config.lagrangian_rho,
            lam_max=self.config.lagrangian_lam_max,
        )

    def _build_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            drop_last=False,
        )

    def _train_inner(self, dataset: Dataset, epochs: int) -> dict:
        """Run Lagrangian inner training loop for N epochs."""
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        loader = self._build_dataloader(dataset)

        total_loss_sum = 0.0
        constraint_loss_sum = 0.0
        csr_sum = 0.0
        n_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_constraint = 0.0
            epoch_csr = 0.0
            epoch_n = 0

            for batch in loader:
                # Move batch to device
                batch_dev = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                result = self.model(**batch_dev)

                loss_task = result.get("loss_digit", result.get("loss_total", torch.tensor(0.0)))
                loss_constraint = result.get("loss_constraint", torch.tensor(0.0))

                # Augmented Lagrangian
                loss = lagrangian_loss(loss_task, loss_constraint, self.lag_state)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_constraint += loss_constraint.item()
                epoch_csr += result.get("csr", 0.0)
                epoch_n += 1

            if epoch_n > 0:
                avg_constraint = epoch_constraint / epoch_n
                update_dual_variable(
                    self.lag_state,
                    avg_constraint,
                    step=epoch,
                    loss_task=epoch_loss / epoch_n,
                )

            total_loss_sum += epoch_loss
            constraint_loss_sum += epoch_constraint
            csr_sum += epoch_csr
            n_batches += epoch_n

        return {
            "train_loss": total_loss_sum / max(n_batches, 1),
            "constraint_loss": constraint_loss_sum / max(n_batches, 1),
            "csr": csr_sum / max(n_batches, 1),
            "lambda": self.lag_state.lam,
        }

    def _verify(self) -> list[dict]:
        """Run verifier on the verification dataset."""
        self.model.eval()
        loader = self._build_dataloader(self.verify_dataset, shuffle=False)
        counterexamples = self.verify_fn(
            self.model, loader, self.device,
            max_ce=self.config.max_counterexamples,
        )
        return counterexamples

    def _augment_dataset(self, base_dataset: Dataset) -> Dataset:
        """Create augmented dataset = base_dataset + CE_buffer."""
        if not self.ce_buffer:
            return base_dataset

        ce_dataset = self.ce_to_dataset_fn(self.ce_buffer, self.device)

        # Oversample counterexamples
        if self.config.ce_oversample > 1:
            ce_datasets = [ce_dataset] * self.config.ce_oversample
            ce_dataset = ConcatDataset(ce_datasets)

        return ConcatDataset([base_dataset, ce_dataset])

    def run(self) -> dict:
        """Execute the full CEGIS loop.

        Returns:
            Final report dict with convergence info, metrics, and log.
        """
        os.makedirs(self.config.outdir, exist_ok=True)
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        logger.info(f"Starting Neural CEGIS (max_rounds={self.config.max_rounds})")
        logger.info(f"Device: {self.device}")

        for round_num in range(1, self.config.max_rounds + 1):
            t0 = time.time()
            logger.info(f"\n{'='*50}")
            logger.info(f"CEGIS Round {round_num}/{self.config.max_rounds}")
            logger.info(f"  CE buffer size: {len(self.ce_buffer)}")
            logger.info(f"  λ = {self.lag_state.lam:.4f}")

            # 1. Augment training data with counterexamples
            augmented = self._augment_dataset(self.train_dataset)
            logger.info(f"  Training set size: {len(augmented)}")

            # 2. Inner Lagrangian training
            train_result = self._train_inner(augmented, self.config.inner_epochs)
            logger.info(
                f"  Inner training done: loss={train_result['train_loss']:.4f}, "
                f"constraint={train_result['constraint_loss']:.4f}, "
                f"CSR={train_result['csr']:.4f}, λ={train_result['lambda']:.4f}"
            )

            # 3. Verify — find counterexamples
            counterexamples = self._verify()
            n_ce = len(counterexamples)
            logger.info(f"  Counterexamples found: {n_ce}")

            elapsed = time.time() - t0

            # Log this round
            self.log.add_round(
                round_num=round_num,
                n_counterexamples=n_ce,
                train_loss=train_result["train_loss"],
                constraint_loss=train_result["constraint_loss"],
                lam=train_result["lambda"],
                csr=train_result["csr"],
                elapsed=elapsed,
                extra={"buffer_size": len(self.ce_buffer) + n_ce},
            )

            # 4. Check convergence
            if n_ce == 0:
                logger.info(f"  ✓ CONVERGED at round {round_num}!")
                self.log.converged = True
                self.log.convergence_round = round_num
                break

            # 5. Add to buffer
            self.ce_buffer.extend(counterexamples)
            logger.info(f"  CE buffer total: {len(self.ce_buffer)}")

        # Save results
        self._save_results(train_result)

        return self._build_report(train_result)

    def _save_results(self, train_result: dict):
        """Save model, logs, and Lagrangian trajectory."""
        # Save model
        ckpt_dir = os.path.join(self.config.outdir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))

        # Save CEGIS log
        self.log.save(os.path.join(self.config.outdir, "cegis_log.json"))

        # Save Lagrangian trajectory
        save_lambda_trajectory(
            self.lag_state,
            os.path.join(self.config.outdir, "lambda_trajectory.json"),
        )

        # Save config
        import yaml
        with open(os.path.join(self.config.outdir, "config.yaml"), "w") as f:
            yaml.dump(
                {k: v for k, v in vars(self.config).items()},
                f,
                default_flow_style=False,
            )

    def _build_report(self, train_result: dict) -> dict:
        """Build final report dictionary."""
        return {
            "converged": self.log.converged,
            "convergence_round": self.log.convergence_round,
            "total_rounds": len(self.log.rounds),
            "total_counterexamples": self.log.total_counterexamples,
            "final_lambda": self.lag_state.lam,
            "final_csr": train_result["csr"],
            "ce_trajectory": [r["n_counterexamples"] for r in self.log.rounds],
            "lambda_trajectory": [r["lambda"] for r in self.log.rounds],
        }


# ─────────────────────────────────────────────────────────────
# Domain-specific helpers: Multi-digit addition
# ─────────────────────────────────────────────────────────────

def multi_digit_verify_fn(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    max_ce: int = 500,
) -> list[dict]:
    """Find counterexamples for multi-digit addition."""
    from symbolic.multi_digit_constraints import verify_multi_digit

    counterexamples = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            result = model(batch_dev["img_a"], batch_dev["img_b"])

            pred_a_tens = result["probs_a_tens"].argmax(-1)
            pred_a_ones = result["probs_a_ones"].argmax(-1)
            pred_b_tens = result["probs_b_tens"].argmax(-1)
            pred_b_ones = result["probs_b_ones"].argmax(-1)
            pred_s_ones = result["probs_s_ones"].argmax(-1)
            pred_s_tens = result["probs_s_tens"].argmax(-1)
            pred_s_hund = result["probs_s_hund"].argmax(-1)

            violations, _ = verify_multi_digit(
                pred_a_tens, pred_a_ones, pred_b_tens, pred_b_ones,
                pred_s_ones, pred_s_tens, pred_s_hund,
            )

            for i in range(len(violations)):
                if violations[i]:
                    ce = {
                        "img_a": batch_dev["img_a"][i].cpu(),
                        "img_b": batch_dev["img_b"][i].cpu(),
                        "a_tens": batch_dev["a_tens"][i].cpu(),
                        "a_ones": batch_dev["a_ones"][i].cpu(),
                        "b_tens": batch_dev["b_tens"][i].cpu(),
                        "b_ones": batch_dev["b_ones"][i].cpu(),
                        "sum_hundreds": batch_dev["sum_hundreds"][i].cpu(),
                        "sum_tens": batch_dev["sum_tens"][i].cpu(),
                        "sum_ones": batch_dev["sum_ones"][i].cpu(),
                    }
                    counterexamples.append(ce)
                    if len(counterexamples) >= max_ce:
                        return counterexamples

    return counterexamples


def multi_digit_ce_to_dataset(
    counterexamples: list[dict],
    device: str,  # unused — tensors stay on CPU in dataset
) -> Dataset:
    """Convert counterexample dicts to a Dataset for DataLoader."""
    return _CEDataset(counterexamples)


class _CEDataset(Dataset):
    """Thin wrapper turning a list of CE dicts into a Dataset."""

    def __init__(self, ce_list: list[dict]):
        self.data = ce_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────────────────────
# Entry point: train multi-digit with CEGIS
# ─────────────────────────────────────────────────────────────

def train_multi_digit_cegis(config_path: str, outdir_override: str | None = None) -> dict:
    """Full training pipeline for multi-digit addition with Neural CEGIS.

    Args:
        config_path: path to YAML config.
        outdir_override: optional override for output directory.

    Returns:
        Report dict.
    """
    import yaml

    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    cegis_cfg = CEGISConfig.from_yaml(config_path)
    if outdir_override:
        cegis_cfg.outdir = outdir_override

    device = _auto_device(cegis_cfg.device)

    # Dataset
    from data.multi_digit_addition import MultiDigitAdditionDataset, multi_digit_collate

    data_cfg = raw_cfg.get("data", {})
    train_ds = MultiDigitAdditionDataset(
        split="train",
        n_samples=data_cfg.get("n_train", 5000),
        seed=cegis_cfg.seed,
        img_size=data_cfg.get("img_size", 28),
    )
    verify_ds = MultiDigitAdditionDataset(
        split="comp_test",
        n_samples=data_cfg.get("n_verify", 2000),
        seed=cegis_cfg.seed + 1,
        img_size=data_cfg.get("img_size", 28),
    )

    # Model
    from models.nst_multi_digit import MultiDigitModel

    model_mode = raw_cfg.get("model", {}).get("mode", "soft")
    model = MultiDigitModel(mode=model_mode if model_mode != "cegis" else "lagrangian")

    # CEGIS trainer
    trainer = CEGISTrainer(
        model=model,
        train_dataset=train_ds,
        verify_dataset=verify_ds,
        verify_fn=multi_digit_verify_fn,
        ce_to_dataset_fn=multi_digit_ce_to_dataset,
        collate_fn=multi_digit_collate,
        config=cegis_cfg,
    )

    report = trainer.run()

    # Final evaluation on all splits
    from data.multi_digit_addition import MultiDigitAdditionDataset

    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION")

    final_report = {"cegis": report}
    for split in ("iid_test", "comp_test", "hard_test"):
        test_ds = MultiDigitAdditionDataset(
            split=split,
            n_samples=data_cfg.get("n_test", 2000),
            seed=cegis_cfg.seed + 10,
            img_size=data_cfg.get("img_size", 28),
        )
        test_loader = DataLoader(
            test_ds, batch_size=cegis_cfg.batch_size,
            collate_fn=multi_digit_collate, shuffle=False,
        )
        metrics = _evaluate_split(model, test_loader, device)
        final_report[split] = metrics
        logger.info(f"  {split}: sum_acc={metrics['sum_acc']:.4f}, CSR={metrics['csr']:.4f}")

    # Save final report
    with open(os.path.join(cegis_cfg.outdir, "report.json"), "w") as f:
        json.dump(final_report, f, indent=2)

    return final_report


@torch.no_grad()
def _evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model on a split and return metrics."""
    model.eval()
    model.to(device)

    correct_sum = 0
    correct_digits = 0
    total = 0
    csr_total = 0.0

    for batch in loader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        result = model(batch_dev["img_a"], batch_dev["img_b"])

        pred_s = (
            result["probs_s_hund"].argmax(-1) * 100 +
            result["probs_s_tens"].argmax(-1) * 10 +
            result["probs_s_ones"].argmax(-1)
        )
        target_s = (
            batch_dev["sum_hundreds"] * 100 +
            batch_dev["sum_tens"] * 10 +
            batch_dev["sum_ones"]
        )
        correct_sum += (pred_s == target_s).sum().item()

        # Digit accuracy
        d_correct = (
            (result["probs_a_tens"].argmax(-1) == batch_dev["a_tens"]).sum() +
            (result["probs_a_ones"].argmax(-1) == batch_dev["a_ones"]).sum() +
            (result["probs_b_tens"].argmax(-1) == batch_dev["b_tens"]).sum() +
            (result["probs_b_ones"].argmax(-1) == batch_dev["b_ones"]).sum()
        )
        correct_digits += d_correct.item()

        csr_total += result["csr"] * len(batch_dev["img_a"])
        total += len(batch_dev["img_a"])

    return {
        "sum_acc": correct_sum / max(total, 1),
        "digit_acc": correct_digits / max(total * 4, 1),
        "csr": csr_total / max(total, 1),
        "n_samples": total,
    }


# ─────────────────────────────────────────────────────────────
# Domain-specific helpers: Kinship relational reasoning
# ─────────────────────────────────────────────────────────────

def kinship_verify_fn(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    max_ce: int = 500,
) -> list[dict]:
    """Find counterexamples for kinship relational reasoning.

    A counterexample is any input where the model's prediction is
    inconsistent with the chain-length constraints (e.g., predicting
    "parent" for a depth-3 chain when only ancestor/descendant/sibling
    are valid).
    """
    from data.kinship import check_kinship_constraint

    counterexamples = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            chain_lengths = batch["chain_lengths"]

            result = model(input_ids, labels=labels, chain_lengths=chain_lengths)
            preds = result["probs"].argmax(dim=-1)

            # Check constraint violations per sample
            for i in range(len(labels)):
                cl = chain_lengths[i]
                pred = preds[i].item()
                label = labels[i].item()

                # Constraint violation: prediction in wrong structural category
                violated = False
                if cl == 1 and pred not in (0, 1):        # must be parent/child
                    violated = True
                elif cl == 2 and pred not in (2, 3, 4):    # grandparent/grandchild/sibling
                    violated = True
                elif cl >= 3 and pred not in (4, 5, 6):    # sibling/ancestor/descendant
                    violated = True

                # Also treat wrong predictions as counterexamples
                if pred != label:
                    violated = True

                if violated:
                    ce = {
                        "input_ids": batch["input_ids"][i].cpu(),
                        "label": batch["label"][i].cpu(),
                        "chain_length": chain_lengths[i],
                        "text": batch["texts"][i] if "texts" in batch else "",
                        "answer": batch["answers"][i] if "answers" in batch else "",
                    }
                    counterexamples.append(ce)
                    if len(counterexamples) >= max_ce:
                        return counterexamples

    return counterexamples


def kinship_ce_to_dataset(
    counterexamples: list[dict],
    device: str,  # unused — tensors stay on CPU
) -> Dataset:
    """Convert kinship counterexample dicts to a Dataset."""
    return _KinshipCEDataset(counterexamples)


class _KinshipCEDataset(Dataset):
    """Thin wrapper for kinship counterexample dicts as a Dataset."""

    def __init__(self, ce_list: list[dict]):
        self.data = ce_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ce = self.data[idx]
        return {
            "input_ids": ce["input_ids"],
            "label": ce["label"],
            "chain_length": ce["chain_length"],
            "text": ce.get("text", ""),
            "answer": ce.get("answer", ""),
        }


# ─────────────────────────────────────────────────────────────
# Entry point: train kinship with CEGIS
# ─────────────────────────────────────────────────────────────

def train_kinship_cegis(config_path: str, outdir_override: str | None = None) -> dict:
    """Full training pipeline for kinship with Neural CEGIS.

    Args:
        config_path: path to YAML config.
        outdir_override: optional override for output directory.

    Returns:
        Report dict.
    """
    import yaml

    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    cegis_cfg = CEGISConfig.from_yaml(config_path)
    if outdir_override:
        cegis_cfg.outdir = outdir_override

    device = _auto_device(cegis_cfg.device)

    # Dataset
    from data.kinship import KinshipDataset, kinship_collate_fn

    data_cfg = raw_cfg.get("data", {})
    train_ds = KinshipDataset(
        split="train",
        n_samples=int(data_cfg.get("n_train", 5000)),
        max_train_depth=int(data_cfg.get("max_train_depth", 3)),
        max_test_depth=int(data_cfg.get("max_test_depth", 6)),
        max_seq_len=int(data_cfg.get("max_seq_len", 384)),
        direction_mix=data_cfg.get("direction_mix", True),
        seed=cegis_cfg.seed,
        balanced_sampling=data_cfg.get("balanced_sampling", True),
        n_distractors=int(data_cfg.get("n_distractors", 2)),
        corruption_rate=float(data_cfg.get("corruption_rate", 0.0)),
    )
    verify_ds = KinshipDataset(
        split="comp_test",
        n_samples=int(data_cfg.get("n_verify", 2000)),
        max_train_depth=int(data_cfg.get("max_train_depth", 3)),
        max_test_depth=int(data_cfg.get("max_test_depth", 6)),
        max_seq_len=int(data_cfg.get("max_seq_len", 384)),
        direction_mix=data_cfg.get("direction_mix", True),
        seed=cegis_cfg.seed + 1,
        balanced_sampling=data_cfg.get("balanced_sampling", True),
        n_distractors=int(data_cfg.get("n_distractors", 2)),
    )

    # Model
    from models.nst_kinship import KinshipTransformer

    model_cfg = raw_cfg.get("model", {})
    model = KinshipTransformer(
        d_model=int(model_cfg.get("d_model", 128)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        d_ff=int(model_cfg.get("d_ff", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_seq_len=int(data_cfg.get("max_seq_len", 384)),
    )

    # CEGIS trainer
    trainer = CEGISTrainer(
        model=model,
        train_dataset=train_ds,
        verify_dataset=verify_ds,
        verify_fn=kinship_verify_fn,
        ce_to_dataset_fn=kinship_ce_to_dataset,
        collate_fn=kinship_collate_fn,
        config=cegis_cfg,
    )

    report = trainer.run()

    # Final evaluation on all splits
    from data.kinship import KinshipDataset as KDS

    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION (Kinship CEGIS)")

    final_report = {"cegis": report}
    for split in ("iid_test", "comp_test"):
        test_ds = KDS(
            split=split,
            n_samples=int(data_cfg.get("n_test", 2000)),
            max_train_depth=int(data_cfg.get("max_train_depth", 3)),
            max_test_depth=int(data_cfg.get("max_test_depth", 6)),
            max_seq_len=int(data_cfg.get("max_seq_len", 384)),
            seed=cegis_cfg.seed + 10,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cegis_cfg.batch_size,
            collate_fn=kinship_collate_fn, shuffle=False,
        )
        metrics = _evaluate_kinship_split(model, test_loader, device)
        final_report[split] = metrics
        logger.info(f"  {split}: acc={metrics['accuracy']:.4f}, CSR={metrics['csr']:.4f}")

    # Save final report
    with open(os.path.join(cegis_cfg.outdir, "report.json"), "w") as f:
        json.dump(final_report, f, indent=2)

    return final_report


@torch.no_grad()
def _evaluate_kinship_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> dict:
    """Evaluate kinship model on a split."""
    model.eval()
    model.to(device)

    correct = total = 0
    csr_total = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        chain_lengths = batch["chain_lengths"]

        result = model(input_ids, labels=labels, chain_lengths=chain_lengths)
        preds = result["probs"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        csr_total += result.get("csr", 0.0)
        n_batches += 1

    return {
        "accuracy": correct / max(1, total),
        "csr": csr_total / max(1, n_batches),
        "n_samples": total,
    }

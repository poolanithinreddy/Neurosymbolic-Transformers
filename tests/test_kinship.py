"""Tests for the kinship relational reasoning module."""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from data.kinship import (
    IDX_TO_RELATION,
    KinshipDataset,
    NUM_RELATIONS,
    RELATION_TO_IDX,
    VOCAB_SIZE,
    check_kinship_constraint,
    generate_sample,
    infer_relation,
    kinship_collate_fn,
    kinship_constraint_loss,
    tokenise,
)
from models.nst_kinship import KinshipTransformer


class TestInferRelation:
    def test_single_parent(self):
        assert infer_relation(["parent"]) == "parent"

    def test_single_child(self):
        assert infer_relation(["child"]) == "child"

    def test_two_parents_is_grandparent(self):
        assert infer_relation(["parent", "parent"]) == "grandparent"

    def test_two_children_is_grandchild(self):
        assert infer_relation(["child", "child"]) == "grandchild"

    def test_parent_child_is_sibling(self):
        assert infer_relation(["parent", "child"]) == "sibling"

    def test_three_parents_is_ancestor(self):
        assert infer_relation(["parent", "parent", "parent"]) == "ancestor"

    def test_empty_is_self(self):
        assert infer_relation([]) == "self"


class TestTokenise:
    def test_output_shape(self):
        tokens = tokenise("Hello world", max_len=64)
        assert tokens.shape == (64,)

    def test_padding(self):
        tokens = tokenise("Hi", max_len=32)
        # Should have padding (zeros) at the end
        assert tokens[-1].item() == 0

    def test_truncation(self):
        long_text = "a" * 500
        tokens = tokenise(long_text, max_len=100)
        assert tokens.shape == (100,)


class TestKinshipDataset:
    def test_train_split(self):
        ds = KinshipDataset("train", n_samples=100, max_train_depth=3, seed=42, balanced_sampling=False, direction_mix=False)
        assert len(ds) == 100

    def test_comp_split_depths(self):
        ds = KinshipDataset("comp_test", n_samples=100, max_train_depth=3, max_test_depth=5, seed=42, balanced_sampling=False, direction_mix=False)
        assert len(ds) == 100
        # All samples should have chain length > 3
        for s in ds.samples:
            assert s.chain_length > 3

    def test_iid_split_depths(self):
        ds = KinshipDataset("iid_test", n_samples=100, max_train_depth=3, seed=42)
        for s in ds.samples:
            assert s.chain_length <= 3

    def test_getitem_returns_dict(self):
        ds = KinshipDataset("train", n_samples=10, seed=42)
        item = ds[0]
        assert "input_ids" in item
        assert "label" in item
        assert "text" in item

    def test_collate(self):
        ds = KinshipDataset("train", n_samples=8, seed=42)
        batch = kinship_collate_fn([ds[i] for i in range(4)])
        assert batch["input_ids"].shape[0] == 4
        assert batch["label"].shape[0] == 4
        assert len(batch["chain_lengths"]) == 4


class TestKinshipConstraint:
    def test_correct_predictions_high_csr(self):
        # Depth 1 predictions should be parent/child (idx 0 or 1)
        probs = torch.zeros(4, NUM_RELATIONS)
        probs[0, 0] = 1.0  # parent — correct for depth 1
        probs[1, 1] = 1.0  # child — correct for depth 1
        probs[2, 2] = 1.0  # grandparent — correct for depth 2
        probs[3, 5] = 1.0  # ancestor — correct for depth 3

        chain_lengths = [1, 1, 2, 3]
        csr, viol = check_kinship_constraint(probs, chain_lengths)
        assert csr == 1.0

    def test_wrong_predictions_low_csr(self):
        probs = torch.zeros(2, NUM_RELATIONS)
        probs[0, 5] = 1.0  # ancestor — wrong for depth 1
        probs[1, 0] = 1.0  # parent — wrong for depth 3

        chain_lengths = [1, 3]
        csr, viol = check_kinship_constraint(probs, chain_lengths)
        assert csr < 1.0

    def test_constraint_loss_differentiable(self):
        probs = torch.randn(4, NUM_RELATIONS, requires_grad=True).softmax(dim=-1)
        chain_lengths = [1, 2, 3, 1]
        loss = kinship_constraint_loss(probs, chain_lengths)
        loss.backward()
        assert torch.isfinite(loss)


class TestKinshipModel:
    def test_forward_shape(self):
        model = KinshipTransformer(d_model=32, n_heads=2, n_layers=1, d_ff=64)
        input_ids = torch.randint(0, VOCAB_SIZE, (4, 64))
        labels = torch.randint(0, NUM_RELATIONS, (4,))
        chain_lengths = [1, 2, 1, 3]

        result = model(input_ids, labels=labels, chain_lengths=chain_lengths)
        assert result["logits"].shape == (4, NUM_RELATIONS)
        assert result["probs"].shape == (4, NUM_RELATIONS)
        assert "loss_task" in result
        assert "loss_constraint" in result

    def test_predict(self):
        model = KinshipTransformer(d_model=32, n_heads=2, n_layers=1, d_ff=64)
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 64))
        preds = model.predict(input_ids, chain_lengths=[1, 2])
        assert preds["pred_idx"].shape == (2,)
        assert len(preds["pred_names"]) == 2

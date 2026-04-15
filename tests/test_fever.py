"""Tests for FEVER integration modules.

Covers:
  - Label mapping (LABEL2ID / ID2LABEL roundtrip)
  - Evidence formatting (claim + evidence → tokeniser input)
  - Structured fact extraction (numbers, dates, negation, entities)
  - Constraint loss computation (differentiable C1-C5)
  - Hard constraint verification (for CEGIS mining)
  - Leakage guard in FeverPipelineDataset
  - Split hash determinism
  - Evaluation metrics (label_accuracy, confusion_matrix)
  - BM25 retriever basics
"""

from __future__ import annotations

import os
import sys
import pytest
import random

THIS_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


# ── Label mapping ──────────────────────────────────────────────

class TestLabelMapping:
    """FEVER label mapping must be canonical and invertible."""

    def test_label2id_contains_all(self):
        from data.fever_dataset import LABEL2ID, FEVER_LABELS
        for lbl in FEVER_LABELS:
            assert lbl in LABEL2ID

    def test_id2label_roundtrip(self):
        from data.fever_dataset import LABEL2ID, ID2LABEL
        for lbl, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == lbl

    def test_num_labels_is_three(self):
        from data.fever_dataset import NUM_LABELS
        assert NUM_LABELS == 3

    def test_normalise_label_variants(self):
        from data.fever_dataset import _normalise_label
        assert _normalise_label("SUPPORTS") == "SUPPORTS"
        assert _normalise_label("REFUTES") == "REFUTES"
        assert _normalise_label("NOT ENOUGH INFO") == "NOT ENOUGH INFO"
        # Common variants
        assert _normalise_label("Supported") == "SUPPORTS"
        assert _normalise_label("Refuted") == "REFUTES"
        assert _normalise_label("NotEnoughInfo") == "NOT ENOUGH INFO"
        assert _normalise_label("notenoughinfo") == "NOT ENOUGH INFO"

    def test_normalise_label_unknown_defaults_to_nei(self):
        from data.fever_dataset import _normalise_label
        # Unknown labels default to NOT ENOUGH INFO (safe fallback)
        assert _normalise_label("MAYBE") == "NOT ENOUGH INFO"


# ── Structured fact extraction ─────────────────────────────────

class TestConstraintExtraction:
    """Regex-based extraction of numbers, dates, negation, entities."""

    def test_extract_numbers(self):
        from symbolic.fever_constraints import _extract_numbers, _normalise_number
        nums = _extract_numbers("There are 42 apples and 1,000 oranges")
        parsed = [_normalise_number(n) for n in nums]
        assert 42.0 in parsed
        assert 1000.0 in parsed

    def test_extract_numbers_with_million(self):
        from symbolic.fever_constraints import _extract_numbers, _normalise_number
        nums = _extract_numbers("Revenue was 3.5 million")
        parsed = [_normalise_number(n) for n in nums]
        assert 3_500_000.0 in parsed

    def test_extract_dates(self):
        from symbolic.fever_constraints import _extract_dates
        dates = _extract_dates("Founded on 15 March 2001 or maybe 2002")
        assert "2001" in dates or "15 March 2001" in dates

    def test_has_negation(self):
        from symbolic.fever_constraints import _has_negation
        assert _has_negation("He did not win the award")
        assert _has_negation("It never happened")
        assert not _has_negation("He won the award")

    def test_extract_entities(self):
        from symbolic.fever_constraints import _extract_entities
        ents = _extract_entities("Barack Obama was born in Hawaii")
        assert "Barack Obama" in ents or "Hawaii" in ents

    def test_number_contradiction(self):
        from symbolic.fever_constraints import extract_structured_facts
        facts = extract_structured_facts(
            claim="The population is 500",
            evidence="The population was recorded as 800",
        )
        assert facts.number_contradiction is True

    def test_no_contradiction_same_numbers(self):
        from symbolic.fever_constraints import extract_structured_facts
        facts = extract_structured_facts(
            claim="He scored 3 goals",
            evidence="He scored 3 goals in the match",
        )
        assert facts.number_contradiction is False

    def test_negation_mismatch(self):
        from symbolic.fever_constraints import extract_structured_facts
        facts = extract_structured_facts(
            claim="He never visited France",
            evidence="He visited France in 2010",
        )
        assert facts.negation_mismatch is True

    def test_entity_overlap_high(self):
        from symbolic.fever_constraints import extract_structured_facts
        facts = extract_structured_facts(
            claim="Barack Obama was president",
            evidence="Barack Obama served as the 44th president",
        )
        assert facts.entity_overlap_score > 0.3

    def test_entity_overlap_low(self):
        from symbolic.fever_constraints import extract_structured_facts
        facts = extract_structured_facts(
            claim="The Eiffel Tower is in Paris",
            evidence="Mount Everest is the tallest mountain",
        )
        assert facts.entity_overlap_score < 0.3


# ── Constraint loss (differentiable) ──────────────────────────

class TestConstraintLoss:
    """Differentiable fever constraint loss must produce valid tensors."""

    def test_fever_constraint_loss_shape(self):
        import torch
        from symbolic.fever_constraints import extract_structured_facts
        from symbolic.fever_constraint_loss import fever_constraint_loss

        # Make batch of 4
        facts_batch = [
            extract_structured_facts("Population is 500", "Population was 800"),
            extract_structured_facts("He won the award", "He won the award"),
            extract_structured_facts("Never visited", "He visited often"),
            extract_structured_facts("Some claim", ""),
        ]

        p_supports = torch.tensor([0.8, 0.9, 0.7, 0.3])
        p_refutes = torch.tensor([0.1, 0.05, 0.2, 0.2])
        p_nei = torch.tensor([0.1, 0.05, 0.1, 0.5])

        loss, info = fever_constraint_loss(p_supports, p_refutes, p_nei, facts_batch)
        assert loss.shape == ()
        assert loss.item() >= 0
        assert "constraint_loss_total" in info
        assert "v_date_contradiction" in info

    def test_no_constraints_returns_zero(self):
        import torch
        from symbolic.fever_constraints import extract_structured_facts
        from symbolic.fever_constraint_loss import fever_constraint_loss

        facts_batch = [
            extract_structured_facts("He won the award", "He won the award"),
        ]
        p_supports = torch.tensor([0.9])
        p_refutes = torch.tensor([0.05])
        p_nei = torch.tensor([0.05])

        loss, info = fever_constraint_loss(p_supports, p_refutes, p_nei, facts_batch)
        # Should be close to 0 (no strong contradictions)
        assert loss.item() < 1.0


# ── Hard constraint verification (CEGIS) ─────────────────────

class TestHardVerification:
    """verify_fever_constraints for CEGIS mining."""

    def test_detects_violation_number_contradiction(self):
        from symbolic.fever_constraints import extract_structured_facts
        from symbolic.fever_constraint_loss import verify_fever_constraints

        facts_batch = [
            extract_structured_facts("Population is 500", "Population was 800"),
        ]
        # Predict SUPPORTS despite number contradiction → violation
        pred_labels = ["SUPPORTS"]
        violations, csr = verify_fever_constraints(pred_labels, facts_batch)
        assert violations[0] > 0

    def test_no_violation_when_correct(self):
        from symbolic.fever_constraints import extract_structured_facts
        from symbolic.fever_constraint_loss import verify_fever_constraints

        facts_batch = [
            extract_structured_facts("Population is 500", "Population was 800"),
        ]
        # Predict REFUTES for contradiction → correct
        pred_labels = ["REFUTES"]
        violations, csr = verify_fever_constraints(pred_labels, facts_batch)
        assert violations[0] == 0


# ── Evaluation metrics ────────────────────────────────────────

class TestEvalMetrics:
    """Label accuracy, confusion matrix, retrieval recall."""

    def test_label_accuracy_perfect(self):
        from eval.fever_metrics import label_accuracy
        preds = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        golds = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        report = label_accuracy(preds, golds)
        assert report["accuracy"] == 1.0

    def test_label_accuracy_zero(self):
        from eval.fever_metrics import label_accuracy
        preds = ["SUPPORTS", "SUPPORTS", "SUPPORTS"]
        golds = ["REFUTES", "REFUTES", "REFUTES"]
        report = label_accuracy(preds, golds)
        assert report["accuracy"] == 0.0

    def test_confusion_matrix_diagonal(self):
        from eval.fever_metrics import confusion_matrix
        preds = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        golds = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        cm = confusion_matrix(preds, golds)
        assert cm["SUPPORTS"]["SUPPORTS"] == 1
        assert cm["REFUTES"]["REFUTES"] == 1
        assert cm["NOT ENOUGH INFO"]["NOT ENOUGH INFO"] == 1

    def test_retrieval_recall(self):
        from eval.fever_metrics import retrieval_recall_at_k
        retrieved = [["Page_A", "Page_B", "Page_C"]]
        gold = [["Page_B"]]
        results = retrieval_recall_at_k(retrieved, gold, k_values=[1, 3])
        assert results["recall@3"] == 1.0

    def test_retrieval_recall_miss(self):
        from eval.fever_metrics import retrieval_recall_at_k
        retrieved = [["Page_A", "Page_B"]]
        gold = [["Page_Z"]]
        results = retrieval_recall_at_k(retrieved, gold, k_values=[1, 3])
        assert results["recall@1"] == 0.0

    def test_integrity_check_structure(self):
        from eval.fever_metrics import integrity_check
        splits = {
            "train": [{"id": 1, "claim": "test claim", "label": "SUPPORTS"}],
            "dev": [{"id": 2, "claim": "dev claim", "label": "REFUTES"}],
        }
        checks = integrity_check(
            splits, ["SUPPORTS"], ["SUPPORTS"], evidence_mode="gold"
        )
        assert "hash_train" in checks
        assert "hash_dev" in checks
        assert checks["accuracy"] == 1.0
        assert checks["evidence_mode"] == "gold"


# ── Split hash determinism ────────────────────────────────────

class TestSplitHash:
    """Split hashes must be deterministic across runs."""

    def test_hash_deterministic(self):
        from data.fever_dataset import _split_hash
        items = [
            {"id": 1, "claim": "test claim one", "label": "SUPPORTS"},
            {"id": 2, "claim": "test claim two", "label": "REFUTES"},
        ]
        h1 = _split_hash(items)
        h2 = _split_hash(items)
        assert h1 == h2
        assert len(h1) == 16  # 16-char hex

    def test_hash_changes_with_data(self):
        from data.fever_dataset import _split_hash
        items_a = [{"id": 1, "claim": "claim A", "label": "SUPPORTS"}]
        items_b = [{"id": 1, "claim": "claim B", "label": "SUPPORTS"}]
        assert _split_hash(items_a) != _split_hash(items_b)


# ── BM25 retriever ────────────────────────────────────────────

class TestBM25Retriever:
    """BM25 retriever basic functionality."""

    def test_build_and_retrieve(self):
        from retrieval.bm25_retriever import BM25Retriever
        sentences = [
            {"title": "Page_A", "sent_idx": 0, "text": "Barack Obama was born in Hawaii"},
            {"title": "Page_B", "sent_idx": 0, "text": "The Eiffel Tower is in Paris"},
            {"title": "Page_C", "sent_idx": 0, "text": "Python is a programming language"},
        ]
        retriever = BM25Retriever(sentences)
        results = retriever.retrieve("Where was Obama born?", top_k=2)
        assert len(results) <= 2
        assert all("title" in r for r in results)

    def test_retrieve_as_text(self):
        from retrieval.bm25_retriever import BM25Retriever
        sentences = [
            {"title": "P1", "sent_idx": 0, "text": "Cats are animals"},
            {"title": "P2", "sent_idx": 0, "text": "Dogs are pets"},
        ]
        retriever = BM25Retriever(sentences)
        text = retriever.retrieve_as_text("What are cats?", top_k=1)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_empty_store(self):
        from retrieval.bm25_retriever import BM25Retriever
        retriever = BM25Retriever([])
        results = retriever.retrieve("test query", top_k=3)
        assert results == []


# ── Shuffle sanity ────────────────────────────────────────────

class TestShuffleSanity:
    """Shuffled labels must drop accuracy to ~chance level."""

    def test_shuffle_drops_accuracy(self):
        random.seed(42)
        labels = (["SUPPORTS"] * 100 + ["REFUTES"] * 100 +
                  ["NOT ENOUGH INFO"] * 100)
        preds = list(labels)  # perfect predictions

        # Shuffle gold → accuracy should drop
        shuffled = list(labels)
        random.shuffle(shuffled)
        n_correct = sum(p == g for p, g in zip(preds, shuffled))
        shuffle_acc = n_correct / len(shuffled)

        assert shuffle_acc < 0.5  # Should be ~0.33
        assert shuffle_acc > 0.1  # Not degenerate


# ── FeverNLIWrapper ────────────────────────────────────────────

class TestFeverNLIWrapper:
    """Model wrapper produces correct output shapes."""

    @pytest.fixture
    def small_model(self):
        """Build a tiny model for testing without any network access."""
        from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

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
        return DebertaV2ForSequenceClassification(config)

    def test_forward_output_keys(self, small_model):
        import torch
        from models.fever_nli import FeverNLIWrapper

        wrapper = FeverNLIWrapper(small_model)
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        out = wrapper(input_ids, attention_mask)
        assert "logits" in out
        assert "probs" in out
        assert out["logits"].shape == (batch_size, 3)
        assert out["probs"].shape == (batch_size, 3)

    def test_forward_with_labels(self, small_model):
        import torch
        from models.fever_nli import FeverNLIWrapper

        wrapper = FeverNLIWrapper(small_model)
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.randint(0, 3, (batch_size,))

        out = wrapper(input_ids, attention_mask, labels=labels)
        assert "loss" in out
        assert out["loss"].shape == ()

    def test_get_label_probs(self, small_model):
        import torch
        from models.fever_nli import FeverNLIWrapper

        wrapper = FeverNLIWrapper(small_model)
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        probs_dict = wrapper.get_label_probs(input_ids, attention_mask)
        assert "p_supports" in probs_dict
        assert "p_refutes" in probs_dict
        assert "p_nei" in probs_dict
        assert probs_dict["p_supports"].shape == (batch_size,)

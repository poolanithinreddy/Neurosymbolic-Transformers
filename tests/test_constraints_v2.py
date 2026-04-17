"""Tests for ConstraintEngineV2 (7 high-precision probabilistic constraints)."""

import pytest


class TestConstraintEngineV2:
    """Verify the constraint engine produces correct signals."""

    def _engine(self):
        from symbolic.constraints_v2 import ConstraintEngineV2
        return ConstraintEngineV2()

    def test_has_seven_constraints(self):
        engine = self._engine()
        assert engine.n_constraints == 7
        assert len(engine.constraints) == 7

    def test_constraint_names(self):
        engine = self._engine()
        expected = [
            "NumericalConstraint",
            "NegationConstraint",
            "EntityOverlapConstraint",
            "EvidenceSufficiencyConstraint",
            "TemporalConstraint",
            "HedgeModalityConstraint",
            "MutualExclusionConstraint",
        ]
        assert engine.constraint_names == expected

    def test_evaluate_single_returns_correct_shape(self):
        engine = self._engine()
        results = engine.evaluate_single(
            "The population is 500",
            "The population was recorded as 800",
        )
        assert len(results) == 7
        for r in results:
            assert isinstance(r.fires, bool)
            assert 0.0 <= r.confidence <= 1.0
            assert r.direction.shape == (3,)
            assert abs(r.direction.sum().item() - 1.0) < 1e-5

    def test_evaluate_batch_shapes(self):
        engine = self._engine()
        claims = ["Paris has 3 million people", "He is tall"]
        evidences = ["Paris has a population of 2.1 million", "She is short"]
        result = engine.evaluate_batch(claims, evidences)
        assert result["fires"].shape == (2, 7)
        assert result["confidence"].shape == (2, 7)
        assert result["direction"].shape == (2, 7, 3)

    def test_numerical_fires_on_shared_entity_number_conflict(self):
        engine = self._engine()
        results = engine.evaluate_single(
            "The population of France is 500",
            "The population of France was recorded as 67",
        )
        num_result = results[0]  # NumericalConstraint is first
        assert num_result.fires is True

    def test_numerical_does_not_fire_on_unrelated_numbers(self):
        engine = self._engine()
        results = engine.evaluate_single(
            "He scored 3 goals in 2010",
            "The stadium holds 50000 fans since 1998",
        )
        num_result = results[0]
        assert num_result.fires is False

    def test_negation_fires_on_shared_content(self):
        engine = self._engine()
        results = engine.evaluate_single(
            "He never won the award",
            "He won the award in 2015",
        )
        neg_result = results[1]  # NegationConstraint is second
        assert neg_result.fires is True

    def test_negation_does_not_fire_on_unrelated_sentences(self):
        engine = self._engine()
        results = engine.evaluate_single(
            "She never visited Paris",
            "The weather was not sunny today",
        )
        neg_result = results[1]
        assert neg_result.fires is False

    def test_entity_overlap_fires_on_low_overlap(self):
        engine = self._engine()
        results = engine.evaluate_single(
            "Barack Obama was born in Hawaii",
            "The weather in London was rainy",
        )
        ent_result = results[2]  # EntityOverlapConstraint is third
        assert ent_result.fires is True

    def test_sufficiency_fires_on_empty_evidence(self):
        engine = self._engine()
        results = engine.evaluate_single(
            "The sky is blue",
            "",
        )
        suf_result = results[3]  # EvidenceSufficiencyConstraint is fourth
        assert suf_result.fires is True

    def test_mutual_exclusion_fires(self):
        engine = self._engine()
        results = engine.evaluate_single(
            "Paris is the capital of Germany",
            "Paris is the capital of France",
        )
        me_result = results[6]  # MutualExclusionConstraint is seventh
        assert me_result.fires is True

    def test_mutual_exclusion_does_not_fire_on_agreement(self):
        engine = self._engine()
        results = engine.evaluate_single(
            "Paris is the capital of France",
            "Paris is the capital of France",
        )
        me_result = results[6]
        assert me_result.fires is False

    def test_no_constraints_fire_on_matching_pair(self):
        engine = self._engine()
        results = engine.evaluate_single(
            "Water boils at 100 degrees Celsius",
            "Water boils at a temperature of 100 degrees Celsius at sea level",
        )
        fires = [r.fires for r in results]
        # At most sufficiency or entity overlap might fire, but the key
        # semantic constraints (numerical, negation, temporal, mutual exclusion)
        # should NOT fire
        assert results[0].fires is False  # numerical
        assert results[1].fires is False  # negation
        assert results[6].fires is False  # mutual exclusion

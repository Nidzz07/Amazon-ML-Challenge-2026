"""Tests for metric.py — the number the whole team trusts (Track D, owner: Krrish).

The worked-example test reproduces the exact number printed in the problem
statement.  If this test fails, every evaluation report is garbage.
"""
import numpy as np
import pytest

from metric import f05_per_entity, macro_f05, macro_f05_arrays


# ── Organisers' worked example ──────────────────────────────────────

class TestWorkedExample:
    """prediction [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812]
    → c=2, k=2, m=3 → 1.25·2/(0.5+3) = 2.5/3.5 = 5/7 ≈ 0.714"""

    def test_exact_value(self):
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        truth = {"S2-00047", "S3-00812"}
        result = f05_per_entity(pred, truth)
        assert abs(result - 5 / 7) < 1e-10

    def test_rounded_value(self):
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        truth = {"S2-00047", "S3-00812"}
        assert round(f05_per_entity(pred, truth), 3) == 0.714


# ── Singleton handling ──────────────────────────────────────────────

class TestSingletons:
    """k=0 entities: 1.0 for empty, 0.0 for any prediction."""

    def test_correct_empty(self):
        assert f05_per_entity(set(), set()) == 1.0

    def test_wrong_prediction(self):
        assert f05_per_entity({"S2-001"}, set()) == 0.0

    def test_two_wrong_predictions(self):
        assert f05_per_entity({"S2-001", "S3-002"}, set()) == 0.0


# ── Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_perfect_match(self):
        pred = {"S2-001", "S3-002"}
        truth = {"S2-001", "S3-002"}
        assert f05_per_entity(pred, truth) == 1.0

    def test_miss_all(self):
        """k>0 but predict nothing → 0.0"""
        assert f05_per_entity(set(), {"S2-001"}) == 0.0

    def test_all_wrong(self):
        """Predict items that aren't in truth → c=0 → 0.0"""
        assert f05_per_entity({"S2-999"}, {"S2-001"}) == 0.0

    def test_single_correct(self):
        """k=1, m=1, c=1 → 1.25/(0.25+1) = 1.0"""
        assert f05_per_entity({"S2-001"}, {"S2-001"}) == 1.0

    def test_over_predict(self):
        """k=1, m=3, c=1 → 1.25/(0.25+3) = 1.25/3.25"""
        result = f05_per_entity({"S2-001", "S2-002", "S2-003"}, {"S2-001"})
        assert abs(result - 1.25 / 3.25) < 1e-10


# ── Macro average ───────────────────────────────────────────────────

class TestMacroAverage:
    def test_includes_singletons(self):
        """Singletons are included in the average, not skipped."""
        preds = {"E1": set(), "E2": {"S2-00047", "S2-00193", "S3-00812"}}
        truths = {"E1": set(), "E2": {"S2-00047", "S3-00812"}}
        result = macro_f05(preds, truths)
        expected = (1.0 + 5 / 7) / 2
        assert abs(result - expected) < 1e-10

    def test_raises_on_missing_entity(self):
        """Missing entity in predictions must raise, not silently score zero."""
        with pytest.raises(KeyError):
            macro_f05(
                predictions={},
                ground_truth={"E1": {"S2-001"}},
            )

    def test_all_singletons(self):
        truths = {"E1": set(), "E2": set(), "E3": set()}
        preds = {"E1": set(), "E2": set(), "E3": set()}
        assert macro_f05(preds, truths) == 1.0

    def test_accepts_lists(self):
        """Ground truth and predictions may be lists, not just sets."""
        preds = {"E1": ["S2-00047", "S2-00193", "S3-00812"]}
        truths = {"E1": ["S2-00047", "S3-00812"]}
        assert abs(macro_f05(preds, truths) - 5 / 7) < 1e-10


# ── Vectorised version ─────────────────────────────────────────────

class TestArrayVersion:
    def test_matches_dict_version(self):
        c = np.array([2, 0, 0, 3])
        k = np.array([2, 0, 1, 3])
        m = np.array([3, 0, 0, 3])
        result = macro_f05_arrays(c, k, m)
        # Entity 0: 1.25*2/(0.5+3) = 5/7
        # Entity 1: k=0,m=0 → 1.0
        # Entity 2: k=1,m=0 → 0.0
        # Entity 3: 1.25*3/(0.75+3) = 3.75/3.75 = 1.0
        expected = (5 / 7 + 1.0 + 0.0 + 1.0) / 4
        assert abs(result - expected) < 1e-10

    def test_worked_example_via_arrays(self):
        result = macro_f05_arrays(c=np.array([2]), k=np.array([2]), m=np.array([3]))
        assert abs(result - 5 / 7) < 1e-10

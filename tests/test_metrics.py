"""Unit tests for statistical primitives in football_llm.eval.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from football_llm.eval import metrics

# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_matches_paper_halftime_headline(self):
        """Paper §5.1: halftime 64.1% result acc on n=128 → Wilson [0.555, 0.719]."""
        ci = metrics.wilson_ci(successes=82, n=128)
        assert ci.point == pytest.approx(0.641, abs=1e-3)
        assert ci.low == pytest.approx(0.555, abs=1e-3)
        assert ci.high == pytest.approx(0.719, abs=1e-3)

    def test_matches_paper_named_halftime_events_ou(self):
        """Paper §5.5: halftime+events O/U 2.5 on n=64 named → [0.736, 0.913]."""
        ci = metrics.wilson_ci(successes=54, n=64)
        assert ci.point == pytest.approx(0.844, abs=1e-3)
        assert ci.low == pytest.approx(0.736, abs=1e-3)
        assert ci.high == pytest.approx(0.913, abs=1e-3)

    def test_zero_successes_has_zero_lower_bound(self):
        ci = metrics.wilson_ci(0, 10)
        assert ci.point == 0.0
        assert ci.low == 0.0
        assert ci.high > 0  # upper bound can still be positive

    def test_all_successes_has_one_upper_bound(self):
        ci = metrics.wilson_ci(10, 10)
        assert ci.point == 1.0
        assert ci.high == pytest.approx(1.0, abs=1e-9)
        assert ci.low < 1.0

    def test_zero_sample_returns_nans(self):
        ci = metrics.wilson_ci(0, 0)
        assert math.isnan(ci.point)
        assert ci.n == 0

    def test_interval_width_shrinks_with_n(self):
        """Higher n → tighter interval, holding p fixed at 0.5."""
        small = metrics.wilson_ci(5, 10)
        large = metrics.wilson_ci(500, 1000)
        assert (large.high - large.low) < (small.high - small.low)


# ---------------------------------------------------------------------------
# Paired McNemar
# ---------------------------------------------------------------------------


class TestMcNemarExact:
    def test_equal_errors_p_equals_one(self):
        """If b == c, null of symmetry is exactly supported."""
        a = [True, True, False, False]
        b = [True, False, True, False]  # b=1, c=1
        r = metrics.mcnemar_exact(a, b)
        assert r.b == 1 and r.c == 1
        assert r.p_value == pytest.approx(1.0, abs=1e-6)

    def test_perfectly_asymmetric(self):
        """b=5, c=0 → p = 2 * 0.5^5 = 0.0625 (two-sided exact)."""
        a = [True] * 5 + [False] * 5  # A correct on first 5
        b = [False] * 5 + [False] * 5  # B never correct → b=5, c=0
        r = metrics.mcnemar_exact(a, b)
        assert r.b == 5 and r.c == 0
        # Two-sided p-value: 2 * P(X ≤ 0 | n=5, p=0.5) = 2 * (0.5)^5 = 0.0625
        assert r.p_value == pytest.approx(0.0625, abs=1e-4)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            metrics.mcnemar_exact([True, False], [True, False, True])

    def test_all_agree_p_is_one(self):
        """If every sample is classified the same by both models, p = 1."""
        a = [True] * 5 + [False] * 3
        b = [True] * 5 + [False] * 3
        r = metrics.mcnemar_exact(a, b)
        assert r.b == 0 and r.c == 0
        assert r.p_value == 1.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_perfect_calibration_has_zero_ece(self):
        """When predicted probability matches empirical frequency exactly, ECE = 0."""
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, size=10_000)
        outcomes = rng.uniform(0, 1, size=10_000) < probs
        ece = metrics.expected_calibration_error(probs, outcomes.astype(bool), bins=10)
        # With 10k samples, ECE should be very small if model is well-calibrated.
        assert ece < 0.02

    def test_constant_wrong_confidence_high_ece(self):
        """Always predict 90% but empirical rate is 50% → ECE ≈ 0.4."""
        probs = [0.9] * 1000
        outcomes = [True, False] * 500
        ece = metrics.expected_calibration_error(probs, outcomes, bins=10)
        assert ece == pytest.approx(0.4, abs=1e-3)

    def test_brier_perfect(self):
        probs = [1.0, 0.0, 1.0, 0.0]
        outcomes = [True, False, True, False]
        assert metrics.brier_score(probs, outcomes) == 0.0

    def test_brier_worst(self):
        probs = [1.0, 0.0, 1.0, 0.0]
        outcomes = [False, True, False, True]
        assert metrics.brier_score(probs, outcomes) == 1.0

    def test_reliability_bins_empty_input(self):
        assert metrics.reliability_bins([], [], bins=5) == []


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


class TestResultFromScore:
    @pytest.mark.parametrize(
        "home,away,expected",
        [
            (2, 1, "home_win"),
            (0, 3, "away_win"),
            (1, 1, "draw"),
            (0, 0, "draw"),
            (6, 2, "home_win"),
        ],
    )
    def test_deterministic_mapping(self, home: int, away: int, expected: str):
        assert metrics.result_from_score(home, away) == expected


class TestGoalMAE:
    def test_zero_error(self):
        assert metrics.goal_mae([2, 1], [0, 3], [2, 1], [0, 3]) == 0.0

    def test_symmetric_error(self):
        # Predicted 1-1, actual 0-0 → MAE = (1+1)/2 / 2 = 0.5 ... wait, per-match is (|1-0| + |1-0|) / 2 = 1.0, mean over 1 match = 1.0
        # With 2 matches: {(1-0) abs=1 + (1-0) abs=1 = 2, halved = 1.0; (1-0) abs=1 + (1-0) abs=1 = 2, halved=1.0}; mean = 1.0
        assert metrics.goal_mae([1, 1], [1, 1], [0, 0], [0, 0]) == 1.0

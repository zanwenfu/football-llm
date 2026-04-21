"""Tests for the Kelly-sized O/U backtest.

These tests lock in the paper's headline numbers against the committed halftime
prediction dump. If they ever drift, the paper's §6.3 claims need to be
re-verified against the new backtest output.
"""

from __future__ import annotations

import numpy as np
import pytest

from football_llm.eval import backtest


class TestKellyFraction:
    def test_zero_when_bet_has_no_edge(self):
        # p = 1/1.9 ≈ 0.526 → Kelly fraction exactly 0
        p = 1.0 / 1.9
        assert backtest._kelly_fraction(p, 1.9) == pytest.approx(0.0, abs=1e-6)

    def test_zero_when_negative_ev(self):
        """Kelly should never recommend a negative-EV bet."""
        assert backtest._kelly_fraction(0.3, 1.9) == 0.0

    def test_positive_when_edge(self):
        # p = 0.8 on 1.9 odds → f = (0.9 * 0.8 - 0.2) / 0.9 = 0.52 / 0.9 ≈ 0.578
        assert backtest._kelly_fraction(0.8, 1.9) == pytest.approx(0.5778, abs=1e-3)


class TestRunBacktest:
    def test_empty_inputs(self):
        result = backtest.run_backtest([], [])
        assert result.num_bets == 0
        assert result.final_bankroll == 1000.0
        assert result.roi == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            backtest.run_backtest([0.5, 0.8], [2.5])

    def test_all_correct_over_bets_are_profitable(self):
        """If the model is perfectly correct and confident, ROI is positive."""
        # Each match: model says P(over)=0.9, actual is over 2.5 (total=4 goals)
        p_over = [0.9] * 10
        totals = [4.0] * 10
        result = backtest.run_backtest(p_over, totals)
        assert result.num_bets == 10
        assert result.win_rate == 1.0
        assert result.roi > 0.0

    def test_edge_threshold_skips_marginal(self):
        """If edge is below 5%, bet is skipped."""
        # implied over = 1/1.9 = 0.526; our p_over = 0.56 → edge = 0.034 < 0.05
        p_over = [0.56] * 10
        totals = [4.0] * 10
        result = backtest.run_backtest(p_over, totals)
        assert result.num_bets == 0
        assert result.bets_skipped == 10
        assert result.final_bankroll == 1000.0

    def test_per_match_cap_binding(self):
        """At Kelly f=0.25 with large edge, per-match cap should bind."""
        # P(over)=0.99 → raw Kelly huge, sized Kelly = 0.25 * (0.9*0.99 - 0.01) / 0.9 ≈ 0.244
        # > cap 0.1, so first-bet stake should equal 10% of bankroll
        p_over = [0.99]
        totals = [4.0]
        result = backtest.run_backtest(p_over, totals)
        assert result.num_bets == 1
        first_bet = result.per_bet_log[0]
        assert first_bet["stake_frac"] == pytest.approx(0.10, abs=1e-6)


class TestBootstrap:
    def test_seeds_are_reproducible(self):
        returns = np.array([0.05, -0.10, 0.08, 0.03, -0.05])
        a = backtest.bootstrap_final_bankroll(returns, n_trials=100, seed=42)
        b = backtest.bootstrap_final_bankroll(returns, n_trials=100, seed=42)
        assert a.median_final == b.median_final
        assert a.percentile_5 == b.percentile_5

    def test_empty_returns_raises(self):
        with pytest.raises(ValueError):
            backtest.bootstrap_final_bankroll([])


# ---------------------------------------------------------------------------
# Integration test: halftime predictions → paper's published numbers
# ---------------------------------------------------------------------------


class TestPaperIntegration:
    """Locks in §6.3 numbers on the committed halftime prediction file.

    These tests will fail if the prediction file is regenerated with a
    different seed / adapter, which is deliberate — drift should force a
    review of the published numbers.
    """

    def test_halftime_kelly_backtest_matches_paper(self, halftime_df):
        p_over, totals = backtest.predictions_to_backtest_inputs(halftime_df, anonymized=False)
        result = backtest.run_backtest(p_over, totals)
        # Paper §6.3: 81.3% win rate, 1,468% ROI, -10.4% max DD on 64 named matches
        assert result.num_bets == 64
        assert result.win_rate == pytest.approx(0.8125, abs=0.005)
        assert result.roi == pytest.approx(14.684, abs=0.05)
        assert result.max_drawdown == pytest.approx(-0.104, abs=0.01)

    def test_sensitivity_grid_all_cells_profitable(self, halftime_df):
        """§6.2: strategy is profitable across all 9 cells of (f, cap)."""
        p_over, totals = backtest.predictions_to_backtest_inputs(halftime_df, anonymized=False)
        grid = backtest.sensitivity_grid(p_over, totals)
        assert (grid["roi"] > 0).all()
        # Drawdown should stay within -25% across all cells
        assert (grid["max_drawdown"] > -0.25).all()

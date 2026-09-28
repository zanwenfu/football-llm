"""Tests for the leakage audit (football_llm.eval.audit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_llm.eval import audit


def _preds(pred, gt, anonymized=False, fixture_ids=None, raw=None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "fixture_id": fixture_ids if fixture_ids is not None else range(len(pred)),
            "anonymized": anonymized,
            "pred_home": [p[0] for p in pred],
            "pred_away": [p[1] for p in pred],
            "gt_home": [g[0] for g in gt],
            "gt_away": [g[1] for g in gt],
        }
    )
    if raw is not None:
        df["raw_output"] = raw
    return df


class TestMirrorRates:
    def test_exact_and_mirror_are_disjoint(self):
        r = audit.mirror_rates([2, 0, 1, 3], [0, 2, 1, 3], [2, 2, 1, 0], [0, 0, 1, 1])
        # 2-0 vs 2-0 exact; 0-2 vs 2-0 mirror; 1-1 vs 1-1 exact (not also mirror); 3-3 miss
        assert r.exact == pytest.approx(0.5)
        assert r.mirror_only == pytest.approx(0.25)
        assert r.exact_or_mirror == pytest.approx(0.75)

    def test_draw_is_never_counted_as_mirror(self):
        r = audit.mirror_rates([1], [1], [1], [1])
        assert r.exact == 1.0
        assert r.mirror_only == 0.0

    def test_mirror_cases_lists_only_swapped_rows(self):
        df = _preds([(0, 1), (2, 0), (1, 1)], [(1, 0), (2, 0), (0, 0)])
        df["home_team"] = ["Tunisia", "Brazil", "Morocco"]
        df["away_team"] = ["France", "Serbia", "Croatia"]
        cases = audit.mirror_cases(df)
        assert list(cases["home_team"]) == ["Tunisia"]


class TestPermutationTest:
    def test_perfect_recall_is_significant(self):
        rng = np.random.default_rng(0)
        gt = [tuple(x) for x in rng.integers(0, 5, size=(64, 2))]
        r = audit.mirror_permutation_test(_preds(gt, gt), n_permutations=500)
        assert r.observed == 1.0
        assert r.p_value < 0.01
        assert r.null_mean < 0.3

    def test_constant_prediction_is_not_significant(self):
        rng = np.random.default_rng(1)
        gt = [tuple(x) for x in rng.integers(0, 4, size=(64, 2))]
        r = audit.mirror_permutation_test(_preds([(1, 0)] * 64, gt), n_permutations=500)
        # Shuffling the labels cannot change a constant prediction's hit rate.
        assert r.p_value == 1.0

    def test_deterministic_given_seed(self):
        gt = [(1, 0), (2, 1), (0, 0), (3, 1)] * 8
        pred = [(1, 0), (1, 2), (1, 1), (2, 0)] * 8
        a = audit.mirror_permutation_test(_preds(pred, gt), n_permutations=200, seed=7)
        b = audit.mirror_permutation_test(_preds(pred, gt), n_permutations=200, seed=7)
        assert a == b


class TestNamedVsAnon:
    def test_pairs_on_fixture_not_row_order(self):
        named = _preds([(1, 0)] * 3, [(1, 0)] * 3, anonymized=False, fixture_ids=[1, 2, 3])
        anon = _preds([(1, 0)] * 3, [(1, 0)] * 3, anonymized=True, fixture_ids=[3, 1, 2])
        named["ok"] = [True, True, False]  # fixtures 1, 2 right; 3 wrong
        anon["ok"] = [True, False, False]  # fixture 3 right; 1, 2 wrong
        r = audit.paired_named_vs_anon(pd.concat([named, anon]), "ok")
        # By fixture: 1 and 2 named-only, 3 anon-only. Row-order pairing would give (1, 0).
        assert (r.b, r.c) == (2, 1)

    def test_split_summary_on_committed_pregame(self, pregame_df):
        s = audit.split_summary(pregame_df)
        assert s.loc["named", "n"] == 64
        assert s.loc["named", "exact_score"] == pytest.approx(28 / 64)
        assert s.loc["anonymized", "exact_score"] == pytest.approx(7 / 64)


class TestTrivialBaselines:
    def test_halftime_rules(self):
        m = pd.DataFrame(
            {
                "gt_home": [2, 0, 1, 3],
                "gt_away": [0, 0, 1, 1],
                "halftime_home": [1, 0, 0, 2],
                "halftime_away": [0, 0, 1, 0],
            }
        )
        t = audit.trivial_baselines(m)
        # HT leader: home, draw, away, home vs actual home, draw, draw, home → 3/4
        assert t.loc["HT leader wins / HT×2 total", "1X2"] == pytest.approx(0.75)
        # HT×2 totals 2, 0, 2, 4 → under, under, under, over vs actual under, under, under, over
        assert t.loc["HT leader wins / HT×2 total", "ou_25"] == pytest.approx(1.0)
        assert t.loc["always home win / always under", "ou_25"] == pytest.approx(0.75)

    def test_pregame_has_no_halftime_rule(self):
        t = audit.trivial_baselines(pd.DataFrame({"gt_home": [1], "gt_away": [0]}))
        assert "HT leader wins / HT×2 total" not in t.index


class TestOutputSanity:
    def test_counts(self):
        df = _preds(
            [(0, 8), (1, 1), (2, 0)],
            [(0, 0), (1, 1), (2, 0)],
            raw=[
                "Prediction: home_win\nScore: 0-8",
                "Prediction: draw\nScore: 1-1",
                "Prediction: home_win\nScore: 2-0",
            ],
        )
        s = audit.output_sanity(df)
        assert s["predicted_draws"] == 1
        assert s["actual_draws"] == 2
        assert s["implausible_scorelines"] == 1
        assert s["label_contradicts_score"] == 1

    def test_text_label(self):
        assert audit.text_label("Prediction: Away_Win\nScore: 0-1") == "away_win"
        assert audit.text_label("no label here") is None


class TestPower:
    def test_smaller_edges_need_more_bets(self):
        assert audit.bets_to_detect_edge(0.02) > audit.bets_to_detect_edge(0.05)

    def test_order_of_magnitude(self):
        # sd ≈ 0.95 at even-ish odds: n ≈ (2.49 * 0.95 / 0.05)^2 ≈ 2,200
        assert 2_000 < audit.bets_to_detect_edge(0.05) < 2_500


class TestHalftimeLift:
    def test_pairs_within_one_split_only(self):
        pre = _preds([(1, 0)] * 4, [(1, 0)] * 4, fixture_ids=[1, 2, 1, 2])
        pre["anonymized"] = [False, False, True, True]
        pre["ok"] = [False, False, True, True]
        ht = pre.copy()
        ht["ok"] = [True, True, True, True]
        named = audit.halftime_lift(pre, ht, "ok", anonymized=False)
        anon = audit.halftime_lift(pre, ht, "ok", anonymized=True)
        assert (named.b, named.c) == (0, 2)
        assert (anon.b, anon.c) == (0, 0)


class TestHtDoubleRule:
    def test_goalless_first_half_predicts_under(self):
        assert audit.ht_double_p_over([0], [0])[0] == 0.0

    def test_matches_poisson_with_doubled_rate(self):
        # 1-1 at HT → λ = 4 → P(X > 2) = 1 - e^-4 (1 + 4 + 8)
        expected = 1 - np.exp(-4) * 13
        assert audit.ht_double_p_over([1], [1])[0] == pytest.approx(expected)


class TestNullDistribution:
    def test_shape_and_range(self):
        df = _preds([(1, 0), (2, 1)] * 10, [(0, 1), (2, 1)] * 10)
        null = audit.mirror_null_distribution(df, n_permutations=300, seed=3)
        assert null.shape == (300,)
        assert ((null >= 0) & (null <= 1)).all()


class TestAuditFigures:
    def test_figures_render(self):
        import matplotlib

        matplotlib.use("Agg")
        from football_llm.eval import backtest, figures

        acc = figures.figure_accuracy_by_model(
            {"Pregame": [("A", 40, 64, "named"), ("B", 30, 64, "baseline")]},
            xlabel="x",
            reference=(0.526, "break-even"),
        )
        mirror = figures.figure_mirror_null({"A": (np.full(50, 0.1), 0.6, 0.0001, "named")})
        res = backtest.run_backtest(np.array([0.9, 0.1, 0.8]), np.array([4, 1, 3]))
        traj = figures.figure_flat_odds_trajectories({"A": (res, "rule")})
        for fig in (acc, mirror, traj):
            assert fig.axes

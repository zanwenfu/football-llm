"""Tests for the Dixon-Coles score-model baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_llm.baselines import dixon_coles as dc


def _league(n_rounds: int = 20, seed: int = 0) -> pd.DataFrame:
    """Round-robin among four teams with known, very different strengths."""
    rng = np.random.default_rng(seed)
    attack = {"Strong": 0.6, "Good": 0.2, "Weak": -0.2, "Poor": -0.6}
    teams = list(attack)
    rows = []
    for r in range(n_rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = np.exp(0.2 + attack[h] - attack[a])
                mu = np.exp(0.2 + attack[a] - attack[h])
                rows.append(
                    {
                        "world_cup_year": 2000 + r,
                        "home_team": h,
                        "away_team": a,
                        "home_goals": rng.poisson(lam),
                        "away_goals": rng.poisson(mu),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model() -> dc.DixonColesModel:
    return dc.fit(_league(), ridge=0.1)


class TestFit:
    def test_recovers_strength_ordering(self, model):
        att = dict(zip(model.teams, model.attack))
        assert att["Strong"] > att["Good"] > att["Weak"] > att["Poor"]

    def test_strong_team_expected_to_outscore_poor(self, model):
        lam, mu = model.expected_goals("Strong", "Poor")
        assert lam > 2 * mu

    def test_heavy_ridge_shrinks_to_average(self):
        m = dc.fit(_league(), ridge=1e4)
        assert np.abs(m.attack).max() < 0.01
        assert np.abs(m.defence).max() < 0.01

    def test_rho_within_bounds(self, model):
        assert dc.RHO_BOUNDS[0] <= model.rho <= dc.RHO_BOUNDS[1]


class TestPredict:
    def test_score_matrix_is_a_distribution(self, model):
        m = model.score_matrix("Good", "Weak")
        assert m.shape == (dc.MAX_GOALS + 1, dc.MAX_GOALS + 1)
        assert m.sum() == pytest.approx(1.0)
        assert (m >= 0).all()

    def test_unseen_team_is_average(self, model):
        assert model.expected_goals("Atlantis", "Atlantis")[1] == pytest.approx(np.exp(model.base))

    def test_prediction_schema_and_probabilities(self, model):
        fixtures = pd.DataFrame(
            {
                "fixture_id": [1, 2],
                "home_team": ["Strong", "Poor"],
                "away_team": ["Poor", "Strong"],
                "final_home": [3, 0],
                "final_away": [0, 2],
            }
        )
        out = dc.predict(model, fixtures)
        probs = out[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
        assert np.allclose(probs, 1.0)
        assert list(out["pred_result"]) == ["home_win", "away_win"]
        assert list(out["gt_result"]) == ["home_win", "away_win"]
        assert out["p_over_25"].between(0, 1).all()


class TestRidgeSelection:
    def test_selects_from_grid_using_last_training_year_only(self):
        league = _league(n_rounds=6)
        chosen = dc.select_ridge(league, grid=(0.1, 1e4))
        # Strengths are real and stable across "tournaments", so light shrinkage wins.
        assert chosen == 0.1


class TestRealData:
    def test_labels_use_final_score_training_uses_90_minutes(self):
        m = dc.load_matches()
        final = m[(m["home_team"] == "Argentina") & (m["away_team"] == "France")]
        final = final[final["world_cup_year"] == 2022].iloc[0]
        assert (final["final_home"], final["final_away"]) == (3, 3)  # after extra time
        assert (final["home_goals"], final["away_goals"]) == (2, 2)  # at 90 minutes

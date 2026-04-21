"""Tests for the prediction file loader and pairing."""

from __future__ import annotations

import pandas as pd
import pytest

from football_llm.eval import loader


class TestLoadPredictions:
    def test_pregame_expected_columns(self, pregame_df):
        expected = {
            "fixture_id",
            "home_team",
            "away_team",
            "anonymized",
            "gt_home",
            "gt_away",
            "gt_result",
            "pred_home",
            "pred_away",
            "pred_result",
            "raw_output",
            "regime",
            "correct_result",
            "correct_score",
            "pred_total",
            "gt_total",
            "gt_over_25",
            "pred_over_25",
            "correct_ou_25",
        }
        assert expected.issubset(set(pregame_df.columns))

    def test_halftime_has_halftime_fields(self, halftime_df):
        assert "halftime_home" in halftime_df.columns
        assert "halftime_away" in halftime_df.columns

    def test_pregame_row_count(self, pregame_df):
        # 64 matches × 2 anonymization variants
        assert len(pregame_df) == 128

    def test_halftime_row_count(self, halftime_df):
        assert len(halftime_df) == 128

    def test_pred_result_derived_from_score(self, pregame_df):
        """Canonical: pred_result always reflects pred_home vs. pred_away."""
        home_wins = pregame_df[pregame_df["pred_home"] > pregame_df["pred_away"]]
        assert (home_wins["pred_result"] == "home_win").all()
        away_wins = pregame_df[pregame_df["pred_home"] < pregame_df["pred_away"]]
        assert (away_wins["pred_result"] == "away_win").all()
        draws = pregame_df[pregame_df["pred_home"] == pregame_df["pred_away"]]
        assert (draws["pred_result"] == "draw").all()

    def test_missing_regime_raises(self):
        with pytest.raises(ValueError, match="Unknown regime"):
            loader.load_predictions("not_a_real_regime")

    def test_missing_file_raises_helpful_error(self, tmp_path):
        """Empty dir → FileNotFoundError with regeneration hint."""
        with pytest.raises(FileNotFoundError, match="run_inference"):
            loader.load_predictions("pregame", results_dir=tmp_path)


class TestPairOnMatch:
    def test_pair_pregame_halftime(self, pregame_df, halftime_df):
        merged = loader.pair_on_match(pregame_df, halftime_df)
        assert len(merged) == 128
        # Columns from both frames must coexist with _a/_b suffixes
        assert "correct_result_a" in merged.columns
        assert "correct_result_b" in merged.columns

    def test_duplicate_keys_raise(self):
        df = pd.DataFrame(
            {
                "fixture_id": [1, 1],
                "anonymized": [False, False],
                "gt_home": [0, 0],
                "gt_away": [0, 0],
                "gt_result": ["draw", "draw"],
                "pred_home": [0, 0],
                "pred_away": [0, 0],
                "pred_result": ["draw", "draw"],
            }
        )
        with pytest.raises(ValueError, match="duplicate"):
            loader.pair_on_match(df, df)

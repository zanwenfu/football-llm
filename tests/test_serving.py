"""Tests for the serving layer — prompt builder and output parser.

These guard against silent breakage: the parser's regex must stay in sync with
the training prompt format, and the prompt builder must emit text that matches
what the model saw during SFT.
"""

from __future__ import annotations

import pytest

from football_llm.serving.api import (
    MatchContext,
    PredictRequest,
    TeamStats,
    build_team_block,
    build_user_message,
    parse_model_output,
)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _sample_team(name: str = "Argentina") -> TeamStats:
    return TeamStats(
        name=name,
        goals=450,
        goals_per_90=0.35,
        assists=180,
        avg_rating=7.2,
        top_scorer_goals=200,
        yellows=120,
        reds=2,
        tackles_per_90=0.6,
        duels_pct=55,
        pass_accuracy=72,
        formation="4-3-3",
        coach="Lionel Scaloni",
    )


class TestBuildTeamBlock:
    def test_round_trips_through_training_format(self):
        stats = _sample_team()
        block = build_team_block(stats, "Home")
        # All training-format sections must be present
        assert "Argentina (Home)" in block
        assert "Coach: Lionel Scaloni" in block
        assert "Formation: 4-3-3" in block
        assert "450 goals" in block
        assert "0.35/90" in block
        assert "180 assists" in block
        assert "Top scorer: 200 goals" in block
        assert "120 yellows, 2 reds" in block
        assert "Tackles/90: 0.60" in block
        assert "Duels: 55%" in block
        assert "Passing: 72% accuracy" in block

    def test_handles_zero_values(self):
        stats = TeamStats(name="Test", goals=0, assists=0, goals_per_90=0.0)
        block = build_team_block(stats, "Away")
        # Zeros render as "-" to match training format
        assert "Test (Away)" in block


class TestBuildUserMessage:
    def test_contains_all_three_sections(self):
        req = PredictRequest(
            home_team=_sample_team("Argentina"),
            away_team=_sample_team("France"),
            match=MatchContext(tournament="WC 2022", stage="Final", venue="Lusail"),
        )
        msg = build_user_message(req)
        assert "WC 2022 | Final | Lusail" in msg
        assert "Argentina (Home)" in msg
        assert "France (Away)" in msg
        assert msg.endswith("Predict result, score, and reasoning.")


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------


class TestParseModelOutput:
    def test_canonical_format(self):
        text = "Prediction: home_win\nScore: 2-1\nReasoning: Argentina dominates midfield."
        r = parse_model_output(text)
        assert r["prediction"] == "home_win"
        assert r["score"] == "2-1"
        assert r["reasoning"].startswith("Argentina dominates")

    def test_score_overrides_text_label(self):
        """Paper §3.3: score derives the result, prose is ignored."""
        text = "Prediction: home_win\nScore: 0-3\nReasoning: ..."
        r = parse_model_output(text)
        assert r["prediction"] == "away_win"  # derived from 0-3, not from "home_win"

    def test_draw_from_score(self):
        text = "Prediction: home_win\nScore: 1-1\nReasoning: close match"
        r = parse_model_output(text)
        assert r["prediction"] == "draw"

    def test_em_dash_score_separator(self):
        text = "Prediction: away_win\nScore: 0–2\nReasoning: ..."  # en-dash variant
        r = parse_model_output(text)
        assert r["score"] == "0-2"
        assert r["prediction"] == "away_win"

    def test_no_score_falls_back_to_text_label(self):
        text = "Prediction: draw\nReasoning: tight game"
        r = parse_model_output(text)
        assert r["prediction"] == "draw"
        assert r["score"] is None

    def test_unparseable_returns_none(self):
        r = parse_model_output("This output doesn't match the format at all.")
        assert r["prediction"] is None
        assert r["score"] is None

    def test_reasoning_captures_multiline(self):
        text = "Prediction: home_win\nScore: 2-0\nReasoning: line one\nline two continues."
        r = parse_model_output(text)
        assert "line one" in r["reasoning"]
        assert "line two" in r["reasoning"]

    @pytest.mark.parametrize(
        "text,expected_pred",
        [
            ("Prediction: home_win\nScore: 2-1", "home_win"),
            ("Prediction: away_win\nScore: 0-1", "away_win"),
            ("Prediction: draw\nScore: 1-1", "draw"),
        ],
    )
    def test_result_consistent_with_score(self, text, expected_pred):
        assert parse_model_output(text)["prediction"] == expected_pred

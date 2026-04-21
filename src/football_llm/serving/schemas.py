"""Pydantic schemas for the FastAPI serving layer.

Kept in a dedicated module so tests and the Gradio UI can import them without
pulling in FastAPI / uvicorn.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Regime = Literal["pregame", "halftime", "halftime_events"]


class TeamStats(BaseModel):
    """Aggregated team statistics (mirrors the compact training format)."""

    name: str = Field(..., description="Team name", examples=["Argentina"])
    goals: int = Field(0, ge=0, description="Total goals (starting XI, prior 3 seasons)")
    goals_per_90: float = Field(0.0, ge=0.0, description="Average goals per 90 minutes")
    assists: int = Field(0, ge=0, description="Total assists")
    avg_rating: float = Field(0.0, ge=0.0, le=10.0, description="Average player rating (0-10)")
    top_scorer_goals: int = Field(0, ge=0, description="Goals by the top scorer in the squad")
    yellows: int = Field(0, ge=0, description="Total yellow cards")
    reds: int = Field(0, ge=0, description="Total red cards")
    tackles_per_90: float = Field(0.0, ge=0.0, description="Average tackles per 90")
    duels_pct: float = Field(0.0, ge=0.0, le=100.0, description="Duel win percentage")
    pass_accuracy: float = Field(0.0, ge=0.0, le=100.0, description="Average pass accuracy %")
    formation: str = Field("?", description="Tactical formation", examples=["4-3-3"])
    coach: str = Field("?", description="Head coach name")


class MatchContext(BaseModel):
    """Match metadata."""

    tournament: str = Field("World Cup 2026", description="Tournament name + year")
    stage: str = Field("Group Stage", description="Competition stage")
    venue: str = Field("Unknown", description="Stadium / venue name")


class HalftimeScore(BaseModel):
    """Observed halftime score — required for halftime / halftime_events regimes."""

    home: int = Field(..., ge=0, le=9, description="Home goals at halftime")
    away: int = Field(..., ge=0, le=9, description="Away goals at halftime")


class PredictRequest(BaseModel):
    """Request body for /predict."""

    home_team: TeamStats
    away_team: TeamStats
    match: MatchContext = Field(default_factory=MatchContext)
    regime: Regime = Field(
        "pregame",
        description="Prompt regime: pregame | halftime | halftime_events",
    )
    halftime_score: HalftimeScore | None = Field(
        None,
        description="Observed halftime score — required for halftime / halftime_events regimes",
    )
    first_half_events: list[str] | None = Field(
        None,
        description=(
            "Chronological first-half event strings (≤45') — required for halftime_events. "
            "Example: ['10\\' Argentina goal (penalty)', '25\\' Saudi Arabia yellow card']"
        ),
        max_length=40,
    )

    @model_validator(mode="after")
    def _validate_regime_fields(self):
        if self.regime in ("halftime", "halftime_events") and self.halftime_score is None:
            raise ValueError("halftime_score is required for halftime / halftime_events regimes")
        if self.regime == "halftime_events" and not self.first_half_events:
            raise ValueError("first_half_events is required (non-empty) for halftime_events regime")
        return self


class PredictResponse(BaseModel):
    """Structured prediction response."""

    prediction: str = Field(..., description="home_win | draw | away_win (derived from score)")
    score: str = Field(..., description="Predicted score, e.g. '2-1'")
    pred_home: int = Field(..., description="Predicted home goals")
    pred_away: int = Field(..., description="Predicted away goals")
    over_2_5_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="P(total goals > 2.5) via Poisson approximation on predicted total",
    )
    regime: Regime = Field(..., description="Prompt regime used")
    reasoning: str = Field(..., description="Brief reasoning paragraph from the model")
    raw_output: str = Field(..., description="Raw model output text")

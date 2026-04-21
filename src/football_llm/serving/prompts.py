"""Prompt construction — mirrors generate_training_data.py format.

Kept in its own module so tests can verify the serving prompts match the
training format exactly without starting a FastAPI app.
"""

from __future__ import annotations

from football_llm.serving.schemas import PredictRequest, TeamStats

SYSTEM_PROMPT = (
    "You are a football match prediction model. "
    "Given team stats, predict the result, score, and brief reasoning."
)


def _fmt(val, decimals: int = 1) -> str:
    """Format a number, returning '-' if zero/missing."""
    if val is None or val == 0:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(int(val))


def build_team_block(stats: TeamStats, role: str) -> str:
    """Build a compact team stats block matching the training format."""
    lines = [
        f"{stats.name} ({role}) | Coach: {stats.coach} | Formation: {stats.formation}",
        f"Squad: 11 starters | Avg Rating: {_fmt(stats.avg_rating)}",
        (
            f"Attack: {_fmt(stats.goals, 0)} goals ({_fmt(stats.goals_per_90, 2)}/90) | "
            f"{_fmt(stats.assists, 0)} assists | Top scorer: {_fmt(stats.top_scorer_goals, 0)} goals"
        ),
        (
            f"Defense: {_fmt(stats.yellows, 0)} yellows, {_fmt(stats.reds, 0)} reds | "
            f"Tackles/90: {_fmt(stats.tackles_per_90, 2)} | Duels: {_fmt(stats.duels_pct, 0)}%"
        ),
        f"Passing: {_fmt(stats.pass_accuracy, 0)}% accuracy",
    ]
    return "\n".join(lines)


def build_user_message(req: PredictRequest) -> str:
    """Construct the user message matching the regime-appropriate training format."""
    header = f"{req.match.tournament} | {req.match.stage} | {req.match.venue}"
    home_block = build_team_block(req.home_team, "Home")
    away_block = build_team_block(req.away_team, "Away")

    sections = [header, "", home_block, "", away_block, ""]

    if req.regime == "pregame":
        sections.append("Predict result, score, and reasoning.")
        return "\n".join(sections)

    # Halftime + halftime_events share a halftime score line.
    assert req.halftime_score is not None
    sections.append(
        f"Halftime Score: {req.home_team.name} {req.halftime_score.home} - "
        f"{req.halftime_score.away} {req.away_team.name}"
    )

    if req.regime == "halftime":
        sections.append(
            "Given the halftime state, predict the FINAL result, FINAL score, and brief reasoning."
        )
        return "\n".join(sections)

    # halftime_events
    assert req.first_half_events  # validated in schema
    sections.append("First-half events: " + "; ".join(req.first_half_events))
    sections.append(
        "Given the halftime state and first-half events, "
        "predict the FINAL result, FINAL score, and brief reasoning."
    )
    return "\n".join(sections)

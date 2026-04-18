"""
Football-LLM: FastAPI serving layer.

Wraps the local vLLM OpenAI-compatible endpoint with a domain-specific
/predict API that constructs the same compact prompt format used in training.

Usage:
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8001 --reload
"""

import re
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "football-llm"  # LoRA adapter name registered in vLLM
SYSTEM_PROMPT = (
    "You are a football match prediction model. "
    "Given team stats, predict the result, score, and brief reasoning."
)

logger = logging.getLogger("football-llm-api")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TeamStats(BaseModel):
    """Aggregated team statistics (mirrors the compact training format)."""
    name: str = Field(..., description="Team name", examples=["Argentina"])
    goals: int = Field(0, description="Total goals (starting XI, prior 3 seasons)")
    goals_per_90: float = Field(0.0, description="Average goals per 90 minutes")
    assists: int = Field(0, description="Total assists")
    avg_rating: float = Field(0.0, description="Average player rating (0-10)")
    top_scorer_goals: int = Field(0, description="Goals by the top scorer in the squad")
    yellows: int = Field(0, description="Total yellow cards")
    reds: int = Field(0, description="Total red cards")
    tackles_per_90: float = Field(0.0, description="Average tackles per 90")
    duels_pct: float = Field(0.0, description="Duel win percentage")
    pass_accuracy: float = Field(0.0, description="Average pass accuracy %")
    formation: str = Field("?", description="Tactical formation", examples=["4-3-3"])
    coach: str = Field("?", description="Head coach name")


class MatchContext(BaseModel):
    """Match metadata."""
    tournament: str = Field("World Cup 2026", description="Tournament name + year")
    stage: str = Field("Group Stage", description="Competition stage")
    venue: str = Field("Unknown", description="Stadium / venue name")


class PredictRequest(BaseModel):
    """Request body for /predict."""
    home_team: TeamStats
    away_team: TeamStats
    match: MatchContext = Field(default_factory=MatchContext)


class PredictResponse(BaseModel):
    """Structured prediction response."""
    prediction: str = Field(..., description="home_win | draw | away_win")
    score: str = Field(..., description="Predicted score, e.g. '2-1'")
    reasoning: str = Field(..., description="Brief reasoning paragraph")
    raw_output: str = Field(..., description="Raw model output text")


# ---------------------------------------------------------------------------
# Prompt construction (mirrors generate_training_data.py format)
# ---------------------------------------------------------------------------

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
    """Construct the user message in the exact compact training format."""
    header = f"{req.match.tournament} | {req.match.stage} | {req.match.venue}"
    home_block = build_team_block(req.home_team, "Home")
    away_block = build_team_block(req.away_team, "Away")

    sections = [
        header,
        "",
        home_block,
        "",
        away_block,
        "",
        "Predict result, score, and reasoning.",
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_model_output(text: str) -> dict:
    """Parse the model's structured output into prediction, score, reasoning."""
    result = {"prediction": None, "score": None, "reasoning": None}

    # Extract prediction line
    pred_match = re.search(r"Prediction:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
    if pred_match:
        pred_text = pred_match.group(1).strip().lower()
        if "draw" in pred_text:
            result["prediction"] = "draw"
        elif "home" in pred_text:
            result["prediction"] = "home_win"
        elif "away" in pred_text:
            result["prediction"] = "away_win"
        elif "win" in pred_text:
            result["prediction"] = "home_win"  # ambiguous → default home

    # Extract score — always derive result from score when available
    score_match = re.search(r"Score:\s*(\d+)\s*[-–]\s*(\d+)", text)
    if score_match:
        home_goals = int(score_match.group(1))
        away_goals = int(score_match.group(2))
        result["score"] = f"{home_goals}-{away_goals}"
        # Score overrides text label for consistency
        if home_goals > away_goals:
            result["prediction"] = "home_win"
        elif away_goals > home_goals:
            result["prediction"] = "away_win"
        else:
            result["prediction"] = "draw"

    # Extract reasoning
    reason_match = re.search(r"Reasoning:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Football-LLM API",
    description="Predict FIFA World Cup match outcomes using a fine-tuned Llama 3.1 8B model.",
    version="1.0.0",
)

# Lazy-init OpenAI client (connects to local vLLM server)
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=VLLM_BASE_URL, api_key="unused")
    return _client


@app.get("/health")
async def health():
    """Health check — also verifies vLLM connectivity."""
    try:
        client = get_client()
        models = client.models.list()
        available = [m.id for m in models.data]
        return {
            "status": "healthy",
            "vllm_connected": True,
            "models": available,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "vllm_connected": False,
            "error": str(e),
        }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Predict a football match outcome.

    Constructs the same compact prompt used during training, sends it to the
    local vLLM server, and returns a structured prediction.
    """
    user_message = build_user_message(req)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.info(
        f"Predicting: {req.home_team.name} vs {req.away_team.name} "
        f"({req.match.stage})"
    )

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.1,
            top_p=0.9,
        )
        raw_output = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"vLLM request failed: {e}")
        raise HTTPException(status_code=503, detail=f"vLLM server error: {e}")

    parsed = parse_model_output(raw_output)

    if parsed["prediction"] is None:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse model output: {raw_output[:300]}",
        )

    return PredictResponse(
        prediction=parsed["prediction"],
        score=parsed["score"] or "?-?",
        reasoning=parsed["reasoning"] or "No reasoning provided.",
        raw_output=raw_output,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

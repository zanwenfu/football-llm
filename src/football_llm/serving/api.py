"""Football-LLM: FastAPI serving layer.

Wraps the local vLLM OpenAI-compatible endpoint with a domain-specific
/predict API that constructs the training prompt format for one of three
regimes (pregame / halftime / halftime_events), calls the LoRA adapter, and
returns a structured prediction plus the Poisson-converted P(over 2.5).

Configuration is entirely via env vars (all optional, sensible defaults):

    VLLM_BASE_URL     vLLM endpoint (default http://localhost:8000/v1)
    MODEL_NAME        LoRA adapter name registered in vLLM (default football-llm)
    API_TOKEN         If set, Authorization: Bearer <API_TOKEN> is required
                      on /predict. Unset = open (dev mode).
    MAX_OUTPUT_TOKENS Max tokens to generate (default 300).
    TEMPERATURE       Sampling temperature (default 0.1).
    TOP_P             Nucleus sampling cutoff (default 0.9).

Usage:
    # Standalone:
    uvicorn football_llm.serving.api:app --host 0.0.0.0 --port 8001

    # With docker-compose:
    docker compose up
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import Depends, FastAPI, Header, HTTPException, status
from openai import OpenAI

from football_llm.eval.poisson import p_over_25
from football_llm.serving.parsing import parse_model_output
from football_llm.serving.prompts import SYSTEM_PROMPT, build_team_block, build_user_message
from football_llm.serving.schemas import (
    HalftimeScore,
    MatchContext,
    PredictRequest,
    PredictResponse,
    Regime,
    TeamStats,
)

# Re-exports preserved so existing imports / tests keep working.
__all__ = [
    "HalftimeScore",
    "MatchContext",
    "PredictRequest",
    "PredictResponse",
    "Regime",
    "TeamStats",
    "app",
    "build_team_block",
    "build_user_message",
    "parse_model_output",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "football-llm")
API_TOKEN = os.environ.get("API_TOKEN")  # None = auth disabled
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "300"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))

logger = logging.getLogger("football_llm.api")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Football-LLM API",
    description=(
        "Predict FIFA World Cup match outcomes using a QLoRA-fine-tuned "
        "Llama 3.1 8B. Supports pregame, halftime-conditioned, and "
        "halftime+first-half-events prompt regimes."
    ),
    version="0.2.0",
)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=VLLM_BASE_URL, api_key="unused")
    return _client


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def require_token(authorization: str | None = Header(None)) -> None:
    """Optional bearer-token auth, active only when API_TOKEN env var is set."""
    if API_TOKEN is None:
        return
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )
    if authorization.removeprefix("Bearer ").strip() != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness + vLLM connectivity check."""
    try:
        models = get_client().models.list()
        return {
            "status": "healthy",
            "vllm_connected": True,
            "models": [m.id for m in models.data],
            "auth_enabled": API_TOKEN is not None,
            "version": app.version,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "vllm_connected": False,
            "error": str(e),
            "auth_enabled": API_TOKEN is not None,
            "version": app.version,
        }


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_token)])
async def predict(req: PredictRequest) -> PredictResponse:
    """Predict a match outcome across three supported prompt regimes.

    Pregame:          team stats only.
    Halftime:         team stats + observed halftime score.
    Halftime+events:  team stats + halftime score + first-half events (≤45').
    """
    t0 = time.perf_counter()
    user_message = build_user_message(req)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.info(
        "predict regime=%s home=%s away=%s stage=%s",
        req.regime,
        req.home_team.name,
        req.away_team.name,
        req.match.stage,
    )

    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        raw_output = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error("vLLM request failed: %s", e)
        raise HTTPException(status_code=503, detail=f"vLLM server error: {e}") from e

    parsed = parse_model_output(raw_output)
    if parsed["prediction"] is None or parsed["pred_home"] is None:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse model output: {raw_output[:300]!r}",
        )

    lambda_total = float(parsed["pred_home"] + parsed["pred_away"])
    p_over = p_over_25(lambda_total)
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "predict done pred=%s score=%s p_over=%.3f latency_ms=%.0f",
        parsed["prediction"],
        parsed["score"],
        p_over,
        latency_ms,
    )

    return PredictResponse(
        prediction=parsed["prediction"],
        score=parsed["score"] or "?-?",
        pred_home=parsed["pred_home"],
        pred_away=parsed["pred_away"],
        over_2_5_probability=p_over,
        regime=req.regime,
        reasoning=parsed["reasoning"] or "No reasoning provided.",
        raw_output=raw_output,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8001"))
    )

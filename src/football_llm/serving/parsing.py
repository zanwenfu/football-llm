"""Parse the model's free-form output into structured predictions.

The model is trained to emit:
    Prediction: home_win
    Score: 2-1
    Reasoning: ...

This parser is intentionally robust:
- Accepts en-dash/em-dash/hyphen separators in the score.
- **Derives the result from the score** (§3.3 of paper), overriding the prose
  label — the model's text label is occasionally inconsistent with its score
  and the score is the downstream signal.
- Never raises on malformed input — callers check `prediction is None`.
"""

from __future__ import annotations

import re
from typing import TypedDict


class ParsedOutput(TypedDict):
    prediction: str | None  # "home_win" | "draw" | "away_win" | None
    score: str | None  # "H-A" or None
    pred_home: int | None
    pred_away: int | None
    reasoning: str | None


_SCORE_RE = re.compile(r"Score:\s*(\d+)\s*[-–—]\s*(\d+)")
_PRED_RE = re.compile(r"Prediction:\s*(.*?)(?:\n|$)", re.IGNORECASE)
_REASON_RE = re.compile(r"Reasoning:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _result_from_score(home: int, away: int) -> str:
    if home > away:
        return "home_win"
    if away > home:
        return "away_win"
    return "draw"


def parse_model_output(text: str) -> ParsedOutput:
    result: ParsedOutput = {
        "prediction": None,
        "score": None,
        "pred_home": None,
        "pred_away": None,
        "reasoning": None,
    }

    # Score first — it's the authoritative source of the result.
    score_match = _SCORE_RE.search(text)
    if score_match:
        h, a = int(score_match.group(1)), int(score_match.group(2))
        result["pred_home"] = h
        result["pred_away"] = a
        result["score"] = f"{h}-{a}"
        result["prediction"] = _result_from_score(h, a)
    else:
        # Fall back to text label if no score.
        pred_match = _PRED_RE.search(text)
        if pred_match:
            pred_text = pred_match.group(1).strip().lower()
            if "draw" in pred_text:
                result["prediction"] = "draw"
            elif "home" in pred_text:
                result["prediction"] = "home_win"
            elif "away" in pred_text:
                result["prediction"] = "away_win"

    reason_match = _REASON_RE.search(text)
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()

    return result

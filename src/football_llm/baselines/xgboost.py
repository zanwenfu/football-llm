"""XGBoost tabular baseline on identical features to the LLM.

This baseline exists to answer a single methodological question (§5.3):
*what does the LLM add beyond the same features, rendered as numbers instead
of natural language?*

Training:
    python -m football_llm.baselines.xgboost train --regime pregame
    python -m football_llm.baselines.xgboost train --regime halftime

Evaluation:
    python -m football_llm.baselines.xgboost predict --regime halftime

Features extracted from match_contexts.json (per team + home/away differentials):
    squad totals    : goals, assists, appearances, minutes, shots, shots_on_target
    per-90 rates    : goals_per_90
    discipline      : yellow_cards, red_cards
    position counts : attackers, midfielders, defenders, goalkeepers
    team size       : num_starters_with_data
    halftime state  : hh, ha, home_lead, away_lead, tied (halftime regime only)

The XGBoost config matches the paper (§3.5): 200 estimators, max_depth=4,
learning_rate=0.05, no feature selection or tuning beyond defaults. This is
deliberate — the question is whether these features are sufficient, not
whether XGBoost can be tuned to win.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb

from football_llm.eval.metrics import Result, result_from_score
from football_llm.paths import PROCESSED_DIR, RESULTS_DIR

logger = logging.getLogger("football_llm.baselines.xgboost")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

Regime = Literal["pregame", "halftime"]
RESULT_LABELS: list[Result] = ["home_win", "draw", "away_win"]
LABEL_TO_IDX: dict[Result, int] = {r: i for i, r in enumerate(RESULT_LABELS)}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_BASE_TEAM_FIELDS = [
    ("total_goals", "team_total_goals"),
    ("total_assists", "team_total_assists"),
    ("total_appearances", "team_total_appearances"),
    ("total_minutes", "team_total_minutes"),
    ("avg_goals_per_90", "team_avg_goals_per_90"),
    ("total_shots", "team_total_shots_total"),
    ("total_shots_on_target", "team_total_shots_on_target"),
    ("yellows", "team_total_cards_yellow"),
    ("reds", "team_total_cards_red"),
    ("num_starters", "num_starters_with_data"),
]

_POSITIONS = ["Attacker", "Midfielder", "Defender", "Goalkeeper"]


def _team_features(profile: dict) -> dict[str, float]:
    """Flatten one team's profile into scalar features."""
    out: dict[str, float] = {}
    for out_key, src_key in _BASE_TEAM_FIELDS:
        v = profile.get(src_key, 0) or 0
        out[out_key] = float(v) if not _is_nan(v) else 0.0
    pos = profile.get("position_breakdown", {}) or {}
    for p in _POSITIONS:
        out[f"pos_{p.lower()}"] = float(pos.get(p, 0) or 0)
    return out


def _is_nan(v) -> bool:
    try:
        return isinstance(v, float) and np.isnan(v)
    except TypeError:
        return False


def _diff(home: dict[str, float], away: dict[str, float]) -> dict[str, float]:
    """Home - away differentials for every numeric feature."""
    return {f"diff_{k}": home[k] - away[k] for k in home if k in away}


def _halftime_features(match: dict) -> dict[str, float]:
    """Halftime-state features, only relevant in halftime regime."""
    hh = int(match.get("halftime_home") or 0)
    ha = int(match.get("halftime_away") or 0)
    return {
        "ht_home": float(hh),
        "ht_away": float(ha),
        "ht_total": float(hh + ha),
        "ht_diff": float(hh - ha),
    }


def extract_features(match: dict, regime: Regime) -> dict[str, float]:
    """Extract flat feature dict for a single match context."""
    home = _team_features(match["home_profile"])
    away = _team_features(match["away_profile"])
    row: dict[str, float] = {}
    for k, v in home.items():
        row[f"home_{k}"] = v
    for k, v in away.items():
        row[f"away_{k}"] = v
    row.update(_diff(home, away))
    if regime == "halftime":
        row.update(_halftime_features(match))
    return row


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


@dataclass
class XGBoostArtifacts:
    clf: xgb.XGBClassifier
    reg_home: xgb.XGBRegressor
    reg_away: xgb.XGBRegressor
    feature_names: list[str]


def load_match_contexts() -> list[dict]:
    path = PROCESSED_DIR / "match_contexts.json"
    with path.open() as f:
        return json.load(f)


def build_dataset(contexts: list[dict], regime: Regime) -> pd.DataFrame:
    """Return a DataFrame with all features + ground-truth labels.

    Drops matches that have neither halftime state nor a completed score.
    """
    rows = []
    for ctx in contexts:
        if ctx.get("home_goals") is None or ctx.get("away_goals") is None:
            continue
        if regime == "halftime" and (
            ctx.get("halftime_home") is None or ctx.get("halftime_away") is None
        ):
            continue
        feats = extract_features(ctx, regime)
        feats["fixture_id"] = int(ctx["fixture_id"])
        feats["world_cup_year"] = int(ctx["world_cup_year"])
        feats["home_goals"] = int(ctx["home_goals"])
        feats["away_goals"] = int(ctx["away_goals"])
        feats["gt_result"] = result_from_score(feats["home_goals"], feats["away_goals"])
        rows.append(feats)
    return pd.DataFrame(rows)


def temporal_split(df: pd.DataFrame, eval_year: int = 2022) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict temporal split — train on pre-eval_year, evaluate on eval_year only."""
    train = df[df["world_cup_year"] < eval_year].copy()
    eval_ = df[df["world_cup_year"] == eval_year].copy()
    return train.reset_index(drop=True), eval_.reset_index(drop=True)


def _fit_models(
    X: pd.DataFrame, y_result: pd.Series, y_home: pd.Series, y_away: pd.Series
) -> XGBoostArtifacts:
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    clf.fit(X, y_result.map(LABEL_TO_IDX))
    reg_home = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )
    reg_away = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )
    reg_home.fit(X, y_home)
    reg_away.fit(X, y_away)
    return XGBoostArtifacts(
        clf=clf, reg_home=reg_home, reg_away=reg_away, feature_names=list(X.columns)
    )


def _predict(artifacts: XGBoostArtifacts, X: pd.DataFrame) -> pd.DataFrame:
    """Return per-row predictions: pred_result, pred_home, pred_away."""
    # Predict class via argmax of softmax probabilities so we can emit
    # class-probabilities alongside the chosen class.
    probs = artifacts.clf.predict_proba(X)
    pred_idx = np.argmax(probs, axis=1)
    pred_result = [RESULT_LABELS[i] for i in pred_idx]
    pred_home = artifacts.reg_home.predict(X)
    pred_away = artifacts.reg_away.predict(X)
    return pd.DataFrame(
        {
            "pred_result": pred_result,
            "pred_home": np.round(np.clip(pred_home, 0, None)).astype(int),
            "pred_away": np.round(np.clip(pred_away, 0, None)).astype(int),
            "p_home_win": probs[:, LABEL_TO_IDX["home_win"]],
            "p_draw": probs[:, LABEL_TO_IDX["draw"]],
            "p_away_win": probs[:, LABEL_TO_IDX["away_win"]],
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_evaluate(regime: Regime, eval_year: int = 2022) -> pd.DataFrame:
    """Train on pre-eval_year matches, predict on eval_year, return a tidy DataFrame.

    Output schema matches the LLM prediction JSONs so downstream metrics code
    works uniformly.
    """
    contexts = load_match_contexts()
    df = build_dataset(contexts, regime)
    train, eval_ = temporal_split(df, eval_year)
    feature_cols = [
        c
        for c in df.columns
        if c not in {"fixture_id", "world_cup_year", "home_goals", "away_goals", "gt_result"}
    ]
    logger.info(
        "Training %s XGBoost on %d features, %d train / %d eval matches",
        regime,
        len(feature_cols),
        len(train),
        len(eval_),
    )
    artifacts = _fit_models(
        X=train[feature_cols],
        y_result=train["gt_result"],
        y_home=train["home_goals"].astype(float),
        y_away=train["away_goals"].astype(float),
    )
    preds = _predict(artifacts, eval_[feature_cols])

    out = pd.DataFrame(
        {
            "fixture_id": eval_["fixture_id"].values,
            "anonymized": False,  # XGBoost on named features only
            "gt_home": eval_["home_goals"].values,
            "gt_away": eval_["away_goals"].values,
            "gt_result": eval_["gt_result"].values,
            "pred_result": preds["pred_result"].values,
            "pred_home": preds["pred_home"].values,
            "pred_away": preds["pred_away"].values,
            "p_home_win": preds["p_home_win"].values,
            "p_draw": preds["p_draw"].values,
            "p_away_win": preds["p_away_win"].values,
        }
    )
    return out


def save_predictions(df: pd.DataFrame, regime: Regime, results_dir: Path | None = None) -> Path:
    out_dir = results_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"xgboost_predictions_{regime}.json"
    df.to_json(path, orient="records", indent=2)
    logger.info("Wrote %d predictions to %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summary(df: pd.DataFrame) -> None:
    from football_llm.eval.metrics import wilson_ci

    correct = (df["pred_result"] == df["gt_result"]).sum()
    n = len(df)
    exact = ((df["pred_home"] == df["gt_home"]) & (df["pred_away"] == df["gt_away"])).sum()
    mae = (
        (df["pred_home"] - df["gt_home"]).abs() + (df["pred_away"] - df["gt_away"]).abs()
    ).mean() / 2
    pred_ou = (df["pred_home"] + df["pred_away"]) > 2.5
    gt_ou = (df["gt_home"] + df["gt_away"]) > 2.5
    ou_correct = int((pred_ou == gt_ou).sum())
    print(f"  Result accuracy:  {wilson_ci(int(correct), int(n))}")
    print(f"  Score exact:      {wilson_ci(int(exact), int(n))}")
    print(f"  Goal MAE:         {mae:.3f}")
    print(f"  O/U 2.5 direct:   {wilson_ci(ou_correct, int(n))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train and evaluate, writing predictions to results/")
    p_train.add_argument("--regime", choices=["pregame", "halftime", "both"], default="both")

    p_predict = sub.add_parser("predict", help="Alias for train (predictions are part of training)")
    p_predict.add_argument("--regime", choices=["pregame", "halftime", "both"], default="both")

    args = parser.parse_args()

    regimes: list[Regime] = ["pregame", "halftime"] if args.regime == "both" else [args.regime]
    for regime in regimes:
        print(f"\n=== XGBoost baseline — {regime} ===")
        df = train_and_evaluate(regime)
        _summary(df)
        save_predictions(df, regime)
    return 0


if __name__ == "__main__":
    sys.exit(main())

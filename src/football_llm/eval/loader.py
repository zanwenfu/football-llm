"""Load prediction files and ground-truth into a canonical DataFrame.

The three prediction files in `results/` have overlapping-but-not-identical
schemas (halftime adds `halftime_*` fields; halftime+events adds
`first_half_events`). This module normalizes them into a single schema with
`regime` as a column so downstream code can work uniformly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from football_llm.eval.metrics import result_from_score
from football_llm.paths import RESULTS_DIR

REGIME_FILES: dict[str, str] = {
    "pregame": "ft_predictions_pregame.json",
    "halftime": "ft_predictions_halftime.json",
    "halftime_events": "ft_predictions_halftime_events.json",
}


def load_predictions(regime: str, results_dir: Path | str | None = None) -> pd.DataFrame:
    """Load a single regime's prediction file into a tidy DataFrame.

    Raises `FileNotFoundError` with a helpful hint if the file is missing.
    """
    if regime not in REGIME_FILES:
        raise ValueError(f"Unknown regime: {regime!r}. Valid: {list(REGIME_FILES)}")
    base = Path(results_dir) if results_dir else RESULTS_DIR
    path = base / REGIME_FILES[regime]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}\n"
            f"Regenerate via:\n"
            f"  python -m football_llm.eval.run_inference --regime {regime}"
        )
    with path.open() as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df["regime"] = regime
    # Canonical pred_result: always derive from score (§3.3).
    df["pred_result"] = [result_from_score(h, a) for h, a in zip(df["pred_home"], df["pred_away"])]
    # Canonical gt_result: re-derive to be safe even if source file disagrees.
    df["gt_result"] = [result_from_score(h, a) for h, a in zip(df["gt_home"], df["gt_away"])]
    df["correct_result"] = df["pred_result"] == df["gt_result"]
    df["correct_score"] = (df["pred_home"] == df["gt_home"]) & (df["pred_away"] == df["gt_away"])
    df["pred_total"] = df["pred_home"] + df["pred_away"]
    df["gt_total"] = df["gt_home"] + df["gt_away"]
    df["gt_over_25"] = df["gt_total"] > 2.5
    df["pred_over_25"] = df["pred_total"] > 2.5
    df["correct_ou_25"] = df["pred_over_25"] == df["gt_over_25"]
    return df


def load_all(
    regimes: Iterable[str] = ("pregame", "halftime", "halftime_events"),
    results_dir: Path | str | None = None,
    strict: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load every regime. Skips missing files unless `strict=True`."""
    out: dict[str, pd.DataFrame] = {}
    for regime in regimes:
        try:
            out[regime] = load_predictions(regime, results_dir)
        except FileNotFoundError:
            if strict:
                raise
    return out


def pair_on_match(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Align two regimes' predictions on (fixture_id, anonymized) for paired tests.

    Raises if either frame contains duplicate keys or if the two frames do not
    cover the same set of keys after alignment.
    """
    key_cols = ["fixture_id", "anonymized"]
    for name, df in (("a", df_a), ("b", df_b)):
        if df.duplicated(subset=key_cols).any():
            raise ValueError(f"df_{name} has duplicate (fixture_id, anonymized) keys")
    merged = df_a.merge(df_b, on=key_cols, suffixes=("_a", "_b"), how="inner")
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        raise ValueError(
            f"Inner-join dropped rows: df_a={len(df_a)}, df_b={len(df_b)}, merged={len(merged)}"
        )
    return merged


def result_accuracy_baselines(
    halftime: pd.DataFrame, results_dir: Path | str | None = None
) -> dict[str, tuple[int, int]]:
    """(correct, n) for the naive 1X2 baselines in the spring report's Figure 1.

    Computed from the prediction files rather than typed in: "Always home",
    "HT-leader" and "HT×2" on the 64 unique fixtures, and "Random" from the
    recorded seed-42 run in eval_results_legacy.json (n=128) when present.
    """
    unique = halftime[~halftime["anonymized"]]
    hh, ha = unique["halftime_home"], unique["halftime_away"]
    ht_leader = [result_from_score(h, a) for h, a in zip(hh, ha)]
    ht_double = [result_from_score(2 * h, 2 * a) for h, a in zip(hh, ha)]
    n = len(unique)
    out: dict[str, tuple[int, int]] = {}

    legacy = (Path(results_dir) if results_dir else RESULTS_DIR) / "eval_results_legacy.json"
    if legacy.exists():
        with legacy.open() as f:
            m = json.load(f)["metrics"]["baseline_random"]
        out["Random"] = (round(m["result_accuracy"] * m["total"]), m["total"])
    out["Always home"] = (int((unique["gt_result"] == "home_win").sum()), n)
    out["HT-leader"] = (
        int((pd.Series(ht_leader, index=unique.index) == unique["gt_result"]).sum()),
        n,
    )
    out["HT×2"] = (int((pd.Series(ht_double, index=unique.index) == unique["gt_result"]).sum()), n)
    return out

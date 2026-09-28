"""Leakage and sanity audit for the committed LLM predictions.

Llama 3.1's pretraining corpus runs into 2023, so the 2022 World Cup results
are already in the base model's weights. The named-vs-anonymized eval split
was built to catch this, but the original analysis only compared 1X2 accuracy
across the split (50.0% vs 53.1%) and concluded the model "learned from stats,
not names". This module runs the checks that analysis skipped:

1. **Named vs anonymized on every metric**, paired on the same fixture.
2. **Mirror-score test.** A model recalling a result it half-remembers tends
   to reproduce the scoreline digits but not always the orientation
   (Tunisia 1-0 France → "0-1"). Exact-or-mirrored hit rates are compared to a
   permutation null in which the model's predictions are matched to the wrong
   fixtures.
3. **Trivial baselines** that need no model at all (always under, HT×2, …).
4. **Output sanity** — draw rate, implausible scorelines, and text labels that
   contradict the predicted score.
5. **Power** — how many bets it takes before a claimed edge is distinguishable
   from luck.
6. **What survives** — the halftime lift, tested within each split so no
   match is counted twice.
7. **The flat-odds backtest** — the same simulation run on a rule with no
   model at all, to see what the simulation itself rewards.

All functions are pure and tested in tests/test_audit.py.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from football_llm.eval import metrics
from football_llm.eval.metrics import result_from_score
from football_llm.eval.poisson import p_over_25_vectorized

IMPLAUSIBLE_GOALS = 6  # one side scoring ≥6 — happened in 3 of the 64 matches in 2022


# ---------------------------------------------------------------------------
# 1. Named vs anonymized
# ---------------------------------------------------------------------------


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """1X2 / exact score / MAE / O/U 2.5 for the named and anonymized halves."""
    rows = []
    for anon, sub in df.groupby("anonymized"):
        rows.append(
            {
                "split": "anonymized" if anon else "named",
                "n": len(sub),
                "1X2": sub["correct_result"].mean(),
                "exact_score": sub["correct_score"].mean(),
                "goal_mae": metrics.goal_mae(
                    sub["pred_home"], sub["pred_away"], sub["gt_home"], sub["gt_away"]
                ),
                "ou_25": sub["correct_ou_25"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def paired_named_vs_anon(df: pd.DataFrame, col: str) -> metrics.McNemarResult:
    """Exact McNemar on a boolean column, named (A) vs anonymized (B), same fixtures."""
    named = df[~df["anonymized"]].set_index("fixture_id")[col]
    anon = df[df["anonymized"]].set_index("fixture_id")[col]
    named, anon = named.align(anon, join="inner")
    return metrics.mcnemar_exact(named.to_numpy(), anon.to_numpy())


# ---------------------------------------------------------------------------
# 2. Mirror-score test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MirrorRates:
    exact: float
    mirror_only: float  # predicted a-b when the result was b-a (a != b)

    @property
    def exact_or_mirror(self) -> float:
        return self.exact + self.mirror_only


def mirror_rates(pred_home, pred_away, gt_home, gt_away) -> MirrorRates:
    ph, pa = np.asarray(pred_home), np.asarray(pred_away)
    gh, ga = np.asarray(gt_home), np.asarray(gt_away)
    exact = (ph == gh) & (pa == ga)
    mirror = (ph == ga) & (pa == gh) & ~exact
    return MirrorRates(exact=float(exact.mean()), mirror_only=float(mirror.mean()))


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    null_mean: float
    null_p95: float
    null_max: float
    p_value: float  # one-sided, with the +1 correction
    n_permutations: int


def mirror_null_distribution(
    df: pd.DataFrame, n_permutations: int = 10_000, seed: int = 42
) -> np.ndarray:
    """Exact-or-mirror hit rates with the ground truth shuffled across fixtures.

    Shuffling keeps both the model's scoreline distribution and the tournament's
    scoreline distribution intact, so the null already credits the model for
    predicting common scores like 1-0 and 2-1. Only a match-specific link between
    prediction and result can beat it.
    """
    ph, pa = df["pred_home"].to_numpy(), df["pred_away"].to_numpy()
    gh, ga = df["gt_home"].to_numpy(), df["gt_away"].to_numpy()
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for k in range(n_permutations):
        perm = rng.permutation(len(df))
        null[k] = mirror_rates(ph, pa, gh[perm], ga[perm]).exact_or_mirror
    return null


def mirror_permutation_test(
    df: pd.DataFrame, n_permutations: int = 10_000, seed: int = 42
) -> PermutationResult:
    """Is the exact-or-mirror hit rate higher than predictions matched to random fixtures?"""
    observed = mirror_rates(
        df["pred_home"], df["pred_away"], df["gt_home"], df["gt_away"]
    ).exact_or_mirror
    null = mirror_null_distribution(df, n_permutations, seed)
    return PermutationResult(
        observed=observed,
        null_mean=float(null.mean()),
        null_p95=float(np.percentile(null, 95)),
        null_max=float(null.max()),
        p_value=float((1 + (null >= observed).sum()) / (1 + n_permutations)),
        n_permutations=n_permutations,
    )


def mirror_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the model predicted the actual scoreline with home/away swapped."""
    exact = (df["pred_home"] == df["gt_home"]) & (df["pred_away"] == df["gt_away"])
    mirror = (df["pred_home"] == df["gt_away"]) & (df["pred_away"] == df["gt_home"]) & ~exact
    out = df.loc[mirror, ["home_team", "away_team", "gt_home", "gt_away", "pred_home", "pred_away"]]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Trivial baselines
# ---------------------------------------------------------------------------


def trivial_baselines(matches: pd.DataFrame) -> pd.DataFrame:
    """Rules that need no model, scored on the same labels as the LLM.

    `matches` needs gt_home, gt_away and (for the halftime rules)
    halftime_home, halftime_away — one row per unique fixture.
    """
    gh, ga = matches["gt_home"].to_numpy(), matches["gt_away"].to_numpy()
    gt_result = np.array([result_from_score(h, a) for h, a in zip(gh, ga)])
    gt_over = (gh + ga) > 2.5

    rules: dict[str, tuple[np.ndarray | str, np.ndarray | bool]] = {
        "always home win / always under": ("home_win", False),
        "always home win / always over": ("home_win", True),
    }
    if {"halftime_home", "halftime_away"} <= set(matches.columns):
        hh, ha = matches["halftime_home"].to_numpy(), matches["halftime_away"].to_numpy()
        ht_leader = np.array([result_from_score(h, a) for h, a in zip(hh, ha)])
        rules["HT leader wins / HT×2 total"] = (ht_leader, (2 * (hh + ha)) > 2.5)

    rows = []
    for name, (pred_result, pred_over) in rules.items():
        rows.append(
            {
                "rule": name,
                "n": len(matches),
                "1X2": float(np.mean(pred_result == gt_result)),
                "ou_25": float(np.mean(pred_over == gt_over)),
            }
        )
    return pd.DataFrame(rows).set_index("rule")


# ---------------------------------------------------------------------------
# 4. Output sanity
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"prediction:\s*(home_win|away_win|draw)", re.IGNORECASE)


def text_label(raw_output: str) -> str | None:
    """The model's own 1X2 label, as written on the `Prediction:` line."""
    m = _LABEL_RE.search(raw_output or "")
    return m.group(1).lower() if m else None


def output_sanity(df: pd.DataFrame) -> dict[str, int]:
    """Counts that reveal decoding problems rather than modelling ones."""
    pred_draws = int((df["pred_home"] == df["pred_away"]).sum())
    gt_draws = int((df["gt_home"] == df["gt_away"]).sum())
    implausible = int(
        ((df["pred_home"] >= IMPLAUSIBLE_GOALS) | (df["pred_away"] >= IMPLAUSIBLE_GOALS)).sum()
    )
    gt_implausible = int(
        ((df["gt_home"] >= IMPLAUSIBLE_GOALS) | (df["gt_away"] >= IMPLAUSIBLE_GOALS)).sum()
    )
    contradictions = 0
    if "raw_output" in df.columns:
        for raw, h, a in zip(df["raw_output"], df["pred_home"], df["pred_away"]):
            label = text_label(raw)
            if label is not None and label != result_from_score(h, a):
                contradictions += 1
    return {
        "n": len(df),
        "predicted_draws": pred_draws,
        "actual_draws": gt_draws,
        "implausible_scorelines": implausible,
        "actual_implausible": gt_implausible,
        "label_contradicts_score": contradictions,
    }


# ---------------------------------------------------------------------------
# 5. Power
# ---------------------------------------------------------------------------


def bets_to_detect_edge(
    edge: float, decimal_odds: float = 1.90, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Bets needed for a one-sided test of mean return > 0 to detect `edge`.

    A flat-stake bet at decimal odds d that wins with probability p returns
    d - 1 or -1, so its standard deviation is d * sqrt(p (1 - p)), with p
    set so the expected return equals `edge`: p = (1 + edge) / d.
    """
    p = (1 + edge) / decimal_odds
    sd = decimal_odds * math.sqrt(p * (1 - p))
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return math.ceil((z * sd / edge) ** 2)


# ---------------------------------------------------------------------------
# 6. What survives — halftime lift within each split
# ---------------------------------------------------------------------------


def halftime_lift(
    pregame: pd.DataFrame, halftime: pd.DataFrame, col: str, anonymized: bool
) -> metrics.McNemarResult:
    """Paired pregame (A) vs halftime (B) on the same fixtures, one split at a time.

    Pooling named and anonymized rows (n=128) counts each match twice, which
    overstates the evidence; each split on its own is 64 independent matches.
    """
    a = pregame[pregame["anonymized"] == anonymized].set_index("fixture_id")[col]
    b = halftime[halftime["anonymized"] == anonymized].set_index("fixture_id")[col]
    a, b = a.align(b, join="inner")
    return metrics.mcnemar_exact(a.to_numpy(), b.to_numpy())


# ---------------------------------------------------------------------------
# 7. The flat-odds backtest
# ---------------------------------------------------------------------------


def ht_double_p_over(halftime_home, halftime_away) -> np.ndarray:
    """P(over 2.5) for the no-model rule "the second half repeats the first".

    Uses the same Poisson conversion the LLM backtest applies to its predicted
    totals, with λ = 2 × halftime goals, so the two are sized identically.
    """
    return p_over_25_vectorized(2 * (np.asarray(halftime_home) + np.asarray(halftime_away)))

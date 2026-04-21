"""Statistical primitives used throughout the paper.

All functions are pure, side-effect-free, and tested in tests/test_metrics.py.
The paper relies on four pieces of statistical machinery:

1. **Wilson score interval** for proportions — better than normal-approximation
   near 0 or 1 and at small n (§4.2 of paper).
2. **Exact McNemar test** for paired within-match comparisons — higher power
   than unpaired tests when the same matches are evaluated twice.
3. **Expected Calibration Error (ECE)** and **Brier score** for probability
   calibration (§5.9).
4. **Result parsing** — derive the 1X2 label from predicted score (§3.3:
   score overrides text label for consistency).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

Result = Literal["home_win", "draw", "away_win"]


# ---------------------------------------------------------------------------
# Proportion CIs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WilsonInterval:
    point: float
    low: float
    high: float
    n: int

    def __repr__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n})"


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> WilsonInterval:
    """Wilson score interval for a binomial proportion.

    Uses the formulation from Wilson (1927). Numerically stable for small n
    and proportions near 0 or 1, unlike the normal approximation.
    """
    if n == 0:
        return WilsonInterval(point=float("nan"), low=float("nan"), high=float("nan"), n=0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return WilsonInterval(point=p, low=max(0.0, center - half), high=min(1.0, center + half), n=n)


# ---------------------------------------------------------------------------
# Paired McNemar test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of a paired exact McNemar test.

    `b` = samples where model A correct and model B wrong.
    `c` = samples where model A wrong  and model B correct.
    If `b > c`, A is better.
    """

    b: int
    c: int
    n: int
    p_value: float

    @property
    def significant_05(self) -> bool:
        return self.p_value < 0.05


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> McNemarResult:
    """Paired exact McNemar test on two boolean arrays of correctness.

    Computes the exact two-sided p-value using the binomial distribution, which
    is correctly calibrated even at small discordant counts (unlike the
    chi-squared approximation, which requires b+c ≥ 25 for good calibration).
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(f"Length mismatch: {len(a_correct)} vs {len(b_correct)}")
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    b_count = int(np.sum(a & ~b))  # A correct, B wrong
    c_count = int(np.sum(~a & b))  # A wrong, B correct
    n = len(a)
    total_discordant = b_count + c_count

    if total_discordant == 0:
        p = 1.0
    else:
        # Two-sided exact binomial test on min(b, c) with p=0.5
        k = min(b_count, c_count)
        p = 2.0 * stats.binom.cdf(k, total_discordant, 0.5)
        p = min(p, 1.0)
    return McNemarResult(b=b_count, c=c_count, n=n, p_value=float(p))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probs: Sequence[float], outcomes: Sequence[bool], bins: int = 10
) -> float:
    """Expected Calibration Error with equal-width bins over [0, 1].

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|.

    Returns 0.0 for empty input to make the function total on its domain.
    """
    if len(probs) == 0:
        return 0.0
    probs_arr = np.asarray(probs, dtype=float)
    outcomes_arr = np.asarray(outcomes, dtype=float)
    if len(probs_arr) != len(outcomes_arr):
        raise ValueError("probs and outcomes must have same length")

    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(probs_arr)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (
            (probs_arr >= lo) & (probs_arr < hi)
            if i < bins - 1
            else (probs_arr >= lo) & (probs_arr <= hi)
        )
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(probs_arr[mask].mean())
        bin_acc = float(outcomes_arr[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
    return ece


def brier_score(probs: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Brier score — mean squared error between predicted probability and outcome."""
    probs_arr = np.asarray(probs, dtype=float)
    outcomes_arr = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs_arr - outcomes_arr) ** 2))


def reliability_bins(
    probs: Sequence[float], outcomes: Sequence[bool], bins: int = 5
) -> list[tuple[float, float, int]]:
    """Return `(bin_center_predicted, bin_empirical_freq, count)` tuples for plotting."""
    probs_arr = np.asarray(probs, dtype=float)
    outcomes_arr = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (
            (probs_arr >= lo) & (probs_arr < hi)
            if i < bins - 1
            else (probs_arr >= lo) & (probs_arr <= hi)
        )
        count = int(mask.sum())
        if count == 0:
            continue
        out.append((float(probs_arr[mask].mean()), float(outcomes_arr[mask].mean()), count))
    return out


# ---------------------------------------------------------------------------
# Result / score helpers
# ---------------------------------------------------------------------------


def result_from_score(home: int, away: int) -> Result:
    """Derive the 1X2 label from a scoreline.

    This is the canonical mapping used throughout the project — the model's text
    label is ignored in favor of its predicted score (§3.3 of paper).
    """
    if home > away:
        return "home_win"
    if away > home:
        return "away_win"
    return "draw"


def goal_mae(
    pred_home: Sequence[int],
    pred_away: Sequence[int],
    gt_home: Sequence[int],
    gt_away: Sequence[int],
) -> float:
    """Mean absolute error across home and away goals."""
    ph = np.asarray(pred_home, dtype=float)
    pa = np.asarray(pred_away, dtype=float)
    gh = np.asarray(gt_home, dtype=float)
    ga = np.asarray(gt_away, dtype=float)
    return float((np.abs(ph - gh) + np.abs(pa - ga)).mean() / 2)


def over_under_direction(total_goals: float, line: float = 2.5) -> Literal["over", "under"]:
    """Classify a total-goals value as over/under a line. Ties break to 'under'."""
    return "over" if total_goals > line else "under"

"""Score → O/U probability conversion via Poisson approximation.

The paper uses a simple Poisson model: given predicted final-score goals
ĝ_home + ĝ_away = λ̂, the probability of over 2.5 is

    P(over 2.5) = 1 - P(X ≤ 2)  where X ~ Poisson(λ̂)
                = 1 - e^-λ̂ (1 + λ̂ + λ̂²/2)

This is a conservative approximation — football scorelines are mildly
over-dispersed relative to Poisson (Dixon & Coles 1997). See §7 of the paper.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def p_over_25(lambda_total: float) -> float:
    """P(X > 2) where X ~ Poisson(λ)."""
    if lambda_total <= 0:
        return 0.0
    # P(X ≤ 2) = e^-λ (1 + λ + λ²/2)
    p_le_2 = math.exp(-lambda_total) * (1 + lambda_total + lambda_total * lambda_total / 2)
    return max(0.0, min(1.0, 1.0 - p_le_2))


def p_over_line(lambda_total: float, line: float = 2.5) -> float:
    """P(X > floor(line)) where X ~ Poisson(λ).

    Handles common half-integer lines (1.5, 2.5, 3.5, ...) where ties are
    impossible, as well as integer lines (2.0) where ties are resolved to the
    under by convention.
    """
    if lambda_total <= 0:
        return 0.0
    threshold = math.floor(line)
    # P(X ≤ threshold) = sum_{k=0..threshold} e^-λ λ^k / k!
    p_le = 0.0
    term = math.exp(-lambda_total)
    p_le += term
    for k in range(1, threshold + 1):
        term *= lambda_total / k
        p_le += term
    return max(0.0, min(1.0, 1.0 - p_le))


def p_over_25_vectorized(totals: Sequence[float]) -> np.ndarray:
    """Vectorized version for arrays of predicted totals."""
    arr = np.asarray(totals, dtype=float)
    arr_safe = np.where(arr > 0, arr, 1e-12)
    p_le_2 = np.exp(-arr_safe) * (1 + arr_safe + arr_safe * arr_safe / 2)
    return np.clip(1.0 - p_le_2, 0.0, 1.0)

"""Kelly-sized O/U 2.5 backtest with bootstrap CI.

Implements the simulation described in §6 of the paper:

- Bookmaker line: flat 1.90/1.90 on both sides of O/U 2.5 (implied 52.6% per
  side, vig 5.2%).
- Entry rule: bet the side with positive edge when |P̂ - P_implied| ≥
  edge_threshold (default 5%).
- Sizing: fractional Kelly with a per-match bankroll cap.
- Chronological ordering, compounding.

The bootstrap characterizes the conditional variance of compounded outcomes
given the observed per-bet return distribution — NOT a forecast of future live
returns. See §6.3 of the paper for the interpretive caveat.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from football_llm.eval.poisson import p_over_25_vectorized

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    initial_bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    per_match_cap: float = 0.10  # fraction of current bankroll
    edge_threshold: float = 0.05  # minimum |p̂ - p_implied| to place a bet
    over_odds: float = 1.90  # decimal odds on the over
    under_odds: float = 1.90  # decimal odds on the under


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    bankroll_trajectory: np.ndarray  # shape (n_bets + 1,), including $1000 start
    per_bet_returns: np.ndarray  # shape (n_bets,), fractional, can be negative
    final_bankroll: float
    roi: float
    win_rate: float
    max_drawdown: float
    sharpe: float  # per-bet Sharpe; multiply by sqrt(n) for scaled version
    num_bets: int
    bets_skipped: int = 0
    per_bet_log: list[dict] = field(default_factory=list)


def _kelly_fraction(p: float, decimal_odds: float) -> float:
    """Optimal Kelly fraction for a single binary bet.

    Given win probability `p` and decimal odds `d`, net odds `b = d - 1`,
    fraction = (b*p - (1-p)) / b. Clipped to [0, 1].
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (b * p - (1.0 - p)) / b
    return max(0.0, f)


def run_backtest(
    predicted_p_over: Sequence[float],
    actual_totals: Sequence[float],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a single chronological backtest pass.

    Args:
        predicted_p_over: Model's P(over 2.5) for each match, in chronological order.
        actual_totals: Observed final total goals for each match.
        config: Sizing and odds configuration.
    """
    cfg = config or BacktestConfig()
    p_over = np.asarray(predicted_p_over, dtype=float)
    totals = np.asarray(actual_totals, dtype=float)
    if len(p_over) != len(totals):
        raise ValueError(f"Length mismatch: {len(p_over)} vs {len(totals)}")

    implied_over = 1.0 / cfg.over_odds
    implied_under = 1.0 / cfg.under_odds

    bankroll = cfg.initial_bankroll
    trajectory = [bankroll]
    per_bet_returns = []
    wins = 0
    skipped = 0
    log = []

    for i, (p, total) in enumerate(zip(p_over, totals)):
        # Side selection: bet whichever side has positive edge above threshold
        over_edge = p - implied_over
        under_edge = (1.0 - p) - implied_under
        if over_edge >= under_edge:
            side, edge, bet_p, odds = "over", over_edge, p, cfg.over_odds
        else:
            side, edge, bet_p, odds = "under", under_edge, 1.0 - p, cfg.under_odds

        if edge < cfg.edge_threshold:
            skipped += 1
            trajectory.append(bankroll)
            continue

        kelly_f = _kelly_fraction(bet_p, odds)
        sized_f = cfg.kelly_fraction * kelly_f
        sized_f = min(sized_f, cfg.per_match_cap)
        stake = bankroll * sized_f
        if stake <= 0:
            skipped += 1
            trajectory.append(bankroll)
            continue

        actual_over = total > 2.5
        won = (side == "over" and actual_over) or (side == "under" and not actual_over)
        if won:
            pnl = stake * (odds - 1.0)
            wins += 1
        else:
            pnl = -stake

        bankroll += pnl
        trajectory.append(bankroll)
        per_bet_returns.append(pnl / (bankroll - pnl))  # return on prior bankroll
        log.append(
            {
                "idx": i,
                "side": side,
                "edge": edge,
                "stake_frac": sized_f,
                "stake": stake,
                "won": won,
                "pnl": pnl,
                "bankroll": bankroll,
            }
        )

    per_bet_arr = np.asarray(per_bet_returns, dtype=float)
    traj = np.asarray(trajectory, dtype=float)
    peak = np.maximum.accumulate(traj)
    # Avoid divide-by-zero — peak starts at $1000 which is > 0.
    dd = (traj - peak) / peak
    max_dd = float(dd.min())
    n_bets = len(per_bet_arr)
    sharpe = (
        float(per_bet_arr.mean() / per_bet_arr.std(ddof=1))
        if n_bets > 1 and per_bet_arr.std(ddof=1) > 0
        else 0.0
    )

    return BacktestResult(
        bankroll_trajectory=traj,
        per_bet_returns=per_bet_arr,
        final_bankroll=float(bankroll),
        roi=float((bankroll - cfg.initial_bankroll) / cfg.initial_bankroll),
        win_rate=float(wins / n_bets) if n_bets else 0.0,
        max_drawdown=max_dd,
        sharpe=sharpe,
        num_bets=n_bets,
        bets_skipped=skipped,
        per_bet_log=log,
    )


# ---------------------------------------------------------------------------
# Helpers: compute P(over) from predicted scores
# ---------------------------------------------------------------------------


def predictions_to_backtest_inputs(
    df: pd.DataFrame, anonymized: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (predicted_p_over, actual_totals) from a predictions DataFrame.

    Filters by `anonymized` flag and orders chronologically by `fixture_id`
    (the 2022 WC fixture IDs are monotonically increasing in match order).
    """
    sub = df[df["anonymized"] == anonymized].copy()
    sub = sub.sort_values("fixture_id").reset_index(drop=True)
    p_over = p_over_25_vectorized(sub["pred_total"].to_numpy())
    actual_totals = sub["gt_total"].to_numpy()
    return p_over, actual_totals


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    median_final: float
    percentile_5: float
    percentile_95: float
    p_profitable: float  # P(final > initial)
    n_trials: int
    all_finals: np.ndarray


def bootstrap_final_bankroll(
    per_bet_returns: Sequence[float],
    initial_bankroll: float = 1000.0,
    n_trials: int = 10_000,
    seed: int = 42,
) -> BootstrapResult:
    """Resample per-bet fractional returns with replacement and compound.

    Returns a CI on the terminal bankroll. Correctly interpreted as
    conditional variance given the observed per-bet return distribution, NOT
    a forecast of future live returns (§6.3 of the paper).
    """
    returns = np.asarray(per_bet_returns, dtype=float)
    if len(returns) == 0:
        raise ValueError("Cannot bootstrap zero bets")

    rng = np.random.default_rng(seed)
    n = len(returns)
    # Sample (n_trials, n) indices in one go for speed.
    idx = rng.integers(0, n, size=(n_trials, n))
    resampled = returns[idx]
    # Compound: final = initial * prod(1 + r_i)
    finals = initial_bankroll * np.prod(1.0 + resampled, axis=1)
    return BootstrapResult(
        median_final=float(np.median(finals)),
        percentile_5=float(np.percentile(finals, 5)),
        percentile_95=float(np.percentile(finals, 95)),
        p_profitable=float((finals > initial_bankroll).mean()),
        n_trials=n_trials,
        all_finals=finals,
    )


# ---------------------------------------------------------------------------
# Kelly × cap sensitivity grid
# ---------------------------------------------------------------------------


def sensitivity_grid(
    predicted_p_over: Sequence[float],
    actual_totals: Sequence[float],
    kelly_fractions: Sequence[float] = (0.10, 0.25, 0.50),
    per_match_caps: Sequence[float] = (0.05, 0.10, 0.20),
    base_config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run a grid of (kelly_fraction, per_match_cap) configurations.

    Returns a long-form DataFrame with columns:
        kelly_fraction, per_match_cap, roi, max_drawdown, win_rate, final_bankroll
    """
    base = base_config or BacktestConfig()
    rows = []
    for f in kelly_fractions:
        for cap in per_match_caps:
            cfg = BacktestConfig(
                initial_bankroll=base.initial_bankroll,
                kelly_fraction=f,
                per_match_cap=cap,
                edge_threshold=base.edge_threshold,
                over_odds=base.over_odds,
                under_odds=base.under_odds,
            )
            result = run_backtest(predicted_p_over, actual_totals, cfg)
            rows.append(
                {
                    "kelly_fraction": f,
                    "per_match_cap": cap,
                    "roi": result.roi,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "final_bankroll": result.final_bankroll,
                }
            )
    return pd.DataFrame(rows)

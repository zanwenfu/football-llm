"""Regenerate the paper's figures from prediction DataFrames.

All figures take the three-regime predictions and emit a `matplotlib.Figure`
so callers can either `fig.savefig(...)` or display interactively.

Figure 1: Result accuracy bar chart with Wilson CIs
Figure 2: Paired McNemar contingency + per-match MAE scatter
Figure 3: Named vs. anonymized accuracy + MAE bars
Figure 4: Calibration reliability diagram + ECE/Brier bars
Figure 5: Kelly-fraction × bet-cap sensitivity heatmaps
Figure 6: Kelly-sized bankroll trajectory
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from football_llm.eval.backtest import BacktestResult
from football_llm.eval.metrics import (
    brier_score,
    expected_calibration_error,
    goal_mae,
    reliability_bins,
    wilson_ci,
)
from football_llm.eval.poisson import p_over_25_vectorized


def _add_wilson_error_bars(
    ax, x: Sequence[float], successes: Sequence[int], totals: Sequence[int]
) -> None:
    """Compute and plot asymmetric Wilson error bars on `ax`."""
    lower_err = []
    upper_err = []
    points = []
    for s, n in zip(successes, totals):
        ci = wilson_ci(int(s), int(n))
        points.append(ci.point)
        lower_err.append(ci.point - ci.low)
        upper_err.append(ci.high - ci.point)
    ax.errorbar(x, points, yerr=[lower_err, upper_err], fmt="none", ecolor="black", capsize=3, lw=1)


# ---------------------------------------------------------------------------
# Figure 1: Result accuracy bar chart
# ---------------------------------------------------------------------------


def figure_result_accuracy(
    baselines: dict[str, tuple[int, int]],
    llm_pregame: tuple[int, int],
    llm_halftime: tuple[int, int],
) -> plt.Figure:
    """Bar chart of 1X2 result accuracy with Wilson CI error bars.

    `baselines` maps label → (successes, n). LLM entries are also (successes, n).
    """
    labels = [
        "Random\n(3-class)",
        "Always\nhome",
        "Pregame FT\nLLM",
        "HT-leader\nwins",
        "HT × 2\nextrapolation",
        "Empirical\nin-play prior",
        "Halftime FT\nLLM (ours)",
    ]

    # Pull values by label; fill 0/64 for missing entries gracefully.
    def _get(label: str, default=(0, 64)) -> tuple[int, int]:
        return baselines.get(label, default)

    entries: list[tuple[int, int]] = [
        _get("Random"),
        _get("Always home"),
        llm_pregame,
        _get("HT-leader"),
        _get("HT×2"),
        _get("Empirical prior"),
        llm_halftime,
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    values = [s / n for s, n in entries]
    x = np.arange(len(labels))
    colors = ["#d0d0d0"] * len(entries)
    colors[2] = "#3b7ddd"  # pregame LLM
    colors[-1] = "#c0392b"  # halftime LLM
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    _add_wilson_error_bars(ax, x, [s for s, _ in entries], [n for _, n in entries])

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Result accuracy")
    ax.set_title("Result accuracy on 2022 FIFA World Cup (error bars: 95% Wilson CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: McNemar contingency + MAE scatter
# ---------------------------------------------------------------------------


def figure_mcnemar_and_mae(
    pregame_correct: Sequence[bool],
    halftime_correct: Sequence[bool],
    pregame_mae_per_match: Sequence[float],
    halftime_mae_per_match: Sequence[float],
) -> plt.Figure:
    """2x1 figure: paired 2x2 contingency table (left) + per-match MAE scatter (right)."""
    pre = np.asarray(pregame_correct, dtype=bool)
    hft = np.asarray(halftime_correct, dtype=bool)
    both = int(np.sum(pre & hft))
    pre_only = int(np.sum(pre & ~hft))
    hft_only = int(np.sum(~pre & hft))
    neither = int(np.sum(~pre & ~hft))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: contingency heatmap
    matrix = np.array([[both, pre_only], [hft_only, neither]])
    axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() * 1.1)
    for i in range(2):
        for j in range(2):
            axes[0].text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="white" if matrix[i, j] > matrix.max() * 0.6 else "black",
            )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Halftime ✓", "Halftime ✗"])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Pregame ✓", "Pregame ✗"])
    axes[0].set_title(f"Paired contingency (n={len(pre)})")

    # Right: per-match MAE scatter
    axes[1].scatter(pregame_mae_per_match, halftime_mae_per_match, alpha=0.5, s=25)
    lim = (
        max(max(pregame_mae_per_match, default=0), max(halftime_mae_per_match, default=0)) * 1.1
        or 1
    )
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].set_xlabel("Pregame MAE per match")
    axes[1].set_ylabel("Halftime MAE per match")
    axes[1].set_title("Per-match score MAE (points below diagonal: halftime wins)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Calibration reliability diagram + ECE/Brier bars
# ---------------------------------------------------------------------------


def figure_calibration(
    regime_dfs: dict[str, pd.DataFrame],
    bins_reliability: int = 5,
) -> plt.Figure:
    """Reliability diagram + ECE/Brier bars across regimes.

    `regime_dfs` maps regime_name ("pregame" / "halftime" / "halftime_events")
    to a predictions DataFrame (from loader.load_predictions).
    """
    colors = {"pregame": "#3b7ddd", "halftime": "#f39c12", "halftime_events": "#27ae60"}
    labels = {"pregame": "Pregame", "halftime": "Halftime", "halftime_events": "Halftime+events"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: reliability diagram
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for name, df in regime_dfs.items():
        probs = p_over_25_vectorized(df["pred_total"].to_numpy())
        outcomes = df["gt_over_25"].to_numpy()
        bins = reliability_bins(probs, outcomes, bins=bins_reliability)
        if not bins:
            continue
        xs = [b[0] for b in bins]
        ys = [b[1] for b in bins]
        sizes = [b[2] * 10 for b in bins]
        axes[0].scatter(
            xs,
            ys,
            s=sizes,
            color=colors.get(name),
            label=labels.get(name, name),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
        axes[0].plot(xs, ys, color=colors.get(name), alpha=0.5, lw=1)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Predicted P(over 2.5)")
    axes[0].set_ylabel("Empirical frequency (over 2.5)")
    axes[0].set_title(f"Reliability diagram ({bins_reliability} bins)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Right: ECE + Brier bars
    regime_names = list(regime_dfs.keys())
    eces = []
    briers = []
    for name in regime_names:
        df = regime_dfs[name]
        probs = p_over_25_vectorized(df["pred_total"].to_numpy())
        outcomes = df["gt_over_25"].to_numpy()
        eces.append(expected_calibration_error(probs, outcomes, bins=10))
        briers.append(brier_score(probs, outcomes))

    x = np.arange(len(regime_names))
    width = 0.35
    b1 = axes[1].bar(
        x - width / 2, eces, width, label="ECE", color="#c0392b", edgecolor="black", lw=0.5
    )
    b2 = axes[1].bar(
        x + width / 2, briers, width, label="Brier", color="#7f8c8d", edgecolor="black", lw=0.5
    )
    for bar, v in list(zip(b1, eces)) + list(zip(b2, briers)):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels.get(r, r) for r in regime_names])
    axes[1].set_ylabel("Error (lower is better)")
    axes[1].set_title("Calibration error on O/U 2.5")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Kelly × cap sensitivity grid
# ---------------------------------------------------------------------------


def figure_sensitivity_grid(grid_df: pd.DataFrame) -> plt.Figure:
    """Two heatmaps (ROI + max drawdown) over (kelly_fraction × per_match_cap)."""
    roi_pivot = grid_df.pivot(index="kelly_fraction", columns="per_match_cap", values="roi")
    dd_pivot = grid_df.pivot(index="kelly_fraction", columns="per_match_cap", values="max_drawdown")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, pivot, title, fmt, cmap in [
        (axes[0], roi_pivot, "ROI sensitivity", "{:.0%}", "Greens"),
        (axes[1], dd_pivot, "Max drawdown sensitivity", "{:.1%}", "Reds_r"),
    ]:
        ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, fmt.format(pivot.iloc[i, j]), ha="center", va="center", fontsize=10)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.0%}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{i:.2f}" for i in pivot.index])
        ax.set_xlabel("Per-match bankroll cap")
        ax.set_ylabel("Kelly fraction")
        ax.set_title(title)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Bankroll trajectory
# ---------------------------------------------------------------------------


def figure_bankroll_trajectory(
    trajectories: dict[str, BacktestResult],
    initial_bankroll: float = 1000.0,
) -> plt.Figure:
    """Semi-log bankroll plot; one line per strategy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "halftime": "#c0392b",
        "pregame": "#3b7ddd",
        "halftime_events": "#27ae60",
        "xgboost": "#7f8c8d",
    }
    for name, result in trajectories.items():
        ax.plot(
            result.bankroll_trajectory,
            label=(f"{name} (final: ${result.final_bankroll:,.0f}; Sharpe {result.sharpe:.2f})"),
            color=colors.get(name),
            lw=1.5,
        )
    ax.axhline(
        initial_bankroll,
        color="black",
        lw=0.5,
        ls="--",
        alpha=0.5,
        label=f"Initial (${initial_bankroll:,.0f})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Match number (chronological)")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("O/U 2.5 Kelly-sized backtest")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: named-vs-anonymized bars
# ---------------------------------------------------------------------------


def figure_named_vs_anon(regime_dfs: dict[str, pd.DataFrame]) -> plt.Figure:
    """Bars comparing result accuracy, score EM, MAE across named vs. anonymized."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    regime_names = list(regime_dfs.keys())
    x = np.arange(len(regime_names) * 2)
    labels = []
    result_accs = []
    score_ems = []
    maes = []
    for name in regime_names:
        df = regime_dfs[name]
        for anon, suffix in [(False, "N"), (True, "A")]:
            sub = df[df["anonymized"] == anon]
            labels.append(f"{name}\n{suffix}")
            result_accs.append(sub["correct_result"].mean())
            score_ems.append(sub["correct_score"].mean())
            maes.append(
                goal_mae(sub["pred_home"], sub["pred_away"], sub["gt_home"], sub["gt_away"])
            )

    width = 0.35
    axes[0].bar(
        x - width / 2,
        result_accs,
        width,
        label="Result acc",
        color="#3b7ddd",
        edgecolor="black",
        lw=0.5,
    )
    axes[0].bar(
        x + width / 2,
        score_ems,
        width,
        label="Score EM",
        color="#c0392b",
        edgecolor="black",
        lw=0.5,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Proportion correct")
    axes[0].set_title("Accuracy by regime and anonymization")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, maes, color="#7f8c8d", edgecolor="black", lw=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Goal MAE")
    axes[1].set_title("Goal MAE by regime and anonymization")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig

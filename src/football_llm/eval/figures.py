"""Regenerate the paper's figures from prediction DataFrames.

All figures take the regime predictions and emit a `matplotlib.Figure`
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

    `baselines` maps label → (successes, n), e.g. from
    `loader.result_accuracy_baselines`. Labels not present are skipped rather
    than drawn as zero. LLM entries are also (successes, n).
    """
    pregame_rules = [("Random", "Random\n(3-class)"), ("Always home", "Always\nhome")]
    halftime_rules = [("HT-leader", "HT-leader\nwins"), ("HT×2", "HT × 2\nextrapolation")]
    labels: list[str] = []
    entries: list[tuple[int, int]] = []
    for key, label in pregame_rules:
        if key in baselines:
            labels.append(label)
            entries.append(baselines[key])
    labels.append("Pregame FT\nLLM")
    entries.append(llm_pregame)
    pregame_llm_idx = len(entries) - 1
    for key, label in halftime_rules:
        if key in baselines:
            labels.append(label)
            entries.append(baselines[key])
    labels.append("Halftime FT\nLLM")
    entries.append(llm_halftime)

    fig, ax = plt.subplots(figsize=(10, 5))
    values = [s / n for s, n in entries]
    x = np.arange(len(labels))
    colors = ["#d0d0d0"] * len(entries)
    colors[pregame_llm_idx] = "#3b7ddd"  # pregame LLM
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


# ---------------------------------------------------------------------------
# Audit figures (A2 report)
# ---------------------------------------------------------------------------
#
# Colour follows the entity across every audit figure: the named-prompt LLM is
# always orange, the anonymized LLM always blue, the no-model halftime rule
# always aqua, and every other baseline a recessive gray. Palette validated for
# colour-vision deficiency; aqua is below 3:1 on white, so all marks carry
# direct labels.

AUDIT_COLORS: dict[str, str] = {
    "named": "#eb6834",
    "anon": "#2a78d6",
    "rule": "#1baf7a",
    "baseline": "#c3c2b7",
}
_INK, _INK_2, _MUTED, _GRID, _AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

_AUDIT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.edgecolor": _AXIS,
    "axes.linewidth": 0.6,
    "axes.labelcolor": _INK_2,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlecolor": _INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "xtick.labelcolor": _INK_2,
    "ytick.labelcolor": _INK_2,
    "grid.color": _GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}


def figure_accuracy_by_model(
    panels: dict[str, list[tuple[str, int, int, str]]],
    xlabel: str,
    reference: tuple[float, str] | None = None,
) -> plt.Figure:
    """Horizontal bars with 95% Wilson whiskers, one panel per regime.

    `panels` maps panel title → rows of (label, successes, n, role), where role
    is a key of AUDIT_COLORS. `reference` draws one labelled vertical hairline,
    e.g. the break-even hit rate at the assumed odds.
    """
    with plt.rc_context(_AUDIT_RC):
        heights = [len(rows) for rows in panels.values()]
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(6.5, 0.24 * sum(heights) + 0.75 * len(panels)),
            sharex=True,
            gridspec_kw={"height_ratios": heights},
        )
        axes = np.atleast_1d(axes)
        for ax, (title, rows) in zip(axes, panels.items()):
            y = np.arange(len(rows))[::-1]
            for yi, (_label, s, n, role) in zip(y, rows):
                ci = wilson_ci(s, n)
                ax.barh(yi, ci.point, height=0.6, color=AUDIT_COLORS[role], zorder=2)
                ax.plot([ci.low, ci.high], [yi, yi], color=_INK_2, lw=0.8, zorder=3)
                ax.text(
                    ci.high + 0.012,
                    yi,
                    f"{ci.point:.1%}",
                    va="center",
                    color=_INK,
                    fontsize=8,
                    fontweight="bold" if role in ("named", "anon") else "normal",
                )
            ax.set_yticks(y)
            ax.set_yticklabels([r[0] for r in rows])
            ax.tick_params(axis="y", length=0)
            ax.set_title(title, loc="left")
            ax.grid(axis="x", zorder=0)
            ax.set_xlim(0, 1.0)
            if reference is not None:
                ax.axvline(reference[0], color=_INK_2, lw=0.8, zorder=4)
        if reference is not None:
            axes[0].annotate(
                reference[1],
                xy=(reference[0], 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, 2),
                textcoords="offset points",
                color=_INK_2,
                fontsize=7,
                va="bottom",
            )
        axes[-1].set_xlabel(xlabel)
        axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        fig.tight_layout()
    return fig


def figure_mirror_null(
    tests: dict[str, tuple[np.ndarray, float, float, str]], n_matches: int = 64
) -> plt.Figure:
    """Small multiples: each model's permutation null vs its observed hit rate.

    `tests` maps model label → (null distribution, observed rate, p-value, role).
    """
    with plt.rc_context(_AUDIT_RC):
        fig, axes = plt.subplots(len(tests), 1, figsize=(6.5, 0.95 * len(tests) + 0.5), sharex=True)
        axes = np.atleast_1d(axes)
        edges = (np.arange(0, n_matches + 2) - 0.5) / n_matches
        for ax, (label, (null, observed, p_value, role)) in zip(axes, tests.items()):
            weights = np.full(len(null), 1 / len(null))
            ax.hist(null, bins=edges, weights=weights, color=AUDIT_COLORS["baseline"], zorder=2)
            # A gray observed line would vanish into the gray null, so baselines use ink.
            line_color = _INK_2 if role == "baseline" else AUDIT_COLORS[role]
            ax.axvline(observed, color=line_color, lw=2, zorder=3)
            p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
            right_side = observed > 0.45
            ax.annotate(
                f"observed {observed:.1%}  ({p_text})",
                xy=(observed, 0.95),
                xycoords=("data", "axes fraction"),
                xytext=(-5 if right_side else 5, 0),
                textcoords="offset points",
                ha="right" if right_side else "left",
                va="top",
                color=_INK,
                fontsize=8,
                fontweight="bold",
            )
            ax.set_title(label, loc="left")
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x", zorder=0)
        axes[-1].set_xlim(0, 0.7)
        axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        axes[-1].set_xlabel(
            f"Share of {n_matches} matches predicted with the exact scoreline or its mirror"
        )
        fig.tight_layout()
    return fig


def figure_flat_odds_trajectories(runs: dict[str, tuple[BacktestResult, str]]) -> plt.Figure:
    """Bankroll paths under the flat 1.90/1.90 simulation, log scale, labelled at the end."""
    with plt.rc_context(_AUDIT_RC):
        fig, ax = plt.subplots(figsize=(6.5, 2.6))
        for label, (res, role) in runs.items():
            traj = res.bankroll_trajectory
            x = np.arange(len(traj))
            ax.plot(x, traj, color=AUDIT_COLORS[role], lw=1.6, zorder=3)
            ax.annotate(
                f"{label}  {res.roi:+.0%}",
                xy=(x[-1], traj[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=_INK,
                fontsize=8,
            )
        start = next(iter(runs.values()))[0].bankroll_trajectory[0]
        ax.axhline(start, color=_AXIS, lw=0.8)  # break-even
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.set_xlabel("Bets placed, in match order (2022 World Cup)")
        ax.set_ylabel("Bankroll")
        ax.grid(axis="y", which="major", zorder=0)
        ax.set_xlim(0, max(len(r.bankroll_trajectory) for r, _ in runs.values()) - 1)
        # End-of-line labels sit outside the axes; save with bbox_inches="tight".
        fig.subplots_adjust(left=0.1, right=0.7, bottom=0.17, top=0.96)
    return fig

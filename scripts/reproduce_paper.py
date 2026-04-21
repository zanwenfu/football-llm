#!/usr/bin/env python
"""Reproduce every headline number and figure in the paper.

Consumes committed prediction dumps under results/ and regenerates:

  Table 1 (Core logic), Table 2 (Three regimes), the paired McNemar tests,
  Wilson CIs, calibration (ECE/Brier), the Kelly-fraction x cap sensitivity
  grid, the 10,000-trial bootstrap, and all six paper figures.

Usage:
    python scripts/reproduce_paper.py                        # everything
    python scripts/reproduce_paper.py --output-dir figures/
    python scripts/reproduce_paper.py --skip-figures         # numbers only
    python scripts/reproduce_paper.py --skip-bootstrap       # faster (~0.5s)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from football_llm.eval import backtest, figures, loader, metrics, poisson
from football_llm.paths import FIGURES_DIR, RESULTS_DIR


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _fmt_ci(successes: int, n: int) -> str:
    ci = metrics.wilson_ci(successes, n)
    return f"{ci.point:.1%}  [{ci.low:.3f}, {ci.high:.3f}]  (n={ci.n})"


def _subset(df: pd.DataFrame, *, named_only: bool = False) -> pd.DataFrame:
    return df[~df["anonymized"]] if named_only else df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table_regime_summary(regime_dfs: dict[str, pd.DataFrame]) -> None:
    """Reproduce Table 2 (three LLM regimes on 128 eval samples)."""
    _section("Table 2 — Three LLM regimes on 128 eval samples")
    rows = []
    for name, df in regime_dfs.items():
        probs = poisson.p_over_25_vectorized(df["pred_total"].to_numpy())
        outcomes = df["gt_over_25"].to_numpy()
        rows.append(
            {
                "Regime": name,
                "N": len(df),
                "Result acc": df["correct_result"].mean(),
                "Score EM": df["correct_score"].mean(),
                "Goal MAE": metrics.goal_mae(
                    df["pred_home"], df["pred_away"], df["gt_home"], df["gt_away"]
                ),
                "O/U 2.5 dir": df["correct_ou_25"].mean(),
                "Brier": metrics.brier_score(probs, outcomes),
                "ECE": metrics.expected_calibration_error(probs, outcomes, bins=10),
            }
        )
    out = pd.DataFrame(rows).set_index("Regime")
    pd.options.display.float_format = "{:,.3f}".format
    print(out.to_string())


def xgboost_comparison(regime_dfs: dict[str, pd.DataFrame], results_dir: Path) -> None:
    """Reproduce Table 1 — LLM vs. XGBoost on 64 named 2022 WC matches."""
    import json

    _section("Table 1 — LLM vs. XGBoost (64 named 2022 WC matches)")
    rows = []
    for regime in ("pregame", "halftime"):
        if regime not in regime_dfs:
            continue
        # LLM: filter to named subset
        llm_named = regime_dfs[regime][~regime_dfs[regime]["anonymized"]]
        rows.append(
            {
                "Model": f"LLM {regime}",
                "Result acc": llm_named["correct_result"].mean(),
                "Score EM": llm_named["correct_score"].mean(),
                "Goal MAE": metrics.goal_mae(
                    llm_named["pred_home"],
                    llm_named["pred_away"],
                    llm_named["gt_home"],
                    llm_named["gt_away"],
                ),
                "O/U 2.5 dir": llm_named["correct_ou_25"].mean(),
            }
        )
        # XGBoost: look for predictions file
        xgb_path = results_dir / f"xgboost_predictions_{regime}.json"
        if xgb_path.exists():
            with xgb_path.open() as f:
                xgb_preds = pd.DataFrame(json.load(f))
            correct_r = (xgb_preds["pred_result"] == xgb_preds["gt_result"]).mean()
            correct_em = (
                (xgb_preds["pred_home"] == xgb_preds["gt_home"])
                & (xgb_preds["pred_away"] == xgb_preds["gt_away"])
            ).mean()
            mae = metrics.goal_mae(
                xgb_preds["pred_home"],
                xgb_preds["pred_away"],
                xgb_preds["gt_home"],
                xgb_preds["gt_away"],
            )
            pred_ou = (xgb_preds["pred_home"] + xgb_preds["pred_away"]) > 2.5
            gt_ou = (xgb_preds["gt_home"] + xgb_preds["gt_away"]) > 2.5
            rows.append(
                {
                    "Model": f"XGBoost {regime}",
                    "Result acc": correct_r,
                    "Score EM": correct_em,
                    "Goal MAE": mae,
                    "O/U 2.5 dir": (pred_ou == gt_ou).mean(),
                }
            )
        else:
            print(
                f"  (no xgboost_predictions_{regime}.json found — "
                f"run `python -m football_llm.baselines.xgboost train --regime {regime}`)"
            )
    if rows:
        out = pd.DataFrame(rows).set_index("Model")
        print(out.to_string(float_format="{:,.3f}".format))


def table_paired_mcnemar(regime_dfs: dict[str, pd.DataFrame]) -> None:
    """Reproduce §5.2 and §5.8 paired-test tables."""
    _section("Paired McNemar tests (exact, two-sided)")
    regimes = list(regime_dfs.keys())
    if "pregame" not in regimes or "halftime" not in regimes:
        print("  [skipped — need at least pregame+halftime]")
        return

    pairs = []
    if "pregame" in regimes and "halftime" in regimes:
        pairs.append(("pregame", "halftime"))
    if "pregame" in regimes and "halftime_events" in regimes:
        pairs.append(("pregame", "halftime_events"))
    if "halftime" in regimes and "halftime_events" in regimes:
        pairs.append(("halftime", "halftime_events"))

    def pair_test(df_a: pd.DataFrame, df_b: pd.DataFrame, metric_col: str) -> metrics.McNemarResult:
        merged = loader.pair_on_match(df_a, df_b)
        return metrics.mcnemar_exact(
            merged[f"{metric_col}_a"].to_numpy(), merged[f"{metric_col}_b"].to_numpy()
        )

    rows = []
    for a, b in pairs:
        df_a = regime_dfs[a]
        df_b = regime_dfs[b]
        for metric_name, col in [
            ("1X2 result", "correct_result"),
            ("Score EM", "correct_score"),
            ("O/U 2.5", "correct_ou_25"),
        ]:
            r = pair_test(df_a, df_b, col)
            rows.append(
                {
                    "A": a,
                    "B": b,
                    "Metric": metric_name,
                    "b (A✓ B✗)": r.b,
                    "c (A✗ B✓)": r.c,
                    "p-value": r.p_value,
                    "sig@0.05": "✓" if r.significant_05 else "—",
                }
            )
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))


def table_wilson_headlines(regime_dfs: dict[str, pd.DataFrame]) -> None:
    """Headline Wilson CIs cited in the abstract and §5."""
    _section("Wilson CIs for abstract / §5 headlines")
    for name, df in regime_dfs.items():
        for split_label, sub in [("all", df), ("named", _subset(df, named_only=True))]:
            print(
                f"  {name:<18} ({split_label:<5}) "
                f"1X2 acc: {_fmt_ci(int(sub['correct_result'].sum()), len(sub))}"
            )
            print(
                f"  {name:<18} ({split_label:<5}) "
                f"O/U 2.5:  {_fmt_ci(int(sub['correct_ou_25'].sum()), len(sub))}"
            )


# ---------------------------------------------------------------------------
# Backtest & bootstrap
# ---------------------------------------------------------------------------


def run_backtest_section(df_halftime: pd.DataFrame, do_bootstrap: bool) -> backtest.BacktestResult:
    """Reproduce §6.3 simulated P&L + 10k-trial bootstrap."""
    _section("§6.3 — Kelly-sized O/U 2.5 backtest (halftime FT on named 64)")
    p_over, totals = backtest.predictions_to_backtest_inputs(df_halftime, anonymized=False)
    result = backtest.run_backtest(p_over, totals)
    print(f"  Bets placed:     {result.num_bets} (skipped {result.bets_skipped})")
    print(f"  Win rate:        {result.win_rate:.1%}")
    print(f"  Final bankroll:  ${result.final_bankroll:,.0f}")
    print(f"  ROI:             {result.roi:+.1%}")
    print(f"  Max drawdown:    {result.max_drawdown:.1%}")
    print(f"  Match Sharpe:    {result.sharpe:.2f}")

    if do_bootstrap and result.num_bets > 0:
        _section("§6.3 — 10,000-trial bootstrap (conditional-variance CI)")
        boot = backtest.bootstrap_final_bankroll(result.per_bet_returns, n_trials=10_000)
        print(f"  Median final:    ${boot.median_final:,.0f}")
        print(f"  5th percentile:  ${boot.percentile_5:,.0f}")
        print(f"  95th percentile: ${boot.percentile_95:,.0f}")
        print(f"  P(profitable):   {boot.p_profitable:.1%}")
        print(f"  Trials:          {boot.n_trials:,}")
        print("  NB: This is a CI on conditional variance, NOT a forecast of live returns.")
    return result


def run_sensitivity_grid(df_halftime: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Figure 5 — Kelly × cap sensitivity."""
    _section("Figure 5 — Kelly-fraction × bet-cap sensitivity")
    p_over, totals = backtest.predictions_to_backtest_inputs(df_halftime, anonymized=False)
    grid = backtest.sensitivity_grid(p_over, totals)
    pivot_roi = grid.pivot(index="kelly_fraction", columns="per_match_cap", values="roi")
    pivot_dd = grid.pivot(index="kelly_fraction", columns="per_match_cap", values="max_drawdown")
    print("  ROI:")
    print(pivot_roi.to_string(float_format="{:,.0%}".format).replace("\n", "\n    "))
    print("  Max drawdown:")
    print(pivot_dd.to_string(float_format="{:.1%}".format).replace("\n", "\n    "))
    return grid


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(
    regime_dfs: dict[str, pd.DataFrame],
    grid: pd.DataFrame,
    bt_halftime: backtest.BacktestResult,
    output_dir: Path,
) -> None:
    _section(f"Figures → {output_dir}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: result accuracy with baselines (approximated from paper since
    # naive baselines aren't in the prediction files).
    if "pregame" in regime_dfs and "halftime" in regime_dfs:
        df_pre = regime_dfs["pregame"]
        df_hft = regime_dfs["halftime"]
        baselines = {
            "Random": (46, 128),  # 35.9% approx (from paper §5.1)
            "Always home": (58, 128),  # 45.3%
            "HT-leader": (36, 64),  # 56.2%
            "HT×2": (36, 64),  # 56.2%
            "Empirical prior": (35, 64),  # 54.7%
        }
        fig1 = figures.figure_result_accuracy(
            baselines,
            llm_pregame=(int(df_pre["correct_result"].sum()), len(df_pre)),
            llm_halftime=(int(df_hft["correct_result"].sum()), len(df_hft)),
        )
        fig1.savefig(output_dir / "fig1_result_accuracy.png", dpi=150)
        print("  fig1_result_accuracy.png")

    # Figure 2: McNemar contingency + MAE scatter
    if "pregame" in regime_dfs and "halftime" in regime_dfs:
        merged = loader.pair_on_match(regime_dfs["pregame"], regime_dfs["halftime"])
        # Per-match MAE on named 64 only
        named = merged[~merged["anonymized"]]
        pre_mae = (
            np.abs(named["pred_home_a"] - named["gt_home_a"]) / 2
            + np.abs(named["pred_away_a"] - named["gt_away_a"]) / 2
        )
        hft_mae = (
            np.abs(named["pred_home_b"] - named["gt_home_b"]) / 2
            + np.abs(named["pred_away_b"] - named["gt_away_b"]) / 2
        )
        fig2 = figures.figure_mcnemar_and_mae(
            merged["correct_result_a"].to_numpy(),
            merged["correct_result_b"].to_numpy(),
            pre_mae.to_numpy(),
            hft_mae.to_numpy(),
        )
        fig2.savefig(output_dir / "fig2_mcnemar_mae.png", dpi=150)
        print("  fig2_mcnemar_mae.png")

    # Figure 3: named vs. anon
    fig3 = figures.figure_named_vs_anon(regime_dfs)
    fig3.savefig(output_dir / "fig3_named_anon.png", dpi=150)
    print("  fig3_named_anon.png")

    # Figure 4: calibration
    fig4 = figures.figure_calibration(regime_dfs)
    fig4.savefig(output_dir / "fig4_calibration.png", dpi=150)
    print("  fig4_calibration.png")

    # Figure 5: sensitivity grid
    fig5 = figures.figure_sensitivity_grid(grid)
    fig5.savefig(output_dir / "fig5_sensitivity.png", dpi=150)
    print("  fig5_sensitivity.png")

    # Figure 6: bankroll trajectory
    fig6 = figures.figure_bankroll_trajectory({"halftime": bt_halftime})
    fig6.savefig(output_dir / "fig6_trajectory.png", dpi=150)
    print("  fig6_trajectory.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing prediction JSON files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=FIGURES_DIR, help="Where to write PNG figures"
    )
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip 10k-trial bootstrap")
    args = parser.parse_args()

    _section("Football-LLM — reproduce_paper.py")
    print(f"  Results dir:  {args.results}")
    print(f"  Output dir:   {args.output_dir}")

    regime_dfs = loader.load_all(results_dir=args.results, strict=False)
    if not regime_dfs:
        print(f"\nERROR: No prediction files found in {args.results}")
        print("Expected at least one of:")
        for f in loader.REGIME_FILES.values():
            print(f"  - {f}")
        return 1
    print(f"  Loaded regimes: {list(regime_dfs)}")

    table_regime_summary(regime_dfs)
    table_wilson_headlines(regime_dfs)
    table_paired_mcnemar(regime_dfs)
    xgboost_comparison(regime_dfs, args.results)

    bt_halftime = None
    grid = None
    if "halftime" in regime_dfs:
        bt_halftime = run_backtest_section(
            regime_dfs["halftime"], do_bootstrap=not args.skip_bootstrap
        )
        grid = run_sensitivity_grid(regime_dfs["halftime"])

    if not args.skip_figures and bt_halftime is not None and grid is not None:
        make_figures(regime_dfs, grid, bt_halftime, args.output_dir)

    _section("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

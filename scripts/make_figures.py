#!/usr/bin/env python
"""Regenerate just the paper's figures to figures/.

This is the figures-only entry point — faster than reproduce_paper.py when you
don't need the stats tables. Useful for doc builds, paper revisions, or
generating figures against a new prediction file during development.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --output-dir docs/figures/
    python scripts/make_figures.py --results path/to/alt_results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from football_llm.eval import backtest, figures, loader
from football_llm.paths import FIGURES_DIR, RESULTS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regime_dfs = loader.load_all(results_dir=args.results, strict=False)
    if not regime_dfs:
        print(f"ERROR: no prediction files in {args.results}")
        return 1
    print(f"Loaded regimes: {list(regime_dfs)}")

    # Figure 1 — result accuracy (requires at least pregame + halftime)
    if "pregame" in regime_dfs and "halftime" in regime_dfs:
        df_pre, df_hft = regime_dfs["pregame"], regime_dfs["halftime"]
        baselines = {
            "Random": (46, 128),
            "Always home": (58, 128),
            "HT-leader": (36, 64),
            "HT×2": (36, 64),
            "Empirical prior": (35, 64),
        }
        fig = figures.figure_result_accuracy(
            baselines,
            llm_pregame=(int(df_pre["correct_result"].sum()), len(df_pre)),
            llm_halftime=(int(df_hft["correct_result"].sum()), len(df_hft)),
        )
        fig.savefig(args.output_dir / "fig1_result_accuracy.png", dpi=args.dpi)
        print("  ✓ fig1_result_accuracy.png")

        # Figure 2 — McNemar + per-match MAE (named-only for MAE)
        merged = loader.pair_on_match(df_pre, df_hft)
        named = merged[~merged["anonymized"]]
        pre_mae = (
            (named["pred_home_a"] - named["gt_home_a"]).abs()
            + (named["pred_away_a"] - named["gt_away_a"]).abs()
        ) / 2
        hft_mae = (
            (named["pred_home_b"] - named["gt_home_b"]).abs()
            + (named["pred_away_b"] - named["gt_away_b"]).abs()
        ) / 2
        fig = figures.figure_mcnemar_and_mae(
            merged["correct_result_a"].to_numpy(),
            merged["correct_result_b"].to_numpy(),
            pre_mae.to_numpy(),
            hft_mae.to_numpy(),
        )
        fig.savefig(args.output_dir / "fig2_mcnemar_mae.png", dpi=args.dpi)
        print("  ✓ fig2_mcnemar_mae.png")

    # Figure 3 — named vs. anon
    fig = figures.figure_named_vs_anon(regime_dfs)
    fig.savefig(args.output_dir / "fig3_named_anon.png", dpi=args.dpi)
    print("  ✓ fig3_named_anon.png")

    # Figure 4 — calibration
    fig = figures.figure_calibration(regime_dfs)
    fig.savefig(args.output_dir / "fig4_calibration.png", dpi=args.dpi)
    print("  ✓ fig4_calibration.png")

    # Figure 5 + 6 — sensitivity grid + trajectory (requires halftime)
    if "halftime" in regime_dfs:
        p_over, totals = backtest.predictions_to_backtest_inputs(
            regime_dfs["halftime"], anonymized=False
        )
        grid = backtest.sensitivity_grid(p_over, totals)
        fig = figures.figure_sensitivity_grid(grid)
        fig.savefig(args.output_dir / "fig5_sensitivity.png", dpi=args.dpi)
        print("  ✓ fig5_sensitivity.png")

        result = backtest.run_backtest(p_over, totals)
        fig = figures.figure_bankroll_trajectory({"halftime": result})
        fig.savefig(args.output_dir / "fig6_trajectory.png", dpi=args.dpi)
        print("  ✓ fig6_trajectory.png")

    print(f"\nWrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

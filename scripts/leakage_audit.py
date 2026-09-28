#!/usr/bin/env python
"""Audit the committed LLM predictions for pretraining leakage and decoding bugs.

Every number in report/A2_report.pdf is printed by this script, and
`--figures-dir` regenerates its figures. It needs only the files already in
results/ (plus the Dixon-Coles predictions, which it builds if missing).

Usage:
    python scripts/leakage_audit.py
    python scripts/leakage_audit.py --figures-dir report/figures
    python scripts/leakage_audit.py --permutations 2000   # faster
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from football_llm.eval import audit, backtest, figures, loader, metrics
from football_llm.paths import RESULTS_DIR

pd.options.display.float_format = "{:,.3f}".format

BREAK_EVEN_190 = 1 / 1.90  # hit rate needed to break even at decimal odds 1.90


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _load_baseline(results_dir: Path, name: str) -> pd.DataFrame | None:
    path = results_dir / name
    if not path.exists():
        return None
    with path.open() as f:
        return pd.DataFrame(json.load(f))


def _dixon_coles(results_dir: Path) -> pd.DataFrame:
    df = _load_baseline(results_dir, "dixon_coles_predictions_pregame.json")
    if df is None:
        from football_llm.baselines import dixon_coles

        _, df = dixon_coles.train_and_evaluate()
        dixon_coles.save_predictions(df, results_dir)
    return df


def _correct(df: pd.DataFrame, pred_over: pd.Series | None = None) -> dict[str, pd.Series]:
    """Per-row correctness for any predictions frame in the shared schema."""
    gt_over = (df["gt_home"] + df["gt_away"]) > 2.5
    if pred_over is None:
        pred_over = (df["pred_home"] + df["pred_away"]) > 2.5
    return {
        "1X2": df["pred_result"] == df["gt_result"],
        "exact_score": (df["pred_home"] == df["gt_home"]) & (df["pred_away"] == df["gt_away"]),
        "ou_25": pred_over == gt_over,
    }


def _one_per_fixture(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["anonymized"]].sort_values("fixture_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def named_vs_anon(regimes: dict[str, pd.DataFrame]) -> None:
    _section("1. Named vs anonymized — same 64 fixtures, paired exact McNemar")
    for regime, df in regimes.items():
        print(f"\n  [{regime}]")
        print("  " + audit.split_summary(df).to_string().replace("\n", "\n  "))
        for label, col in [
            ("1X2", "correct_result"),
            ("exact score", "correct_score"),
            ("O/U 2.5", "correct_ou_25"),
        ]:
            r = audit.paired_named_vs_anon(df, col)
            print(
                f"    {label:<12} named-only correct: {r.b:>2}  anon-only correct: {r.c:>2}"
                f"  p = {r.p_value:.4f}"
            )


def mirror_test(
    regimes: dict[str, pd.DataFrame], results_dir: Path, n_permutations: int
) -> dict[str, tuple[np.ndarray, float, float, str]]:
    _section("2. Mirror-score test — exact or home/away-swapped scoreline (pregame)")
    pre = regimes["pregame"]
    candidates = {
        "LLM, team names": (pre[~pre["anonymized"]], "named"),
        "LLM, names hidden": (pre[pre["anonymized"]], "anon"),
        # Leak-free reference: same features, no path to the 2022 results.
        "XGBoost, same features (no access to 2022 results)": (
            _load_baseline(results_dir, "xgboost_predictions_pregame.json"),
            "baseline",
        ),
    }
    rows, for_figure = [], {}
    for name, (sub, role) in candidates.items():
        if sub is None:
            continue
        rates = audit.mirror_rates(
            sub["pred_home"], sub["pred_away"], sub["gt_home"], sub["gt_away"]
        )
        null = audit.mirror_null_distribution(sub, n_permutations=n_permutations)
        perm = audit.mirror_permutation_test(sub, n_permutations=n_permutations)
        for_figure[name] = (null, rates.exact_or_mirror, perm.p_value, role)
        rows.append(
            {
                "model": name.split(" (")[0],
                "exact": rates.exact,
                "mirror_only": rates.mirror_only,
                "exact_or_mirror": rates.exact_or_mirror,
                "null_mean": perm.null_mean,
                "null_p95": perm.null_p95,
                "p_value": perm.p_value,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"  null: each model's own predictions matched to shuffled fixtures ({n_permutations:,}x)."
    )
    print("  Halftime is excluded: the halftime score is legitimate match-specific input,")
    print("  so beating a shuffled-fixture null there is expected without any leakage.")

    print("\n  Pregame named — actual scoreline predicted with home/away swapped:")
    for c in audit.mirror_cases(pre[~pre["anonymized"]]).itertuples(index=False):
        print(
            f"    {c.home_team:>13} {c.gt_home}-{c.gt_away} {c.away_team:<13}"
            f"  predicted {c.pred_home}-{c.pred_away}"
        )
    return for_figure


def model_comparison(
    regimes: dict[str, pd.DataFrame], results_dir: Path
) -> dict[str, list[tuple[str, int, int, str]]]:
    """Every model on the same fixtures. Returns O/U rows for the accuracy figure."""
    _section("3. Every model on the same 64 fixtures (labels incl. extra time)")
    panels: dict[str, list[tuple[str, int, int, str]]] = {}
    for regime, df in regimes.items():
        # (table name, O/U figure label, per-row correctness, colour role)
        entries: list[tuple[str, str, dict[str, pd.Series], str]] = [
            ("LLM, named prompts", "LLM, team names", _correct(df[~df["anonymized"]]), "named"),
            (
                "LLM, anonymized prompts",
                "LLM, names hidden",
                _correct(df[df["anonymized"]]),
                "anon",
            ),
        ]
        xgb = _load_baseline(results_dir, f"xgboost_predictions_{regime}.json")
        if xgb is not None:
            entries.append(
                ("XGBoost, same features", "XGBoost, same features", _correct(xgb), "baseline")
            )
        if regime == "pregame":
            dc = _dixon_coles(results_dir)
            entries.append(
                (
                    "Dixon-Coles, WC 2010-18",
                    "Dixon-Coles, WC 2010-18",
                    _correct(dc, dc["p_over_25"] > 0.5),
                    "baseline",
                )
            )

        unique = _one_per_fixture(df)
        if regime == "halftime":
            hh, ha = unique["halftime_home"], unique["halftime_away"]
            ht_leader = pd.Series([metrics.result_from_score(h, a) for h, a in zip(hh, ha)])
            entries.append(
                (
                    "No model: HT leader wins, HT goals x2",
                    "No model: HT goals ×2",
                    {
                        "1X2": ht_leader == unique["gt_result"],
                        "exact_score": pd.Series([np.nan] * len(unique)),
                        "ou_25": ((2 * (hh + ha)) > 2.5) == unique["gt_over_25"],
                    },
                    "rule",
                )
            )
        entries.append(
            (
                "No model: always home win, always under",
                "No model: always under",
                {
                    "1X2": unique["gt_result"] == "home_win",
                    "exact_score": pd.Series([np.nan] * len(unique)),
                    "ou_25": ~unique["gt_over_25"],
                },
                "baseline",
            )
        )

        table = pd.DataFrame(
            [
                {"model": name, "n": len(c["1X2"]), **{k: v.mean() for k, v in c.items()}}
                for name, _, c, _ in entries
            ]
        ).set_index("model")
        print(f"\n  [{regime}]")
        print("  " + table.to_string().replace("\n", "\n  "))
        panels[regime.capitalize()] = [
            (label, int(c["ou_25"].sum()), len(c["ou_25"]), role) for _, label, c, role in entries
        ]
    return panels


def sanity(regimes: dict[str, pd.DataFrame], results_dir: Path) -> None:
    _section("4. Output sanity and run-to-run stability")
    rows = []
    for regime, df in regimes.items():
        for anon, sub in df.groupby("anonymized"):
            rows.append(
                {
                    "regime": regime,
                    "split": "anonymized" if anon else "named",
                    **audit.output_sanity(sub),
                }
            )
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"  implausible = one side predicted ≥{audit.IMPLAUSIBLE_GOALS} goals. "
        "Decoding used repetition_penalty=1.1 (notebooks/eval_harness.ipynb)."
    )

    legacy_path = results_dir / "eval_results_legacy.json"
    if legacy_path.exists() and "pregame" in regimes:
        with legacy_path.open() as f:
            legacy = json.load(f)["metrics"]["fine_tuned"]
        pre = regimes["pregame"]
        print("\n  Same weights, same 128 prompts, sampled at temperature 0.1, two runs:")
        print(
            f"    run A (eval_results_legacy.json):   1X2 {legacy['result_accuracy']:.1%}"
            f"  exact {legacy['score_exact_match']:.1%}  MAE {legacy['goal_mae']:.3f}"
        )
        print(
            f"    run B (ft_predictions_pregame.json): 1X2 {pre['correct_result'].mean():.1%}"
            f"  exact {pre['correct_score'].mean():.1%}"
            f"  MAE {metrics.goal_mae(pre['pred_home'], pre['pred_away'], pre['gt_home'], pre['gt_away']):.3f}"
        )


def what_survives(regimes: dict[str, pd.DataFrame]) -> None:
    _section("5. What survives — halftime lift, tested within each split")
    pre, ht = regimes["pregame"], regimes["halftime"]
    for anon in (False, True):
        split = "anonymized" if anon else "named"
        for label, col in [("1X2", "correct_result"), ("O/U 2.5", "correct_ou_25")]:
            a = pre[pre["anonymized"] == anon][col].mean()
            b = ht[ht["anonymized"] == anon][col].mean()
            r = audit.halftime_lift(pre, ht, col, anonymized=anon)
            print(
                f"    {split:<10} {label:<8} pregame {a:.1%} → halftime {b:.1%}"
                f"   (lost {r.b}, gained {r.c}; p = {r.p_value:.3f})"
            )
    print("  Pooling both splits (n=128) counts each match twice; the spring report did.")

    anon_ht = ht[ht["anonymized"]].sort_values("fixture_id").reset_index(drop=True)
    unique = _one_per_fixture(ht)
    rule_ok = ((2 * (unique["halftime_home"] + unique["halftime_away"])) > 2.5) == unique[
        "gt_over_25"
    ]
    r = metrics.mcnemar_exact(anon_ht["correct_ou_25"].to_numpy(), rule_ok.to_numpy())
    print(
        f"\n  O/U 2.5, anonymized halftime LLM {anon_ht['correct_ou_25'].mean():.1%} vs "
        f"no-model HT goals x2 {rule_ok.mean():.1%}  (LLM-only {r.b}, rule-only {r.c};"
        f" p = {r.p_value:.3f})"
    )


def flat_odds_backtest(regimes: dict[str, pd.DataFrame]) -> dict:
    _section("6. The flat-odds backtest — what does the simulation itself reward?")
    print("  Same simulation as the spring report: O/U 2.5 at 1.90/1.90 on every match,")
    print("  quarter Kelly, 10% cap, 5% edge threshold, $1,000 start, match order.")
    runs = {}
    for regime in ("pregame", "halftime"):
        for anon in (False, True):
            p, t = backtest.predictions_to_backtest_inputs(regimes[regime], anonymized=anon)
            runs[(regime, anon)] = backtest.run_backtest(p, t)
    unique = _one_per_fixture(regimes["halftime"])
    rule = backtest.run_backtest(
        audit.ht_double_p_over(unique["halftime_home"], unique["halftime_away"]),
        unique["gt_total"].to_numpy(),
    )
    rows = [
        ("LLM pregame, named", runs[("pregame", False)]),
        ("LLM pregame, anonymized", runs[("pregame", True)]),
        ("LLM halftime, named", runs[("halftime", False)]),
        ("LLM halftime, anonymized", runs[("halftime", True)]),
        ("No model: HT goals x2", rule),
    ]
    for name, res in rows:
        print(
            f"    {name:<26} bets {res.num_bets:>2}  win {res.win_rate:5.1%}"
            f"  ROI {res.roi:>+8.1%}  max drawdown {res.max_drawdown:6.1%}"
        )
    print("  A real book reprices O/U at halftime; flat 1.90 odds hand any HT-aware rule")
    print("  an edge the market would never offer.")
    return {
        "LLM halftime, team names": (runs[("halftime", False)], "named"),
        "No model: HT goals ×2": (rule, "rule"),
        "LLM halftime, names hidden": (runs[("halftime", True)], "anon"),
    }


def power() -> None:
    _section("7. How many bets before an edge is distinguishable from luck?")
    print("  One-sided test, α = 0.05, power = 0.8, flat stakes at decimal odds 1.90")
    for edge in (0.01, 0.02, 0.03, 0.05, 0.10):
        print(f"    true edge {edge:>4.0%} per bet  →  {audit.bets_to_detect_edge(edge):>7,} bets")
    print("  For scale: one World Cup is 64 matches (104 from 2026).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--figures-dir", type=Path, default=None, help="Write report figures")
    args = parser.parse_args()

    regimes = loader.load_all(("pregame", "halftime"), results_dir=args.results)
    if set(regimes) != {"pregame", "halftime"}:
        print(f"ERROR: need pregame and halftime prediction files in {args.results}")
        return 1

    named_vs_anon(regimes)
    mirror_fig = mirror_test(regimes, args.results, args.permutations)
    accuracy_fig = model_comparison(regimes, args.results)
    sanity(regimes, args.results)
    what_survives(regimes)
    backtest_fig = flat_odds_backtest(regimes)
    power()

    if args.figures_dir is not None:
        _section(f"Figures → {args.figures_dir}/")
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "fig_ou_accuracy.pdf": figures.figure_accuracy_by_model(
                accuracy_fig,
                xlabel="Over/under 2.5 goals: share of 64 matches called correctly",
                reference=(BREAK_EVEN_190, "break-even at odds 1.90 (52.6%)"),
            ),
            "fig_mirror_test.pdf": figures.figure_mirror_null(mirror_fig),
            "fig_flat_odds_backtest.pdf": figures.figure_flat_odds_trajectories(backtest_fig),
        }
        for name, fig in outputs.items():
            fig.savefig(args.figures_dir / name, bbox_inches="tight", pad_inches=0.04)
            print(f"  {name}")

    _section("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

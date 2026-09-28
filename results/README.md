# `results/` — Predictions and Evaluation Outputs

The audit trail for every number in the [report](../report/A2_report.pdf). Each file holds per-match predictions; [`scripts/leakage_audit.py`](../scripts/leakage_audit.py) and [`scripts/reproduce_paper.py`](../scripts/reproduce_paper.py) read them and recompute every table, test and figure.

## File inventory

| File | Model / regime | Rows | Produced by |
|:---|:---|:---:|:---|
| [`ft_predictions_pregame.json`](ft_predictions_pregame.json) | Fine-tuned LLM, pregame | 128 | [`notebooks/eval_harness.ipynb`](../notebooks/eval_harness.ipynb) (Colab T4) |
| [`ft_predictions_halftime.json`](ft_predictions_halftime.json) | Fine-tuned LLM, halftime score in prompt | 128 | [`notebooks/eval_harness.ipynb`](../notebooks/eval_harness.ipynb) (Colab T4) |
| [`xgboost_predictions_pregame.json`](xgboost_predictions_pregame.json) | XGBoost on the same features | 64 | `python -m football_llm.baselines.xgboost train --regime pregame` |
| [`xgboost_predictions_halftime.json`](xgboost_predictions_halftime.json) | XGBoost plus halftime features | 64 | `python -m football_llm.baselines.xgboost train --regime halftime` |
| [`dixon_coles_predictions_pregame.json`](dixon_coles_predictions_pregame.json) | Dixon-Coles fitted on WC 2010–2018 | 64 | `python -m football_llm.baselines.dixon_coles` |
| [`eval_results_legacy.json`](eval_results_legacy.json) | Aggregate metrics from an earlier sampling run | — | Kept to show run-to-run variation (audit §4) |

**LLM files have 128 rows** = 64 matches of the 2022 World Cup × 2 prompt variants: real team names (`anonymized: false`) and `Team A` / `Team B` (`anonymized: true`). Baseline files have one row per match.

The spring report also described a "halftime + first-half events" regime. Its predictions came from a scratch pipeline and were never saved, so no numbers from it are reported.

## Schema — LLM predictions

```json
{
  "fixture_id": 855736,            // API-Football fixture ID
  "home_team": "Qatar",            // Real name, even when anonymized=true
  "away_team": "Ecuador",
  "anonymized": false,             // false = names in prompt, true = Team A / Team B
  "gt_home": 0,                    // Final score incl. extra time, home
  "gt_away": 2,
  "gt_result": "away_win",
  "pred_result": "home_win",       // Derived from the predicted score
  "pred_home": 2,                  // Parsed predicted score
  "pred_away": 0,
  "raw_output": "Prediction: ...\nScore: 2-0\nReasoning: ..."
}
```

Halftime files add `halftime_home` and `halftime_away`, the observed halftime score that was put in the prompt.

## Schema — baseline predictions

XGBoost and Dixon-Coles share the LLM fields (without `raw_output`) plus 1X2 probabilities `p_home_win`, `p_draw`, `p_away_win`. Dixon-Coles also has its expected goals `lambda_home`, `lambda_away` and `p_over_25`, the probability of three or more goals.

## Label conventions

- `*_result` ∈ `{"home_win", "draw", "away_win"}`.
- `pred_result` is always derived from `pred_home` vs `pred_away`, not from the model's text label. The two disagree in 24 of 64 pregame outputs with names (audit §4).
- `gt_*` is the score **after extra time** (Argentina–France is 3–3). Betting markets settle at 90 minutes; this affects 2 of the 64 matches.
- The `Team A` / `Team B` prompts still contain the coach's name, stadium and round, so they are only partly anonymized.

## Reproducing

```bash
pip install -e ".[dev]"
python scripts/leakage_audit.py --figures-dir report/figures   # the audit
python scripts/reproduce_paper.py --output-dir figures/        # the spring tables and figures
```

Both scripts are deterministic (NumPy seed 42) and run in seconds.

## Not in this directory

- **Raw match data:** [`data/raw/`](../data/raw/) (see [`docs/DATA_CARD.md`](../docs/DATA_CARD.md)).
- **Training data:** [`data/training/`](../data/training/).
- **Model weights:** on Hugging Face at [`zanwenfu/football-llm-qlora`](https://huggingface.co/zanwenfu/football-llm-qlora).

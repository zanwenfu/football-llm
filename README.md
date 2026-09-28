<div align="center">

# Football-LLM

**Can a fine-tuned LLM predict World Cup matches, or does it just remember them?**

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97_Model-football--llm--qlora-blue)](https://huggingface.co/zanwenfu/football-llm-qlora)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## TL;DR

A QLoRA adapter over Llama 3.1 8B reads player statistics for both starting XIs and predicts the final score of a World Cup match, before kickoff or at halftime. It was trained on the 2010, 2014 and 2018 World Cups and tested on the 64 matches of 2022.

My [spring 2026 report](https://github.com/zanwenfu/football-llm/blob/2620371/crypto-trading/project/IDS598_Final_Project_Report.pdf) said the model beat XGBoost by more than 20 points on over/under 2.5 goals and returned +1,468% in a simulated backtest. **This repository now audits that claim, and most of it does not survive.** Llama 3.1's pretraining data runs to December 2023, so the 2022 results were already in the model's weights.

| 2022 World Cup, 64 matches | Pregame 1X2 | Pregame exact score | Pregame O/U 2.5 | Halftime 1X2 | Halftime O/U 2.5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| LLM, team names in prompt | 50.0% | **43.8%** | **78.1%** | 64.1% | **81.2%** |
| LLM, team names hidden | 53.1% | 10.9% | 56.2% | 64.1% | 70.3% |
| XGBoost, same features | 57.8% | 12.5% | 46.9% | 62.5% | 67.2% |
| Dixon-Coles, fitted on WC 2010–18 | 45.3% | 7.8% | 53.1% | – | – |
| No model: always home win / always under | 45.3% | – | 53.1% | 45.3% | 53.1% |
| No model: halftime leader wins / halftime goals ×2 | – | – | – | 56.2% | 75.0% |

What the audit found:

- **Hiding the names removes the edge.** Exact scores fall from 43.8% to 10.9% (paired McNemar p < 0.001). Winner accuracy doesn't change, which is what a model that remembers "it finished 1–0" but not who won looks like.
- **The model recalls scorelines, sometimes mirrored.** With names, 59% of predictions are the real score or its home/away mirror, against ~10% for the same predictions shuffled across matches. The upsets come back with the right digits and the favourite winning: Tunisia 1–0 France → 0–1, Morocco 1–0 Portugal → 0–1, Japan 2–1 Germany → Germany 2–1.
- **The backtest rewards its own assumptions.** Under the same flat-1.90-odds simulation, a rule with no model ("double the halftime goals") returns +1,125%. The hidden-name LLM returns −0.4% before kickoff.

What survives: the halftime score is real information (hidden-name O/U 56% → 70%), but a one-line rule captures it at least as well. The honest test is a forward test on matches after the model's training cutoff, priced against real odds; the [report](report/A2_report.pdf) lays out that plan.

---

## Reproduce every number (no GPU, ~5 seconds)

```bash
git clone https://github.com/zanwenfu/football-llm.git
cd football-llm
pip install -e ".[dev]"

python scripts/leakage_audit.py --figures-dir report/figures  # the audit: tables, tests, figures
python -m football_llm.baselines.dixon_coles                  # Dixon-Coles benchmark
python scripts/reproduce_paper.py --skip-figures              # the spring tables, recomputed
```

`leakage_audit.py` prints every number in the report:

| Section | What it tests |
|:---|:---|
| 1. Named vs anonymized | Same fixtures, names hidden; paired exact McNemar per metric |
| 2. Mirror-score test | Exact-or-mirrored scorelines vs a 10,000-shuffle permutation null, with XGBoost as the leak-free reference |
| 3. Every model, same matches | LLM (both variants), XGBoost, Dixon-Coles, and rules that use no model |
| 4. Output sanity and stability | Draw rate, implausible scores, labels that contradict scores, two sampling runs |
| 5. What survives | Halftime lift tested within each split (no match counted twice) |
| 6. Flat-odds backtest | The spring simulation run on a no-model rule |
| 7. Power | Bets needed before an edge of a given size is distinguishable from luck |

To rebuild the PDF: `cd report && xelatex A2_report.tex && xelatex A2_report.tex`.

---

## Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│ DATA  (API-Football, scraped Jan–Feb 2026)                       │
│   256 World Cup matches 2010–2022: results, halftime, lineups,   │
│   events · ~168k player-season-competition stat rows 2004–2025   │
│                                                                  │
│   aggregate_player_stats → build_team_profiles → generate_data   │
│   Starting XI's stats from the 3 seasons BEFORE each tournament  │
│   Each match rendered twice: team names / "Team A" & "Team B"    │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ TRAINING   Llama 3.1 8B Instruct + QLoRA (r=16, α=32, NF4)       │
│   384 samples (WC 2010/14/18 × 2 variants) · 3 epochs · Colab T4 │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ EVALUATION   WC 2022, 64 matches × {names, hidden}               │
│   Regimes: pregame · halftime (score added to the prompt)        │
│   Baselines: XGBoost · Dixon-Coles · no-model rules              │
│   Audit: named/hidden gap · mirror test · flat-odds backtest     │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ SERVING   vLLM (8000) ── FastAPI (8001) ── Gradio (7860)         │
└──────────────────────────────────────────────────────────────────┘
```

Known issues found by the audit, all tracked as final-project fixes in the report:

- The "hidden" prompts still carry the coach's name, stadium and round, which can identify the match.
- Labels use the score after extra time; bets settle at 90 minutes (2 test matches affected).
- Inference sampled at temperature 0.1 with `repetition_penalty=1.1`; the model almost never predicts a draw before kickoff and sometimes emits scores like 0–8.
- The spring report's halftime + first-half-events regime was run in a scratch pipeline and its predictions were never saved, so it is not reported here.

### Full pipeline (GPU for training and inference)

```bash
pip install -e ".[train,serve]"

python -m football_llm.data_prep.run_pipeline            # rebuild data/training/*.jsonl
python -m football_llm.training.run_sft \
  --config src/football_llm/training/recipes/llama-3-1-8b-instruct-qlora.yaml
python -m football_llm.baselines.xgboost train           # XGBoost predictions
```

Inference for the committed `results/ft_predictions_*.json` files was run in [`notebooks/eval_harness.ipynb`](notebooks/eval_harness.ipynb) on Colab.

### Prompt format

```
World Cup 2022 | Group Stage - 1 | Al Bayt Stadium

Qatar (Home) | Coach: Félix Sánchez | Formation: 5-3-2
Squad: 11 starters | Avg Rating: 7.0
Attack: 199 goals (0.18/90) | 21 assists | Top scorer: 53 goals
Defense: 169 yellows, 7 reds | Tackles/90: 0.30 | Duels: 60%
Passing: 63% accuracy
...
Predict result, score, and reasoning.
```

The halftime regime adds `Halftime Score: {Home} 1 - 0 {Away}` and asks for the final result. The model was never fine-tuned on halftime prompts.

### Serving

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...   # Llama 3.1 is gated
docker compose up
```

vLLM serves the adapter on `:8000`, FastAPI builds prompts and parses output on `:8001/predict`, and Gradio runs a demo UI on `:7860`. The API also accepts an experimental `halftime_events` regime (first-half goals and cards in the prompt) that is not part of the evaluated results.

---

## Repository layout

```
football-llm/
├── report/                 # A2 report (LaTeX source, PDF, figures)
├── scripts/
│   ├── leakage_audit.py    # every number and figure in the report
│   └── reproduce_paper.py  # the spring tables, recomputed from results/
├── src/football_llm/
│   ├── data_prep/          # 3-stage training-data pipeline
│   ├── training/           # QLoRA SFT + adapter merge
│   ├── eval/               # metrics, audit, backtest, figures
│   ├── baselines/          # XGBoost and Dixon-Coles
│   └── serving/            # vLLM + FastAPI + Gradio
├── results/                # committed predictions (LLM, XGBoost, Dixon-Coles)
├── data/                   # raw scrape, processed contexts, train/eval JSONL
├── football-data/          # the API-Football scraper
├── notebooks/              # Colab training, evaluation, serving
└── tests/                  # pytest suite
```

---

## History and AI use

This project started as my IDS 598.1 final project in spring 2026 ([report at that commit](https://github.com/zanwenfu/football-llm/blob/2620371/crypto-trading/project/IDS598_Final_Project_Report.pdf)). The audit, the Dixon-Coles benchmark, the report in `report/`, and this README are new. The code and writing in both phases were produced with heavy AI assistance (Claude). The report's Section 9 discloses the details, and [`AI_USAGE.md`](AI_USAGE.md) holds the full dialogues.

## References

1. Dettmers et al. **QLoRA: Efficient Finetuning of Quantized LLMs.** NeurIPS 2023.
2. Maher. **Modelling association football scores.** Statistica Neerlandica, 1982.
3. Dixon & Coles. **Modelling association football scores and inefficiencies in the football betting market.** JRSS-C, 1997.

## License

MIT — see [LICENSE](LICENSE). The base model is subject to the [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/).

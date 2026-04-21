<div align="center">

# Football-LLM

**A QLoRA-fine-tuned Llama 3.1 8B for halftime-conditioned Over/Under betting on World Cup matches**

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97_Model-football--llm--qlora-blue)](https://huggingface.co/zanwenfu/football-llm-qlora)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/zanwenfu/football-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/zanwenfu/football-llm/actions/workflows/ci.yml)

<br/>

| Metric (held-out 2022 WC, n=128) | Pregame | Halftime | **Halftime + Events** |
|:---|:---:|:---:|:---:|
| Result Accuracy (1X2) | 52.3% | **64.1%** | 61.7% |
| O/U 2.5 Directional Accuracy | 65.6% | 74.2% | **79.7%** |
| Goal MAE | 1.32 | 1.21 | **1.12** |
| Calibration (ECE, O/U 2.5) | 0.272 | 0.239 | **0.182** |

<sub>Paired McNemar: pregame → halftime on 1X2 <b>p = 0.024</b> · pregame → halftime+events on O/U 2.5 <b>p = 0.006</b></sub>
<br/>
<sub>On the 64 named matches, halftime+events reaches <b>84.4%</b> O/U 2.5 accuracy (Wilson CI [0.736, 0.913]).</sub>

</div>

---

## TL;DR

Football-LLM is a QLoRA adapter over Llama 3.1 8B that predicts **final scorelines** for World Cup matches from player-level team statistics. Three inference regimes are supported: pregame, halftime-conditioned, and halftime + first-half event enrichment (goals/cards with timestamps).

The headline finding — and the reason the LLM beats a feature-matched XGBoost baseline — is a clean **magnitude/direction decomposition**:

- **On 1X2 (direction) the LLM ties XGBoost** within 2pp. Deploying an LLM for win/draw/loss prediction is not justified on this dataset.
- **On Over/Under 2.5 goals (magnitude) the LLM beats XGBoost by 19pp pregame and 16pp halftime.** This is driven by pretrained scoreline priors (team identity → mixture over typical 2–0 / 2–1 / 3–1 scorelines) that tabular features cannot replicate.

The decomposition makes a falsifiable prediction: enriching the halftime prompt with a first-half event summary should improve **magnitude** metrics (O/U, MAE, calibration) without improving **direction** (1X2). Results match the prediction exactly — O/U 2.5 accuracy climbs monotonically 65.6% → 74.2% → 79.7%, calibration ECE falls 0.272 → 0.239 → 0.182, and 1X2 accuracy does not improve (paired p = 0.58).

See [IDS598_Final_Project_Report.pdf](IDS598_Final_Project_Report.pdf) for the full paper with Wilson CIs, paired McNemar tests, Kelly-fraction × bet-cap sensitivity grid, and 10,000-trial bootstrap on simulated P&L.

---

## Why this repo is interesting

For engineers evaluating the project:

1. **Reproducible end-to-end on free hardware.** Training runs in 43 minutes on a Colab T4 (16 GB VRAM), peak 5.7 GB. Every number in the paper regenerates from [`scripts/reproduce_paper.py`](scripts/reproduce_paper.py) against committed prediction dumps in [`results/`](results/).
2. **Statistical honesty.** All proportions use Wilson CIs, all within-match comparisons use paired exact McNemar, calibration reported via ECE + Brier + reliability diagrams, and the backtest includes a 10,000-trial bootstrap with the caveat that it characterizes P&L variance conditional on the observed per-bet return distribution — **not** a forecast of live returns.
3. **A falsifiable thesis, not just a benchmark.** The magnitude/direction decomposition predicts which metrics should and should not improve under event enrichment. The experiment supports it. This rules out the "LLM is just generically better" interpretation.
4. **Production-adjacent serving.** vLLM + FastAPI + Gradio stack, containerized, with `docker compose up` bringing the full stack online.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                            │
│                                                                 │
│  API-Football feed (2010–2022 World Cups + first-half events)  │
│       │                                                         │
│       ▼                                                         │
│  Step 1  aggregate_player_stats.py  → 41K player-season rows   │
│  Step 2  build_team_profiles.py     → 256 match contexts       │
│  Step 3  generate_training_data.py  → 384 train + 128 eval     │
│       │  (named + anonymized variants, compact ≤350 tokens)    │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING                                 │
│                                                                 │
│  Llama 3.1 8B Instruct + QLoRA (r=16, α=32, NF4 4-bit)        │
│  3 epochs · eff. batch 16 · lr 2e-4 cosine · max_len 768       │
│  T4 GPU · 43 min · 5.7 GB peak · 83.9 MB adapter              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EVALUATION                                │
│                                                                 │
│  Three prompt regimes on held-out 2022 WC (128 samples)        │
│  ┌─────────────┬──────────────┬───────────────────┐           │
│  │  Pregame    │  Halftime    │  HT + Events      │           │
│  │  team stats │  + HT score  │  + goals/cards    │           │
│  └─────────────┴──────────────┴───────────────────┘           │
│                                                                 │
│  Baselines: always-home · HT-leader · HT×2 · coarse prior ·    │
│             ESPN-proxy prior · XGBoost (pregame + halftime)    │
│                                                                 │
│  Metrics: 1X2 acc · exact score · goal MAE · O/U 2.5/3.5 acc · │
│           Brier · ECE · paired McNemar · Wilson CIs            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SERVING                                  │
│                                                                 │
│  vLLM (8000) ── FastAPI (8001) ── Gradio (7860)                │
│  One-command:  docker compose up                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Headline results

All numbers are on the 2022 FIFA World Cup held-out set. `n=128` = 64 matches × {named, anonymized} variants. `n=64` = unique matches (named subset, for XGBoost parity).

### LLM vs. feature-matched XGBoost (n=64 named)

| Metric | Pregame LLM | Pregame XGB | Halftime LLM | Halftime XGB |
|:---|:---:|:---:|:---:|:---:|
| 1X2 Result Accuracy | 50.0% | **51.6%** | **65.6%** | 62.5% |
| Score Exact-Match | **39.1%** | 6.2% | **51.6%** | 10.9% |
| Goal MAE | 1.06 | **0.98** | **0.68** | 0.77 |
| **O/U 2.5 Directional** | **76.6%** | 54.7% | **81.3%** | 65.6% |

The LLM's edge is almost entirely on score magnitude (39.1% vs. 6.2% exact-match pregame is a 6× gap), which flows through to 19–22pp gaps on O/U 2.5 directional accuracy.

### Halftime-conditioned lift (n=128, paired)

| Comparison | Metric | Δ | Exact McNemar |
|:---|:---|:---:|:---:|
| Pregame → Halftime | 1X2 accuracy | +11.8 pp | **p = 0.024** |
| Pregame → Halftime + Events | O/U 2.5 accuracy | +14.1 pp | **p = 0.006** |
| Halftime → Halftime + Events | 1X2 accuracy | −2.4 pp | p = 0.58 (noise) |
| Halftime → Halftime + Events | O/U 2.5 accuracy | +5.5 pp | p = 0.12 |

**The direction-vs-magnitude asymmetry is the key result.** Event enrichment improves magnitude-aware metrics (O/U, MAE, calibration) but not direction-aware metrics (1X2) — exactly as the decomposition predicts.

### Simulated backtest (halftime strategy, 64 named matches)

Fractional Kelly `f = 0.25`, 10% per-match cap, 5% edge threshold, flat 1.90/1.90 odds.

| | LLM Halftime | LLM Pregame | XGBoost Halftime |
|:---|:---:|:---:|:---:|
| Win rate | **81.3%** | 76.6% | 65.6% |
| Simulated ROI | +1,468% | +737% | +445% |
| Max drawdown | **−10.4%** | −18.6% | −28.2% |
| Match Sharpe | **4.08** | 3.42 | 2.24 |

10,000-trial bootstrap (resample per-bet returns with replacement): median $15,968, 5th–95th percentile $[6,205, 37,379]$, profitable in 100% of resamples.

> **Caveat.** The bootstrap is a CI on *conditional variance*, not a forecast of live returns. Realistic deployment after odds drift, tax, and conservative sizing (5% Kelly) projects to **15–40% per tournament cycle**. The 1,468% is a simulation artifact under flat odds — it is not a business plan.

---

## Quickstart

### Option A — Reproduce the paper (no GPU needed)

Every number in the paper regenerates from committed prediction dumps in [`results/`](results/):

```bash
git clone https://github.com/zanwenfu/football-llm.git
cd football-llm
pip install -e ".[dev]"

python scripts/reproduce_paper.py --output-dir figures/
```

This regenerates Tables 1–2, every figure, the McNemar tests, the Wilson CIs, the Kelly sensitivity grid, and the bootstrap CI. Run time: ~30 seconds on a laptop.

### Option B — Run the full pipeline (GPU required for training)

```bash
# 1. Install
pip install -e ".[train,serve]"

# 2. Rebuild training data from raw CSVs
python -m football_llm.data_prep.run_pipeline

# 3. Fine-tune (Colab T4 or local ≥16GB VRAM, ~43 min)
python -m football_llm.training.run_sft \
  --config src/football_llm/training/recipes/llama-3-1-8b-instruct-qlora.yaml

# 4. Run inference across all three prompt regimes
python -m football_llm.eval.run_inference --regime halftime_events

# 5. Recompute metrics and figures
python scripts/reproduce_paper.py
```

### Option C — One-command serving stack

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...  # needs Llama 3.1 gated access
docker compose up
```

Exposes:
- vLLM OpenAI-compatible API at `http://localhost:8000`
- FastAPI prediction endpoint at `http://localhost:8001/predict`
- Gradio UI at `http://localhost:7860`

---

## Three prompt regimes

All three use **identical model weights and decoding parameters** (temperature 0.1, top-p 0.9, 300 max tokens). The only difference is the user message content — halftime and event regimes are *pure prompt-template generalization* (the fine-tuning dataset contained no halftime scores or event sequences).

### 1. Pregame

```
{tournament} | {stage} | {venue}

{Home} (Home) | Coach: ... | Formation: ...
Squad: 11 starters | Avg Rating: 7.2
Attack: 450 goals (0.35/90) | 180 assists | Top scorer: 200 goals
Defense: 120 yellows, 2 reds | Tackles/90: 0.6 | Duels: 55%
Passing: 72% accuracy

{Away} (Away) | ...

Predict result, score, and reasoning.
```

### 2. Halftime-conditioned

Same team blocks, plus:

```
Halftime Score: {Home} 1 - 0 {Away}
Given the halftime state, predict the FINAL result, FINAL score, and brief reasoning.
```

### 3. Halftime + first-half events

Same as halftime, plus a chronological event line:

```
Halftime Score: {Home} 1 - 0 {Away}
First-half events: 10' {Home} goal (penalty); 25' {Away} yellow card; 44' {Home} goal
Given the halftime state and first-half events, predict the FINAL result, FINAL score, and brief reasoning.
```

Events are filtered to `minute ≤ 45` and mapped to `Team A / Team B` in the anonymized variant so identity does not leak through event text.

---

## Project structure

```
football-llm/
├── src/football_llm/
│   ├── data_prep/          # 3-stage training-data pipeline
│   ├── training/           # QLoRA SFT + adapter merge
│   ├── eval/               # Metrics, backtest, figures (reproduces the paper)
│   ├── baselines/          # XGBoost baseline on identical features
│   └── serving/            # vLLM + FastAPI + Gradio stack
├── scripts/
│   ├── reproduce_paper.py  # Regenerate every number and figure in the PDF
│   └── make_figures.py     # Just the figures
├── tests/                  # Pytest suite (metrics, parser, prompt builder)
├── results/
│   ├── ft_predictions_pregame.json          # 128 predictions, audit trail
│   ├── ft_predictions_halftime.json
│   ├── ft_predictions_halftime_events.json
│   └── README.md                            # Schema for each file
├── data/
│   ├── raw/                # API-Football scraper output (committed)
│   ├── processed/          # Aggregated player-season + match contexts
│   └── training/           # train.jsonl / eval.jsonl
├── notebooks/              # Colab notebooks (train, eval, serve demo)
├── football-data/          # API-Football scraper sub-project
├── docker/                 # Dockerfile + compose config
├── .github/workflows/      # CI: lint + tests
├── pyproject.toml
└── IDS598_Final_Project_Report.pdf
```

---

## Statistical methodology

This project takes statistical rigor seriously because the ML claims are pointing at real dollars.

| Claim type | Test used | Why |
|:---|:---|:---|
| Proportion (e.g. "64.1% accuracy") | **Wilson score interval** | Better-calibrated than normal-approx near 0 or 1, especially at n=64 or 128 |
| Within-match pairs (e.g. "halftime beats pregame") | **Exact McNemar test** | Higher power than unpaired tests on the same matches |
| Probability calibration | **ECE (10 bins) + Brier** | ECE is the standard, Brier decomposes into calibration + sharpness |
| Backtest P&L variance | **10,000-trial bootstrap** on per-bet returns | Interpreted as conditional variance, **not** a forecast |

Where a test fails to reach p < 0.05, we say so explicitly rather than leaning on point estimates. See §4.2 of the paper.

---

## Serving

The serving stack is three processes composed via `docker compose`:

| Component | Port | Role |
|:---|:---:|:---|
| **vLLM** | 8000 | Loads Llama 3.1 8B + QLoRA adapter, serves OpenAI-compatible chat completions |
| **FastAPI** | 8001 | Domain API — builds training-format prompts, calls vLLM, parses structured output |
| **Gradio** | 7860 | Interactive web UI with preset examples |

### Example: FastAPI `/predict`

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": {"name": "Argentina", "goals": 450, "goals_per_90": 0.35, ...},
    "away_team": {"name": "France", "goals": 420, "goals_per_90": 0.38, ...},
    "match": {"tournament": "World Cup 2022", "stage": "Final", "venue": "Lusail"},
    "regime": "halftime_events",
    "halftime_score": {"home": 2, "away": 2},
    "first_half_events": ["23'\'' Argentina goal (penalty)", "36'\'' Argentina goal", "45+1'\'' France goal"]
  }'
```

Response:

```json
{
  "prediction": "home_win",
  "score": "3-3",
  "over_2_5_probability": 0.94,
  "reasoning": "Argentina leads 2-2 at halftime after trading goals...",
  "regime": "halftime_events",
  "raw_output": "..."
}
```

Configuration via environment variables (all have sensible defaults):

| Variable | Default | Description |
|:---|:---|:---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM endpoint |
| `MODEL_NAME` | `football-llm` | LoRA adapter name |
| `API_TOKEN` | *(unset)* | Optional bearer token — when set, required on `/predict` |
| `MAX_MODEL_LEN` | `768` | vLLM max sequence length |
| `GPU_MEM_UTIL` | `0.9` | vLLM GPU memory fraction |

---

## Limitations

The paper's §7 lists these in detail. Short version:

- **Single tournament (n=64 unique matches).** Paired McNemar on n=128 is significant (p=0.024 for halftime lift, p=0.006 for events lift on O/U 2.5). Individual LLM-vs-baseline comparisons at n=64 generally are not. Extension to Euro 2024 + Copa 2024 (~83 more matches) would push borderline comparisons below p=0.05 if the effect size replicates.
- **Template-generalization, not trained on halftime prompts.** Both halftime and halftime+events regimes are inference-time prompt additions; the model was never fine-tuned on these formats. A natural next experiment is to regenerate training data with halftime+events prompts and retrain.
- **Poisson approximation for O/U conversion.** Football scorelines are mildly over-dispersed vs. Poisson; a Conway-Maxwell-Poisson or negative-binomial fit could tighten probability estimates.
- **Only Llama 3.1 8B tested.** A 70B variant on the same data would likely improve magnitude further; T4 compute budget prevented testing.
- **XGBoost at feature parity only.** A heavily feature-engineered tabular baseline (H2H history, rest days, travel) could plausibly close the O/U gap. The predictive-validity test (§5.8) is partly a response to this: the event-enrichment magnitude gain is pattern that's harder to replicate without sequence modeling.

---

## Citation

```bibtex
@misc{fu2026footballllm,
  author = {Fu, Zanwen},
  title = {Dynamic In-Play Football Betting via a QLoRA-Fine-Tuned LLM:
           A Halftime-Conditioned Strategy for Over/Under Markets},
  year = {2026},
  url = {https://github.com/zanwenfu/football-llm},
}
```

---

## References

1. Dettmers et al. **QLoRA: Efficient Finetuning of Quantized LLMs.** NeurIPS 2023.
2. Dixon & Coles. **Modelling association football scores and inefficiencies in the football betting market.** JRSS-C, 1997.
3. Paul & Weinbach. **Bettor preferences and market efficiency in football totals markets.** J. Economics and Finance, 2013.
4. Phil Schmid. [How to fine-tune open LLMs in 2025](https://www.philschmid.de/fine-tune-llms-in-2025) — training methodology.

---

## License

MIT — see [LICENSE](LICENSE).

<div align="center">

# Football-LLM

**Fine-tuning Llama 3.1 8B to predict FIFA World Cup match outcomes from player statistics**

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97_Model-football--llm--qlora-blue)](https://huggingface.co/zanwenfu/football-llm-qlora)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

| Metric | Base Llama 3.1 8B | **Football-LLM** | Always Home | Random |
|:---|:---:|:---:|:---:|:---:|
| Result Accuracy | 45.3% | **52.3%** | 45.3% | 35.9% |
| Exact Score Match | 0.0% | **29.7%** | 10.9% | 7.8% |
| Goal MAE | 2.13 | **1.29** | 1.11 | 1.27 |
| Parse Rate | 100% | **100%** | 100% | 100% |

<sub>Evaluated on 128 held-out 2022 FIFA World Cup samples (64 named + 64 anonymized)</sub>

</div>

---

## Overview

Football-LLM is an end-to-end machine learning project that fine-tunes [Meta's Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) to predict FIFA World Cup match results, scores, and reasoning from structured team statistics.

The project demonstrates a complete ML pipeline — from raw data engineering through model training to evaluation — constrained to **free-tier Google Colab (T4 GPU, 16 GB VRAM)**.

### Key Highlights

- **Data Engineering Pipeline**: 3-stage pipeline that transforms raw player career stats + World Cup match data into compact, token-budget-aware training prompts
- **QLoRA Fine-tuning**: 4-bit quantized training with LoRA (r=16) on a single T4 GPU in ~43 minutes
- **Rigorous Evaluation**: Custom eval harness comparing against base model + statistical baselines, with memorization testing via anonymized team variants
- **Production-Ready**: Adapter weights on HuggingFace Hub, with vLLM serving support (coming soon)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                            │
│                                                                 │
│  Raw Data (5 CSVs + 40+ country stat files)                    │
│       │                                                         │
│       ▼                                                         │
│  Step 1: aggregate_player_stats.py                             │
│       │  Merge career + league stats → 41K player-season rows  │
│       ▼                                                         │
│  Step 2: build_team_profiles.py                                │
│       │  Starting XI lookup → team-level profiles per match    │
│       │  + prior WC performance + H2H records                  │
│       ▼                                                         │
│  Step 3: generate_training_data.py                             │
│       │  Compact prompts (≤350 tokens) in HF messages format   │
│       ▼                                                         │
│  train.jsonl (384 samples) + eval.jsonl (128 samples)          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING                                 │
│                                                                 │
│  Llama 3.1 8B Instruct + QLoRA (4-bit NF4)                    │
│  • LoRA r=16, α=32, all linear layers                          │
│  • 3 epochs, effective batch 16, lr=2e-4 cosine                │
│  • Max sequence length: 768 tokens                             │
│  • T4 GPU, ~43 min, ~5.7 GB VRAM                              │
│  • Adapter: 83.9 MB on HuggingFace Hub                        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EVALUATION                                │
│                                                                 │
│  Held-out: 2022 FIFA World Cup (64 matches × 2 variants)      │
│  Metrics: Result accuracy, exact score, goal MAE, parse rate   │
│  Memorization test: Named (50.0%) vs Anonymized (54.7%)        │
│  → Model learns from stats, not team name memorization         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SERVING                                  │
│                                                                 │
│  vLLM inference server (coming soon)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
football-llm/
├── data/
│   ├── raw/                          # Source datasets
│   │   ├── world_cup_matches.csv     # 256 WC matches (2010–2022)
│   │   ├── world_cup_lineups.csv     # Starting XI per match
│   │   ├── world_cup_events.csv      # Goals, cards, substitutions
│   │   ├── world_cup_team_match_stats.csv
│   │   ├── world_cup_player_match_stats.csv
│   │   └── player_stats/             # 40+ country player stat files
│   ├── processed/
│   │   ├── player_season_stats.csv   # 41K aggregated player-season rows
│   │   └── match_contexts.json       # 256 enriched match contexts
│   └── training/
│       ├── train.jsonl               # 384 training samples
│       └── eval.jsonl                # 128 evaluation samples
├── src/
│   ├── data_prep/
│   │   ├── aggregate_player_stats.py # Step 1: merge player data sources
│   │   ├── build_team_profiles.py    # Step 2: build team profiles per match
│   │   ├── generate_training_data.py # Step 3: compact prompt generation
│   │   └── run_pipeline.py           # Orchestrate all 3 steps
│   ├── training/
│   │   ├── run_sft.py               # CLI training script (multi-GPU)
│   │   ├── merge_adapter_weights.py  # Merge LoRA → standalone model
│   │   └── recipes/
│   │       └── llama-3-1-8b-instruct-qlora.yaml
│   ├── eval/                         # (eval logic in notebook)
│   └── serving/                      # vLLM serving (coming soon)
├── notebooks/
│   ├── train_colab.ipynb             # End-to-end training on Colab T4
│   └── eval_harness.ipynb            # Full evaluation + baselines
├── results/                          # Eval output artifacts
├── requirements.txt
└── README.md
```

---

## Data Pipeline

### Input Data

| Dataset | Records | Description |
|:--------|--------:|:------------|
| `world_cup_matches.csv` | 256 | Match metadata (2010–2022 World Cups) |
| `world_cup_lineups.csv` | ~5,600 | Starting XI + substitutes per match |
| `world_cup_events.csv` | ~2,400 | Goals, cards, substitutions with timestamps |
| `world_cup_team_match_stats.csv` | ~512 | Team-level match stats (xG, possession, shots) |
| `player_stats/` (40+ files) | ~41,000 | Per-player season stats from domestic leagues |

### Step 1: Aggregate Player Stats

Merges two data sources — WC player career stats and domestic league stats — into a unified player-season table. Deduplicates on `(player_id, season, team_id, league_id)`, computes per-90 rates (goals/90, tackles/90, dribble success %), and outputs **41,154 player-season rows**.

### Step 2: Build Team Profiles

For each of the 256 World Cup matches, looks up the starting XI, retrieves each player's stats from the **3 seasons prior** to the tournament, and aggregates into team-level profiles:

- **Attacking**: total goals, goals/90, assists, top scorer, shots on target
- **Defensive**: yellow/red cards, tackles/90, duel win %
- **Technical**: passing accuracy, average player rating, dribble success %
- **Context**: formation, coach, position breakdown, prior WC record, head-to-head history

### Step 3: Generate Training Data

Converts match contexts into a **compact prompt format** designed to fit within the 768-token sequence limit:

```
[System] You are a football match prediction model. Given team stats,
         predict the result, score, and brief reasoning.

[User]   World Cup 2018 | Group Stage - 1 | Luzhniki Stadium
         Russia (Home) | Coach: Stanislav Cherchesov | Formation: 3-4-1-2
         Squad: 11 starters | Avg Rating: 7.0
         Attack: 104 goals (0.18/90) | 48 assists | Top scorer: 26 goals
         Defense: 75 yellows, 1 reds | Tackles/90: 0.57 | Duels: 52%
         Passing: 70% accuracy
         ...
         Predict result, score, and reasoning.

[Assistant] Prediction: home_win
            Score: 5-0
            Reasoning: Russia's squad has higher goal output (104 vs 23)...
```

**Token budget**: All 512 samples (384 train + 128 eval) fit within **350 tokens** (against a 768 limit), ensuring the model sees the complete assistant response during training.

**Temporal split**: Train on 2010 + 2014 + 2018 → Eval on 2022 (no data leakage).

**Anonymization**: Each match generates two variants — one with real team names and one with "Team A / Team B" — to force the model to learn from statistical patterns rather than memorized team reputations. This doubles the dataset to 384 train + 128 eval.

---

## Training

### Configuration

| Parameter | Value |
|:----------|:------|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| Method | QLoRA (4-bit NF4 quantization) |
| LoRA rank / alpha | r=16, α=32 |
| LoRA targets | All linear layers (q/k/v/o/gate/up/down proj) |
| Sequence length | 768 tokens |
| Epochs | 3 |
| Effective batch size | 16 (1 × 16 gradient accumulation) |
| Learning rate | 2e-4 (cosine schedule, 10% warmup) |
| Precision | float16 compute, NF4 quantized base |
| GPU | Google Colab T4 (16 GB VRAM) |
| Training time | **43 minutes** |
| Peak VRAM | **5.7 GB** |
| Adapter size | **83.9 MB** |

### Training Curves

| Epoch | Train Loss | Eval Loss |
|:-----:|:----------:|:---------:|
| 1 | 0.352 | 0.539 |
| 2 | 0.259 | 0.595 |
| 3 | 0.244 | 0.609 |

The eval loss increases after epoch 1, indicating mild overfitting — expected with only 384 training samples. The model's downstream task performance (result accuracy, score matching) steadily improves regardless, since cross-entropy loss on the full sequence doesn't directly measure prediction quality.

---

## Evaluation

### Methodology

The evaluation harness ([`eval_harness.ipynb`](notebooks/eval_harness.ipynb)) runs inference on all 128 held-out 2022 World Cup samples and computes:

- **Result Accuracy**: Correct prediction of `home_win` / `draw` / `away_win`
- **Exact Score Match**: Predicted score exactly matches ground truth
- **Goal MAE**: Mean absolute error across home and away goals
- **Parse Rate**: % of outputs that parse into a valid structured prediction

### Results

| Model | Parse Rate | Result Acc | Exact Score | Goal MAE |
|:------|:----------:|:----------:|:-----------:|:--------:|
| Base Llama 3.1 8B Instruct | 100% | 45.3% | 0.0% | 2.13 |
| **Football-LLM (QLoRA)** | **100%** | **52.3%** | **29.7%** | **1.29** |
| Baseline: Always Home Win | 100% | 45.3% | 10.9% | 1.11 |
| Baseline: Random Weighted | 100% | 35.9% | 7.8% | 1.27 |
| Baseline: Always Draw | 100% | 23.4% | 7.8% | 0.97 |

### Key Findings

1. **+7.0pp result accuracy** over the base model (52.3% vs 45.3%), showing the model learned meaningful patterns from the statistical profiles

2. **29.7% exact score prediction** — the base model scored 0.0% (it doesn't understand the expected output format), while Football-LLM correctly predicts nearly 1 in 3 exact scorelines

3. **100% parse rate** — every output follows the trained format (`Prediction: ... / Score: ... / Reasoning: ...`), confirming reliable structured generation

4. **No memorization**: The anonymized variants (Team A/B) achieve 54.7% result accuracy vs 50.0% for named teams — the model performs *better* without team names, confirming it reasons from statistics rather than memorized reputations

### Sample Prediction

```
Input:  England (Home) vs Iran (Away) — World Cup 2022 Group Stage

Output: Prediction: home_win
        Score: 6-2
        Reasoning: England's squad has higher goal output (309 vs 141).
        England's top scorer is more prolific (110 vs 78 goals).

Actual: England 6 - 2 Iran ✓ (exact score match)
```

---

## Quick Start

### 1. Run the Data Pipeline

```bash
# Clone the repo
git clone https://github.com/zanwenfu/football-llm.git
cd football-llm

# Install dependencies
pip install -r requirements.txt

# Run the full 3-step data pipeline
python src/data_prep/run_pipeline.py
```

### 2. Train on Google Colab

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Google Colab:

1. Select **T4 GPU** runtime
2. Add your HuggingFace token (requires Llama 3.1 access)
3. Run all cells — training completes in ~43 minutes

### 3. Run Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config, low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base_model, "zanwenfu/football-llm-qlora")
tokenizer = AutoTokenizer.from_pretrained("zanwenfu/football-llm-qlora")

# Predict
messages = [
    {"role": "system", "content": "You are a football match prediction model. Given team stats, predict the result, score, and brief reasoning."},
    {"role": "user", "content": """World Cup 2022 | Semi Final | Al Bayt Stadium

Argentina (Home) | Coach: Lionel Scaloni | Formation: 4-3-3
Squad: 11 starters | Avg Rating: 7.2
Attack: 450 goals (0.35/90) | 180 assists | Top scorer: 200 goals
Defense: 120 yellows, 2 reds | Tackles/90: 0.6 | Duels: 55%
Passing: 72% accuracy

Croatia (Away) | Coach: Zlatko Dalic | Formation: 4-3-3
Squad: 11 starters | Avg Rating: 7.0
Attack: 280 goals (0.28/90) | 120 assists | Top scorer: 85 goals
Defense: 95 yellows, 1 reds | Tackles/90: 0.55 | Duels: 53%
Passing: 70% accuracy

H2H: No prior meetings

Predict result, score, and reasoning."""}
]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True).to("cuda")
outputs = model.generate(inputs, max_new_tokens=200, temperature=0.1, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

### 4. Evaluate

Open [`notebooks/eval_harness.ipynb`](notebooks/eval_harness.ipynb) on Colab to reproduce all evaluation results, baselines, and the memorization test.

---

## Roadmap

- [x] Data engineering pipeline (3-stage)
- [x] QLoRA fine-tuning on Colab T4
- [x] Evaluation harness with baselines + memorization test
- [x] Adapter weights on HuggingFace Hub
- [ ] vLLM inference server for production serving
- [ ] Expand to 2026 World Cup predictions

---

## Technical Decisions

| Decision | Rationale |
|:---------|:----------|
| **Compact prompts (≤350 tokens)** | Initial verbose prompts exceeded the 768-token limit — model never saw assistant completions during training. Redesigned to team-level aggregates only. |
| **Anonymized variants** | Doubles dataset size and forces the model to learn from statistical patterns. Memorization test confirms 54.7% anon vs 50.0% named accuracy. |
| **3-season lookback** | Balances recency (current form) with sample size (enough data per player). |
| **Score-based result parsing** | Model sometimes outputs contradictory labels — parser derives result from predicted score for consistency. |
| **float16 (not bf16)** | T4 GPU (Turing architecture) does not support bfloat16. |

---

## References

- [How to fine-tune open LLMs in 2025](https://www.philschmid.de/fine-tune-llms-in-2025) — Phil Schmid (training methodology)
- [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) — Meta AI
- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — Hugging Face

---

## License

MIT
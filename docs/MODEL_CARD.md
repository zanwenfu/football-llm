---
language:
  - en
license: mit
tags:
  - llama
  - qlora
  - peft
  - football
  - sports-analytics
  - sports-betting
  - fine-tuning
datasets:
  - custom
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: peft
pipeline_tag: text-generation
model-index:
  - name: football-llm-qlora
    results:
      - task:
          type: text-generation
          name: World Cup 1X2 Prediction
        dataset:
          name: 2022 FIFA World Cup (held-out)
          type: custom
        metrics:
          - name: 1X2 Result Accuracy (halftime-conditioned)
            type: accuracy
            value: 0.641
          - name: O/U 2.5 Directional Accuracy (halftime+events, named)
            type: accuracy
            value: 0.844
          - name: ECE on O/U 2.5 (halftime+events)
            type: expected_calibration_error
            value: 0.182
---

# Football-LLM: QLoRA-fine-tuned Llama 3.1 8B for World Cup Prediction

A LoRA adapter over [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for predicting final scorelines of FIFA World Cup matches. Supports three inference regimes: pregame, halftime-conditioned, and halftime + first-half event enrichment.

The adapter's **magnitude-aware edge** — driven by pretrained scoreline priors — is the intended contribution. See [the paper](https://github.com/zanwenfu/football-llm/blob/main/IDS598_Final_Project_Report.pdf) for the full magnitude/direction decomposition.

| Metric (held-out 2022 WC, n=128) | Pregame | Halftime | **Halftime + Events** |
|:---|:---:|:---:|:---:|
| 1X2 Result Accuracy | 52.3% | **64.1%** | 61.7% |
| O/U 2.5 Directional Accuracy | 65.6% | 74.2% | **79.7%** |
| Goal MAE | 1.32 | 1.21 | **1.12** |
| ECE (O/U 2.5) | 0.272 | 0.239 | **0.182** |

On the 64 named matches specifically: halftime+events reaches **84.4%** O/U 2.5 accuracy (Wilson CI [0.736, 0.913]).

## Intended use

- **Research** on sports-market inefficiencies, calibration, and the magnitude/direction decomposition in LLMs.
- **Educational** examples of QLoRA fine-tuning on free-tier hardware (Colab T4, 43 minutes, 5.7 GB peak VRAM).
- **Defensive/analytical** use by risk teams at sportsbooks who want a reference implementation of halftime-conditioned O/U pricing.

## Out-of-scope use

- **Deployment for live real-money betting without independent verification.** The published 1,468% ROI is a *simulation* under flat 1.90/1.90 odds and Kelly 25% — realistic deployment projects to 15–40% per tournament cycle after odds drift, vig, and jurisdiction-dependent tax. Do not skip paper-trading.
- **Non-World-Cup competitions** without retraining or validation. Named/anonymized ablation shows the adapter's magnitude edge is partly driven by team-identity priors; other leagues may behave differently.
- **Predicting outcomes for matches where the team-stat pipeline is not available.** The adapter expects a specific compact prompt format (see below).

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
)
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", quantization_config=bnb, low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base, "zanwenfu/football-llm-qlora")
tok = AutoTokenizer.from_pretrained("zanwenfu/football-llm-qlora")

# Halftime-conditioned prompt
messages = [
    {"role": "system", "content": "You are a football match prediction model. Given team stats, predict the result, score, and brief reasoning."},
    {"role": "user", "content": """World Cup 2022 | Group Stage | Lusail Stadium

Argentina (Home) | Coach: Scaloni | Formation: 4-3-3
Squad: 11 starters | Avg Rating: 7.2
Attack: 450 goals (0.35/90) | 180 assists | Top scorer: 200 goals
Defense: 120 yellows, 2 reds | Tackles/90: 0.6 | Duels: 55%
Passing: 72% accuracy

Saudi Arabia (Away) | Coach: Renard | Formation: 4-1-4-1
Squad: 11 starters | Avg Rating: 6.7
Attack: 180 goals (0.20/90) | 60 assists | Top scorer: 35 goals
Defense: 90 yellows, 1 reds | Tackles/90: 0.7 | Duels: 50%
Passing: 68% accuracy

Halftime Score: Argentina 1 - 0 Saudi Arabia
Given the halftime state, predict the FINAL result, FINAL score, and brief reasoning."""},
]
inputs = tok.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True).to("cuda")
out = model.generate(inputs, max_new_tokens=300, temperature=0.1, top_p=0.9, do_sample=True)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

Expected output format:

```
Prediction: home_win
Score: 2-0
Reasoning: Argentina leads 1-0 with stronger attack...
```

## Training details

- **Base model:** `meta-llama/Llama-3.1-8B-Instruct`
- **Method:** QLoRA (4-bit NF4 quantization of base + LoRA adapter)
- **LoRA config:** rank 16, α 32, dropout 0.05, all 7 linear projections (`q,k,v,o_proj + gate,up,down_proj`)
- **Training set:** 384 samples = 192 WC matches (2010/2014/2018) × 2 anonymization variants
- **Evaluation set:** 128 samples = 64 WC 2022 matches × 2 anonymization variants (strict temporal split)
- **Optimizer:** AdamW, lr 2e-4, cosine schedule
- **Effective batch size:** 16 (1 per-device × 16 gradient accumulation)
- **Sequence length:** 768 tokens
- **Epochs:** 3
- **Hardware:** Google Colab T4 (16 GB VRAM)
- **Training time:** ~43 minutes
- **Peak VRAM:** 5.7 GB
- **Adapter size:** 83.9 MB

## Evaluation protocol

- **Wilson score intervals** for all proportions (§4.2 of paper).
- **Exact McNemar tests** for paired within-match comparisons.
- **ECE (10 bins)** and **Brier score** for probability calibration.
- **10,000-trial bootstrap** on per-bet returns for backtest CI.

The paired pregame → halftime lift on 1X2 accuracy is **statistically significant** (exact McNemar p = 0.024, n = 128). The paired pregame → halftime+events lift on O/U 2.5 directional accuracy is even stronger (p = 0.006, n = 128).

## Three prompt regimes

Critically, the fine-tuning dataset contained **no halftime scores or event sequences** — both halftime regimes are **pure prompt-template generalization** at inference time.

### 1. Pregame

Team stats only, ending in `"Predict result, score, and reasoning."`

### 2. Halftime-conditioned

Team stats + one additional line:
```
Halftime Score: {Home} {HH} - {HA} {Away}
Given the halftime state, predict the FINAL result, FINAL score, and brief reasoning.
```

### 3. Halftime + first-half events

Team stats + halftime score + chronological event line (goals/cards with timestamps ≤ 45'):
```
Halftime Score: {Home} 2 - 2 {Away}
First-half events: 23' {Home} goal (penalty); 36' {Home} goal; 45+1' {Away} goal
Given the halftime state and first-half events, predict the FINAL result, FINAL score, and brief reasoning.
```

## Known limitations

- **Single-tournament evaluation (n=64 unique matches).** Claims are underpowered outside the aggregate 128-sample set. Extension to Euro 2024 + Copa 2024 (+83 matches) would push borderline comparisons below p=0.05 if the effect size replicates.
- **Named vs. anonymized gap.** Team names carry magnitude priors (named halftime+events O/U 2.5: 84.4%, anonymized: 75.0%). Model is strongest when team identities are present; anonymized performance is still positive-EV but reduced.
- **Poisson over-dispersion.** The O/U conversion assumes Poisson; football scorelines are mildly over-dispersed (Dixon & Coles 1997). A CMP or negative-binomial fit could tighten probability estimates.
- **Template-generalization risk.** Halftime+events shows a small result-accuracy regression vs. halftime-only (−2.4pp, not significant), possibly indicating mild template drift. Fine-tuning on halftime+events prompts directly is expected to eliminate this.
- **Gated base model.** Llama 3.1 access must be granted on Hugging Face before this adapter can be loaded.

## Ethical considerations

Sports-betting strategies have real-world money consequences. The published backtest is a **simulation** and should not be interpreted as financial advice or a forecast of live returns. Jurisdiction-dependent regulations apply in the US and elsewhere. The research framing of this adapter is the magnitude/direction decomposition — the backtest exists to show the mechanism produces positive-EV signal, not to sell a trading system.

## Citation

```bibtex
@misc{fu2026footballllm,
  author = {Fu, Zanwen},
  title = {Dynamic In-Play Football Betting via a QLoRA-Fine-Tuned LLM:
           A Halftime-Conditioned Strategy for Over/Under Markets},
  year = {2026},
  howpublished = {\url{https://github.com/zanwenfu/football-llm}},
}
```

## License

MIT. The base model (Llama 3.1 8B Instruct) is subject to the [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/).

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
          name: World Cup score prediction (team names hidden)
        dataset:
          name: 2022 FIFA World Cup (64 matches; inside the base model's pretraining window)
          type: custom
        metrics:
          - name: 1X2 accuracy, pregame, names hidden
            type: accuracy
            value: 0.531
          - name: O/U 2.5 accuracy, pregame, names hidden
            type: accuracy
            value: 0.562
          - name: O/U 2.5 accuracy, halftime, names hidden
            type: accuracy
            value: 0.703
---

# Football-LLM: QLoRA-fine-tuned Llama 3.1 8B for World Cup Prediction

A LoRA adapter over [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) that predicts final scorelines of FIFA World Cup matches from player-level team statistics, before kickoff or at halftime.

> **Read this before using the numbers.** The adapter was evaluated on the 2022 World Cup, but Llama 3.1's pretraining data runs to December 2023, so the base model has seen those results. With team names in the prompt it reproduces many real 2022 scorelines, sometimes with home and away swapped. With names hidden it performs about as well as simple rules. See the [audit report](https://github.com/zanwenfu/football-llm/blob/main/report/A2_report.pdf).

| 2022 World Cup, 64 matches | Pregame 1X2 | Pregame O/U 2.5 | Halftime 1X2 | Halftime O/U 2.5 |
|:---|:---:|:---:|:---:|:---:|
| Team names in prompt (contaminated) | 50.0% | 78.1% | 64.1% | 81.2% |
| Team names hidden | 53.1% | 56.2% | 64.1% | 70.3% |
| No model: always under / halftime goals ×2 | – | 53.1% | – | 75.0% |

## Intended use

- **Research** on evaluation leakage in LLM forecasting: the named/hidden prompt pairs make a ready-made contamination test.
- **Educational** examples of QLoRA fine-tuning on free-tier hardware (Colab T4, 43 minutes, 5.7 GB peak VRAM).
- **Forward testing** on matches after December 2023 (Euro 2024, Copa América 2024, World Cup 2026), which is the only setting where its accuracy says anything about skill.

## Out-of-scope use

- **Real-money betting.** No evaluation of this adapter has shown an edge against real odds. The spring backtest's +1,468% used flat 1.90/1.90 odds, and a rule with no model earns +1,125% under the same simulation.
- **Treating accuracy on pre-2024 matches as evidence of skill.** Those results are in the base model's pretraining data.

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

- 64 matches of the 2022 World Cup, each prompted twice: with team names and with "Team A" / "Team B".
- Wilson score intervals for proportions; paired exact McNemar within each prompt variant (pooling both variants counts each match twice).
- Leakage checks: named-vs-hidden gap, a mirror-score permutation test, and comparison with XGBoost, Dixon-Coles and no-model rules. All in [`scripts/leakage_audit.py`](https://github.com/zanwenfu/football-llm/blob/main/scripts/leakage_audit.py).

## Prompt regimes

The fine-tuning data contained no halftime scores, so the halftime regime is prompt-template generalisation at inference time.

- **Pregame:** team stats only, ending in `"Predict result, score, and reasoning."`
- **Halftime:** team stats plus
  ```
  Halftime Score: {Home} {HH} - {HA} {Away}
  Given the halftime state, predict the FINAL result, FINAL score, and brief reasoning.
  ```

The serving API also accepts an experimental `halftime_events` regime (first-half goals and cards). It has no committed evaluation.

## Known limitations

- **Contaminated evaluation.** See the note at the top. Only matches after December 2023 can measure skill.
- **Hidden prompts are only partly hidden.** They still include the coach's name, stadium and round.
- **Decoding.** Evaluation sampled at temperature 0.1 with `repetition_penalty=1.1`. Before kickoff the adapter almost never predicts a draw and occasionally emits scores such as 0–8.
- **Labels include extra time**, while betting markets settle at 90 minutes.
- **Small sample.** 64 matches give 95% intervals of about ±12 points on any accuracy.
- **Gated base model.** Llama 3.1 access must be granted on Hugging Face before this adapter can be loaded.

## Ethical considerations

Sports betting has real financial consequences. Nothing in this model card or the accompanying report is financial advice, and no result here supports betting with this adapter.

## License

MIT. The base model (Llama 3.1 8B Instruct) is subject to the [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/).

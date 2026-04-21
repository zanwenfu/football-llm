# Data Card — Football-LLM Training & Evaluation Set

The training pipeline ingests raw API-Football data, aggregates it into team-level profiles, and renders compact token-budget-aware prompts. This card documents the schema, provenance, and temporal split for each derived artifact.

## Summary

| Artifact | Rows | Location | Purpose |
|:---|:---:|:---|:---|
| `data/raw/world_cup_matches.csv` | 256 | raw CSV | One row per World Cup match (2010–2022) |
| `data/raw/world_cup_lineups.csv` | ~5,600 | raw CSV | Starting XI + substitutes per match |
| `data/raw/world_cup_events.csv` | ~2,400 | raw CSV | Goals, cards, substitutions with timestamps |
| `data/raw/world_cup_team_match_stats.csv` | ~512 | raw CSV | Team-level per-match stats (xG, possession, shots) |
| `data/raw/player_stats/` | ~41,000 | 40+ files | Per-player per-season stats from domestic leagues + WC |
| `data/processed/player_season_stats.csv` | 41,154 | processed | Aggregated player-season stats (output of Step 1) |
| `data/processed/match_contexts.json` | 256 | processed | Enriched per-match contexts used by XGBoost + prompt gen |
| `data/training/train.jsonl` | 384 | training | 192 matches × 2 variants (named + anonymized) |
| `data/training/eval.jsonl` | 128 | training | 64 matches × 2 variants (held-out 2022 WC) |

## Provenance

**Source.** [API-Football](https://www.api-football.com/) vendor feed. Scraping scripts live under [`football-data/`](../football-data/).

**Tournaments covered.** FIFA World Cups 2010, 2014, 2018, 2022 (256 matches = 64 × 4).

**Collection window.** Historical data retrieved via API calls across 2025-Q1. Reruns require a valid API-Football key (see `football-data/README.md`).

**License.** API-Football's [terms of service](https://www.api-football.com/documentation-v3#terms) govern raw data re-distribution. We commit aggregated derivatives (team-level profiles, prompts) — not the raw API responses beyond what's present in `data/raw/`.

## Temporal split

Strict — no leakage:

| Set | Tournaments | Matches | JSONL rows |
|:---|:---|:---:|:---:|
| Train | 2010 + 2014 + 2018 | 192 | 384 |
| Eval  | 2022 | 64 | 128 |

All player statistics are filtered to **prior-season-only** before each match, so a 2018 match is conditioned only on pre-2018 league stats. Detection of leakage would manifest as sharp overfitting to named team identities — see §5.6 ("Named vs. Anonymized") for the diagnostic ablation.

## Schema — `match_contexts.json`

One JSON object per match:

```json
{
  "fixture_id": 855736,                   // API-Football fixture ID
  "world_cup_year": 2022,
  "date": "2022-11-20T16:00:00+00:00",
  "round": "Group Stage - 1",
  "venue": "Al Bayt Stadium",
  "venue_city": "Al Khor",
  "referee": "D. Orsato",
  "home_team": "Qatar",
  "away_team": "Ecuador",
  "home_profile": { /* see below */ },
  "away_profile": { /* see below */ },
  "home_prior_wc": { /* history in previous WCs */ },
  "away_prior_wc": { /* ... */ },
  "h2h": [ /* head-to-head history */ ],
  "events": [ /* goals, cards, subs with timestamps */ ],
  "halftime_home": 0,
  "halftime_away": 2,
  "home_goals": 0,
  "away_goals": 2,
  "result": "away_win",
  "went_to_extra_time": false,
  "went_to_penalties": false
}
```

### Team profile schema (`home_profile` / `away_profile`)

```json
{
  "team_id": 779,
  "team_name": "Ecuador",
  "formation": "4-4-2",
  "coach": "Gustavo Alfaro",
  "num_starters_with_data": 11,
  "total_starters": 11,

  // Squad aggregates (sum over starting XI)
  "team_total_goals": 190.0,
  "team_total_assists": 82.0,
  "team_total_appearances": 621.0,
  "team_total_minutes": 52184.0,
  "team_avg_goals_per_90": 0.33,
  "team_total_shots_total": 0.0,         // 0 = missing in source
  "team_total_shots_on_target": 0.0,
  "team_total_cards_yellow": 45.0,
  "team_total_cards_red": 1.0,

  // Position breakdown of starting XI
  "position_breakdown": {
    "Attacker": 2, "Midfielder": 4, "Defender": 4, "Goalkeeper": 1
  },

  // Per-player summaries (one entry per starter)
  "player_summaries": [
    {"name": "Hernán Galíndez", "position": "Goalkeeper",
     "goals": 0, "assists": 0, "appearances": 20, "minutes": 1658, "goals_per_90": 0.0},
    // ...
  ]
}
```

## Schema — `train.jsonl` / `eval.jsonl`

HuggingFace chat-format JSONL (one match per line):

```json
{
  "messages": [
    {"role": "system", "content": "You are a football match prediction model..."},
    {"role": "user", "content": "World Cup 2022 | Group Stage - 1 | ...\n..."},
    {"role": "assistant", "content": "Prediction: home_win\nScore: 2-0\nReasoning: ..."}
  ],
  "fixture_id": 855736,
  "world_cup_year": 2022,
  "anonymized": false
}
```

**Token budget.** All 512 samples (384 train + 128 eval) fit within **350 tokens** against the 768-token training sequence limit, ensuring the assistant response is never truncated during SFT.

**Anonymization.** Each match generates two variants: one with real team names ("Argentina", "Brazil") and one with `Team A` / `Team B` — including in event strings so identity doesn't leak. This doubles the dataset and enables the named/anonymized ablation in §5.6 of the paper.

## Schema — prediction JSONs (`results/ft_predictions_*.json`)

See [`results/README.md`](../results/README.md).

## Known issues & caveats

- **Missing shot stats.** `team_total_shots_total` and `team_total_shots_on_target` are 0 for 2010/2014 matches — the API-Football feed doesn't carry these for older tournaments. Downstream features that depend on shot counts should use cards + goals as a proxy.
- **Coach/formation NaNs.** ~3% of profiles have missing coach or formation. `build_team_profiles.py` propagates these as NaN; the prompt template renders them as `"?"`.
- **Pre-match staff cards.** The API occasionally reports cards at `minute < 0` for technical staff pre-match (7 instances in the corpus). Filtered in the event-enrichment prompt.
- **Halftime score coverage.** Present for all 256 matches.
- **Fixture IDs are monotonic.** The 2022 fixture IDs increase with match order, so we sort by `fixture_id` for chronological backtests.

## Reproducibility

```bash
# Regenerate all processed artifacts from data/raw/:
python -m football_llm.data_prep.run_pipeline

# Outputs:
#   data/processed/player_season_stats.csv
#   data/processed/match_contexts.json
#   data/training/train.jsonl
#   data/training/eval.jsonl
```

Runtime: ~30 seconds on a laptop.

## Responsible use

The player-season stats contain personally-identifiable information about professional footballers (name, position, stats). These are drawn entirely from publicly-available sports records; no private data is used. Downstream consumers should still respect the privacy norms of the sports-analytics community (e.g., don't use these profiles for non-sporting profiling of individuals).

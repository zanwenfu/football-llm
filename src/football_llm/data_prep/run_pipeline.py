"""Run the full data preparation pipeline.

  Step 1: Aggregate player stats from both data sources
  Step 2: Build team profiles for each WC match
  Step 3: Generate training/eval JSONL files

Usage:
    python -m football_llm.data_prep.run_pipeline
"""

from football_llm.data_prep.aggregate_player_stats import run as step1_aggregate
from football_llm.data_prep.build_team_profiles import run as step2_profiles
from football_llm.data_prep.generate_training_data import run as step3_generate


def main() -> None:
    print("=" * 60)
    print("STEP 1: Aggregating player stats...")
    print("=" * 60)
    step1_aggregate()

    print("\n" + "=" * 60)
    print("STEP 2: Building team profiles for each match...")
    print("=" * 60)
    step2_profiles()

    print("\n" + "=" * 60)
    print("STEP 3: Generating training data...")
    print("=" * 60)
    step3_generate()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Output files:")
    print("  data/processed/player_season_stats.csv")
    print("  data/processed/match_contexts.json")
    print("  data/training/train.jsonl")
    print("  data/training/eval.jsonl")
    print("  data/training/sample_training_example.json")


if __name__ == "__main__":
    main()

"""
Step 1: Aggregate per-player career stats into season-level summaries.

Merges wc_player_career_stats_clean and statistics_clean into a single
player stats pool. For each player, produces per-season aggregated stats
that can later be looked up by team profiles.
"""
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')

# Key stats columns to aggregate
STAT_COLS = [
    'appearances', 'lineups', 'minutes', 'rating',
    'goals', 'assists', 'shots_total', 'shots_on_target',
    'passes_total', 'passes_key', 'passes_accuracy',
    'tackles_total', 'tackles_blocks', 'tackles_interceptions',
    'duels_total', 'duels_won',
    'dribbles_attempts', 'dribbles_success',
    'fouls_drawn', 'fouls_committed',
    'cards_yellow', 'cards_red',
    'goals_conceded', 'saves', 'penalty_saved',
]

# Columns to keep as metadata (take first non-null per player-season)
META_COLS = [
    'player_id', 'player_name', 'firstname', 'lastname',
    'nationality', 'birth_date', 'position', 'height_cm', 'weight_kg',
    'wc_years', 'wc_teams', 'wc_team_ids',
]


def load_career_stats() -> pd.DataFrame:
    """Load historical WC player career stats (players from 2010-2022 WCs)."""
    path = os.path.join(RAW_DIR, 'player_stats', 'wc_player_career_stats_clean',
                        'all_wc_player_career_stats_clean.csv')
    df = pd.read_csv(path)
    df['source'] = 'wc_career'
    logger.info(f"Loaded wc_player_career_stats: {len(df)} rows, {df.player_id.nunique()} players")
    return df


def load_statistics_clean() -> pd.DataFrame:
    """Load current player stats (WC 2026 teams' players, with historical seasons)."""
    path = os.path.join(RAW_DIR, 'player_stats', 'statistics_clean',
                        'all_player_statistics_clean.csv')
    df = pd.read_csv(path)
    df['source'] = 'stats_clean'
    logger.info(f"Loaded statistics_clean: {len(df)} rows, {df.player_id.nunique()} players")
    return df


def merge_player_stats(career: pd.DataFrame, stats_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Merge both player stats sources, deduplicating on (player_id, season, team_id, league_id).
    Prefer wc_career data when duplicates exist (it's the source of truth for WC players).
    """
    combined = pd.concat([career, stats_clean], ignore_index=True)

    # Deduplicate: same player, season, team, league = same record
    # Keep wc_career version first (it's more reliable for historical WC players)
    combined = combined.sort_values('source', ascending=True)  # stats_clean < wc_career alphabetically... 
    # Actually wc_career > stats_clean alphabetically, so ascending puts stats_clean first
    # We want wc_career first, so sort descending
    combined = combined.sort_values('source', ascending=False)
    combined = combined.drop_duplicates(
        subset=['player_id', 'season', 'team_id', 'league_id'],
        keep='first'
    )
    combined = combined.sort_values(['player_id', 'season']).reset_index(drop=True)

    logger.info(f"Merged: {len(combined)} rows, {combined.player_id.nunique()} unique players")
    return combined


def aggregate_player_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a player's stats across all teams/leagues within a single season
    into one row per (player_id, season). This gives us the player's total
    output for that season regardless of which club they played for.
    """
    # Ensure numeric types
    for col in STAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['season'] = df['season'].astype(int)

    # For each player-season, sum counting stats and average rate stats
    # Important: use nansum-like behavior but preserve NaN when ALL values are NaN
    sum_cols = [c for c in STAT_COLS if c in df.columns and c not in ('rating', 'passes_accuracy')]
    avg_cols = ['rating', 'passes_accuracy']

    def sum_or_nan(x):
        """Sum values, but return NaN if all values are NaN."""
        if x.isna().all():
            return np.nan
        return x.sum()

    agg_dict = {}
    for col in sum_cols:
        agg_dict[col] = sum_or_nan
    for col in avg_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'

    # Group by player and season
    grouped = df.groupby(['player_id', 'season']).agg(agg_dict).reset_index()

    # Get metadata (first non-null per player)
    meta_available = [c for c in META_COLS if c in df.columns]
    meta = df.sort_values('season').groupby('player_id')[meta_available].first().reset_index(drop=True)

    # Merge metadata
    result = grouped.merge(meta[['player_id', 'player_name', 'nationality', 'position',
                                  'height_cm', 'weight_kg', 'birth_date',
                                  'wc_years', 'wc_teams']],
                           on='player_id', how='left')

    logger.info(f"Aggregated to {len(result)} player-season rows")
    return result


def compute_derived_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed per-90 and ratio stats."""
    mins = df['minutes'].replace(0, np.nan)

    if 'goals' in df.columns:
        df['goals_per_90'] = (df['goals'] / mins * 90).round(2)
    if 'assists' in df.columns:
        df['assists_per_90'] = (df['assists'] / mins * 90).round(2)
    if 'goals' in df.columns and 'assists' in df.columns:
        df['goal_contributions_per_90'] = ((df['goals'] + df['assists']) / mins * 90).round(2)
    if 'shots_on_target' in df.columns and 'shots_total' in df.columns:
        df['shot_accuracy'] = (df['shots_on_target'] / df['shots_total'].replace(0, np.nan) * 100).round(1)
    if 'duels_won' in df.columns and 'duels_total' in df.columns:
        df['duel_win_pct'] = (df['duels_won'] / df['duels_total'].replace(0, np.nan) * 100).round(1)
    if 'tackles_total' in df.columns:
        df['tackles_per_90'] = (df['tackles_total'] / mins * 90).round(2)
    if 'dribbles_success' in df.columns and 'dribbles_attempts' in df.columns:
        df['dribble_success_pct'] = (df['dribbles_success'] / df['dribbles_attempts'].replace(0, np.nan) * 100).round(1)

    return df


def run():
    """Main pipeline: load, merge, aggregate, compute derived stats, save."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load both data sources
    career = load_career_stats()
    stats_clean = load_statistics_clean()

    # Merge
    merged = merge_player_stats(career, stats_clean)

    # Aggregate per player-season
    agg = aggregate_player_seasons(merged)

    # Compute derived stats
    agg = compute_derived_stats(agg)

    # Save
    out_path = os.path.join(PROCESSED_DIR, 'player_season_stats.csv')
    agg.to_csv(out_path, index=False)
    logger.info(f"Saved aggregated player stats to {out_path}")
    logger.info(f"  {len(agg)} rows, {agg.player_id.nunique()} unique players")
    logger.info(f"  Season range: {agg.season.min()} - {agg.season.max()}")

    return agg


if __name__ == '__main__':
    run()

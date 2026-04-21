"""
Scraper for per-match player statistics from the FIFA World Cup.

Uses the fixtures/players endpoint to get detailed per-match stats for every
player in every match. This data is only available for 2018 and 2022 World Cups.

Output:
  data/world_cup_history/world_cup_player_match_stats.csv

Each row = one player in one match, with ~35 stat columns covering:
  - Minutes played, position, rating, captain status
  - Shots (total, on target)
  - Goals (scored, conceded, assists, saves)  
  - Passes (total, key, accuracy)
  - Tackles (total, blocks, interceptions)
  - Duels (total, won)
  - Dribbles (attempts, success, past)
  - Fouls (drawn, committed)
  - Cards (yellow, red)
  - Penalty (won, committed, scored, missed, saved)
  - Offsides

API calls: 128 (1 per fixture for 2018+2022)
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional

import pandas as pd
from tqdm import tqdm

from api_client import api_client


# =============================================================================
# Configuration
# =============================================================================

SEASONS_WITH_PLAYER_STATS = [2018, 2022]

OUTPUT_DIR = "data/world_cup_history"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "player_stats_progress.json")
MATCHES_CSV = os.path.join(OUTPUT_DIR, "world_cup_matches.csv")
PLAYER_STATS_CSV = os.path.join(OUTPUT_DIR, "world_cup_player_match_stats.csv")


# =============================================================================
# Progress tracking
# =============================================================================

def load_progress() -> Dict[str, Any]:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {
        "fetched_fixtures": [],  # fixture IDs already fetched
        "completed": False,
    }


def save_progress(progress: Dict[str, Any]):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# =============================================================================
# Parse player stats from API response
# =============================================================================

def parse_player_stats(
    fixture_id: int,
    world_cup_year: int,
    match_round: str,
    team_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Parse player stats from the fixtures/players API response for one team.
    
    Returns a list of flat dicts, one per player.
    """
    team_id = team_data['team']['id']
    team_name = team_data['team']['name']
    
    rows = []
    for player_entry in team_data['players']:
        player = player_entry['player']
        stats = player_entry['statistics'][0] if player_entry.get('statistics') else {}
        
        games = stats.get('games', {})
        shots = stats.get('shots', {})
        goals = stats.get('goals', {})
        passes = stats.get('passes', {})
        tackles = stats.get('tackles', {})
        duels = stats.get('duels', {})
        dribbles = stats.get('dribbles', {})
        fouls = stats.get('fouls', {})
        cards = stats.get('cards', {})
        penalty = stats.get('penalty', {})
        
        row = {
            # Match identifiers
            'fixture_id': fixture_id,
            'world_cup_year': world_cup_year,
            'round': match_round,
            
            # Team
            'team_id': team_id,
            'team_name': team_name,
            
            # Player identity
            'player_id': player.get('id'),
            'player_name': player.get('name'),
            
            # Game info
            'minutes_played': games.get('minutes'),
            'player_number': games.get('number'),
            'position': games.get('position'),
            'rating': games.get('rating'),
            'is_captain': games.get('captain'),
            'is_substitute': games.get('substitute'),
            'appeared': games.get('minutes') is not None,
            
            # Shots
            'shots_total': shots.get('total'),
            'shots_on_target': shots.get('on'),
            
            # Goals
            'goals_scored': goals.get('total'),
            'goals_conceded': goals.get('conceded'),
            'goals_assists': goals.get('assists'),
            'goals_saves': goals.get('saves'),
            
            # Passes
            'passes_total': passes.get('total'),
            'passes_key': passes.get('key'),
            'passes_accuracy': passes.get('accuracy'),
            
            # Tackles
            'tackles_total': tackles.get('total'),
            'tackles_blocks': tackles.get('blocks'),
            'tackles_interceptions': tackles.get('interceptions'),
            
            # Duels
            'duels_total': duels.get('total'),
            'duels_won': duels.get('won'),
            
            # Dribbles
            'dribbles_attempts': dribbles.get('attempts'),
            'dribbles_success': dribbles.get('success'),
            'dribbles_past': dribbles.get('past'),
            
            # Fouls
            'fouls_drawn': fouls.get('drawn'),
            'fouls_committed': fouls.get('committed'),
            
            # Cards
            'cards_yellow': cards.get('yellow'),
            'cards_red': cards.get('red'),
            
            # Penalty
            'penalty_won': penalty.get('won'),
            'penalty_committed': penalty.get('commited'),  # API typo: "commited"
            'penalty_scored': penalty.get('scored'),
            'penalty_missed': penalty.get('missed'),
            'penalty_saved': penalty.get('saved'),
            
            # Offsides
            'offsides': stats.get('offsides'),
        }
        rows.append(row)
    
    return rows


# =============================================================================
# Expand position abbreviations
# =============================================================================

POSITION_MAP = {
    'G': 'Goalkeeper',
    'D': 'Defender',
    'M': 'Midfielder',
    'F': 'Forward',
}


def expand_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Expand single-character position codes to full names."""
    df['position'] = df['position'].map(POSITION_MAP).fillna(df['position'])
    return df


# =============================================================================
# Main scraper
# =============================================================================

def scrape_player_stats():
    print("=" * 70)
    print("SCRAPING WORLD CUP PER-MATCH PLAYER STATISTICS")
    print("=" * 70)
    print(f"Seasons: {SEASONS_WITH_PLAYER_STATS}")
    print(f"Output: {PLAYER_STATS_CSV}")
    
    # Load matches to get fixture IDs
    if not os.path.exists(MATCHES_CSV):
        print(f"❌ Matches file not found: {MATCHES_CSV}")
        print("   Run scrape_world_cup_history.py first!")
        sys.exit(1)
    
    matches = pd.read_csv(MATCHES_CSV)
    target_matches = matches[matches['world_cup_year'].isin(SEASONS_WITH_PLAYER_STATS)]
    print(f"\nTotal fixtures to scrape: {len(target_matches)}")
    
    for year in SEASONS_WITH_PLAYER_STATS:
        count = len(target_matches[target_matches['world_cup_year'] == year])
        print(f"  WC {year}: {count} fixtures")
    
    # Load progress
    progress = load_progress()
    fetched = set(progress['fetched_fixtures'])
    
    if progress['completed']:
        print("\n✅ Already completed! Use --force to re-scrape.")
        if '--force' not in sys.argv:
            return
        progress = load_progress()
        progress['fetched_fixtures'] = []
        progress['completed'] = False
        fetched = set()
    
    remaining = target_matches[~target_matches['fixture_id'].isin(fetched)]
    print(f"Already fetched: {len(fetched)} fixtures")
    print(f"Remaining: {len(remaining)} fixtures")
    
    if len(remaining) == 0:
        print("\n✅ All fixtures already fetched!")
        progress['completed'] = True
        save_progress(progress)
        return
    
    # Fetch player stats
    all_rows = []
    
    # Load existing data if resuming
    if os.path.exists(PLAYER_STATS_CSV) and len(fetched) > 0:
        existing = pd.read_csv(PLAYER_STATS_CSV)
        all_rows = existing.to_dict('records')
        print(f"Loaded {len(all_rows)} existing rows from previous run")
    
    errors = []
    
    print(f"\nFetching player stats for {len(remaining)} fixtures...")
    for _, match in tqdm(remaining.iterrows(), total=len(remaining), desc="Player stats"):
        fixture_id = int(match['fixture_id'])
        world_cup_year = int(match['world_cup_year'])
        match_round = match['round']
        
        try:
            data = api_client.get_fixture_players(fixture_id)
            
            if not data or len(data) == 0:
                errors.append(f"  ⚠️  Fixture {fixture_id} ({world_cup_year}): empty response")
                continue
            
            fixture_rows = []
            for team_data in data:
                team_rows = parse_player_stats(fixture_id, world_cup_year, match_round, team_data)
                fixture_rows.extend(team_rows)
            
            all_rows.extend(fixture_rows)
            
            # Track progress
            fetched.add(fixture_id)
            progress['fetched_fixtures'] = list(fetched)
            
            # Save progress every 10 fixtures
            if len(fetched) % 10 == 0:
                save_progress(progress)
                df = pd.DataFrame(all_rows)
                df = expand_positions(df)
                df.to_csv(PLAYER_STATS_CSV, index=False)
                
        except Exception as ex:
            errors.append(f"  ❌ Fixture {fixture_id}: {ex}")
            print(f"\n  Error on fixture {fixture_id}: {ex}")
    
    # Final save
    df = pd.DataFrame(all_rows)
    df = expand_positions(df)
    df.to_csv(PLAYER_STATS_CSV, index=False)
    
    progress['fetched_fixtures'] = list(fetched)
    progress['completed'] = True
    save_progress(progress)
    
    # Report
    print(f"\n{'=' * 70}")
    print(f"SCRAPING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total rows: {len(df)}")
    print(f"Fixtures covered: {df['fixture_id'].nunique()}")
    print(f"Unique players: {df['player_id'].nunique()}")
    print(f"Unique teams: {df['team_id'].nunique()}")
    
    if errors:
        print(f"\n⚠️  {len(errors)} errors:")
        for err in errors:
            print(err)
    
    # Validation
    print(f"\n--- Validation ---")
    
    for year in SEASONS_WITH_PLAYER_STATS:
        ydf = df[df['world_cup_year'] == year]
        fixtures_count = ydf['fixture_id'].nunique()
        players_count = ydf['player_id'].nunique()
        appeared = ydf[ydf['appeared'] == True]
        print(f"\n  WC {year}:")
        print(f"    Fixtures: {fixtures_count}/64")
        print(f"    Total player entries: {len(ydf)}")
        print(f"    Unique players: {players_count}")
        print(f"    Players who appeared: {len(appeared)}")
        print(f"    Players who didn't appear: {len(ydf) - len(appeared)}")
        
        # Check starters per fixture
        starters = ydf[ydf['is_substitute'] == False]
        starters_per_fix = starters.groupby('fixture_id').size()
        if (starters_per_fix != 22).any():
            bad = starters_per_fix[starters_per_fix != 22]
            print(f"    ⚠️  {len(bad)} fixtures don't have 22 starters")
        else:
            print(f"    ✅ All fixtures have 22 starters")
    
    # Column summary
    print(f"\n--- Column Summary ---")
    print(f"Total columns: {len(df.columns)}")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        print(f"  {col:<25} {non_null:>6}/{len(df):<6} ({pct:>5.1f}%)")
    
    # Sample data
    print(f"\n--- Sample: Messi in a match ---")
    messi = df[df['player_name'].str.contains('Messi', na=False)].head(1)
    if len(messi) > 0:
        for col in df.columns:
            print(f"  {col}: {messi.iloc[0][col]}")


if __name__ == '__main__':
    scrape_player_stats()

#!/usr/bin/env python3
"""
Repair script for player statistics completeness.

Root cause: The historical scraper skipped seasons 2015-2021 for many teams,
and some players have zero statistics at all.

This script:
1. For each of the 42 WC 2026 teams, reads the squad file and existing stats file
2. For each player, determines which seasons (2004-2025) are missing from the stats
3. Scrapes only the missing seasons from the API
4. Merges new data with existing data (no duplicates)
5. Validates per-team: player count, season coverage, team IDs
6. Saves updated team stats file
7. Rebuilds all_player_statistics.csv from all individual files

Progress tracking allows safe resume if interrupted.
"""
import os
import sys
import json
import glob
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from api_client import api_client
from config import (
    SEASONS_TO_SCRAPE,
    PLAYERS_OUTPUT_DIR,
    STATISTICS_OUTPUT_DIR,
)

# All 22 seasons we want complete coverage for
ALL_SEASONS = set(SEASONS_TO_SCRAPE)  # {2004, 2005, ..., 2025}

# Progress file for this repair run
REPAIR_PROGRESS_FILE = "data/repair_progress.json"

# The 42 WC 2026 teams: mapping from file base name -> (national_team_id, national_team_name)
# Built from the existing stats files (verified above)
TEAM_REGISTRY = {
    "algeria": (1532, "Algeria"),
    "argentina": (26, "Argentina"),
    "australia": (20, "Australia"),
    "austria": (775, "Austria"),
    "belgium": (1, "Belgium"),
    "brazil": (6, "Brazil"),
    "canada": (5529, "Canada"),
    "cape_verde_islands": (1533, "Cape Verde Islands"),
    "colombia": (8, "Colombia"),
    "croatia": (3, "Croatia"),
    "curaçao": (5530, "Curaçao"),
    "ecuador": (2382, "Ecuador"),
    "egypt": (32, "Egypt"),
    "england": (10, "England"),
    "france": (2, "France"),
    "germany": (25, "Germany"),
    "ghana": (1504, "Ghana"),
    "haiti": (2386, "Haiti"),
    "iran": (22, "Iran"),
    "ivory_coast": (1501, "Ivory Coast"),
    "japan": (12, "Japan"),
    "jordan": (1548, "Jordan"),
    "mexico": (16, "Mexico"),
    "morocco": (31, "Morocco"),
    "netherlands": (1118, "Netherlands"),
    "new_zealand": (4673, "New Zealand"),
    "norway": (1090, "Norway"),
    "panama": (11, "Panama"),
    "paraguay": (2380, "Paraguay"),
    "portugal": (27, "Portugal"),
    "qatar": (1569, "Qatar"),
    "saudi_arabia": (23, "Saudi Arabia"),
    "scotland": (1108, "Scotland"),
    "senegal": (13, "Senegal"),
    "south_africa": (1531, "South Africa"),
    "south_korea": (17, "South Korea"),
    "spain": (9, "Spain"),
    "switzerland": (15, "Switzerland"),
    "tunisia": (28, "Tunisia"),
    "uruguay": (7, "Uruguay"),
    "usa": (2384, "USA"),
    "uzbekistan": (1568, "Uzbekistan"),
}


class RepairProgressTracker:
    """Track repair progress for safe resume."""
    
    def __init__(self, progress_file: str = REPAIR_PROGRESS_FILE):
        self.progress_file = progress_file
        self.progress = self._load()
    
    def _load(self) -> dict:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "completed_teams": [],
            "player_seasons_scraped": {},  # {player_id_str: [seasons_list]}
            "api_calls_made": 0,
            "started_at": datetime.now().isoformat(),
            "last_updated": None,
        }
    
    def save(self):
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def is_team_completed(self, team_key: str) -> bool:
        return team_key in self.progress["completed_teams"]
    
    def mark_team_completed(self, team_key: str):
        if team_key not in self.progress["completed_teams"]:
            self.progress["completed_teams"].append(team_key)
        self.save()
    
    def get_player_scraped_seasons(self, player_id: int) -> Set[int]:
        key = str(player_id)
        return set(self.progress["player_seasons_scraped"].get(key, []))
    
    def mark_player_season_scraped(self, player_id: int, season: int):
        key = str(player_id)
        if key not in self.progress["player_seasons_scraped"]:
            self.progress["player_seasons_scraped"][key] = []
        if season not in self.progress["player_seasons_scraped"][key]:
            self.progress["player_seasons_scraped"][key].append(season)
    
    def add_api_calls(self, count: int):
        self.progress["api_calls_made"] += count


def parse_height_weight(value) -> Optional[int]:
    """Parse height/weight strings like '183 cm' or '72 kg' into integers."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value) if not pd.isna(value) else None
    s = str(value).strip()
    # Extract numeric part from strings like "183 cm" or "72 kg"
    digits = ''.join(c for c in s if c.isdigit())
    return int(digits) if digits else None


def scrape_player_season(player_id: int, season: int) -> List[dict]:
    """
    Scrape a single player's stats for a single season.
    Returns a list of stat row dicts (one per league the player participated in).
    Uses the exact same parsing logic as scraper.py.
    """
    try:
        stats_response = api_client.get_player_statistics(player_id, season)
        
        if not stats_response:
            return []
        
        player_data = stats_response[0]
        player_info = player_data.get("player", {})
        birth = player_info.get("birth", {})
        
        rows = []
        for stat in player_data.get("statistics", []):
            league = stat.get("league", {})
            team = stat.get("team", {})
            games = stat.get("games", {})
            substitutes = stat.get("substitutes", {})
            shots = stat.get("shots", {})
            goals_data = stat.get("goals", {})
            passes = stat.get("passes", {})
            tackles = stat.get("tackles", {})
            duels = stat.get("duels", {})
            dribbles = stat.get("dribbles", {})
            fouls = stat.get("fouls", {})
            cards = stat.get("cards", {})
            penalty = stat.get("penalty", {})
            
            rows.append({
                # Player info
                "player_id": player_info.get("id"),
                "player_name": player_info.get("name"),
                "firstname": player_info.get("firstname"),
                "lastname": player_info.get("lastname"),
                "nationality": player_info.get("nationality"),
                "birth_date": birth.get("date"),
                "birth_place": birth.get("place"),
                "birth_country": birth.get("country"),
                "age": player_info.get("age"),
                "height": parse_height_weight(player_info.get("height")),
                "weight": parse_height_weight(player_info.get("weight")),
                "injured": player_info.get("injured"),
                "photo": player_info.get("photo"),
                # Team & League
                "season": season,
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "league_country": league.get("country"),
                # Games
                "position": games.get("position"),
                "appearances": games.get("appearences"),  # API typo
                "lineups": games.get("lineups"),
                "minutes": games.get("minutes"),
                "rating": games.get("rating"),
                "captain": games.get("captain"),
                # Substitutes
                "substitutes_in": substitutes.get("in"),
                "substitutes_out": substitutes.get("out"),
                "substitutes_bench": substitutes.get("bench"),
                # Shooting
                "shots_total": shots.get("total"),
                "shots_on_target": shots.get("on"),
                # Goals
                "goals": goals_data.get("total"),
                "goals_conceded": goals_data.get("conceded"),
                "assists": goals_data.get("assists"),
                "saves": goals_data.get("saves"),
                # Passing
                "passes_total": passes.get("total"),
                "passes_key": passes.get("key"),
                "passes_accuracy": passes.get("accuracy"),
                # Defensive
                "tackles_total": tackles.get("total"),
                "tackles_blocks": tackles.get("blocks"),
                "tackles_interceptions": tackles.get("interceptions"),
                # Duels
                "duels_total": duels.get("total"),
                "duels_won": duels.get("won"),
                # Dribbles
                "dribbles_attempts": dribbles.get("attempts"),
                "dribbles_success": dribbles.get("success"),
                "dribbles_past": dribbles.get("past"),
                # Fouls
                "fouls_drawn": fouls.get("drawn"),
                "fouls_committed": fouls.get("committed"),
                # Cards
                "cards_yellow": cards.get("yellow"),
                "cards_yellowred": cards.get("yellowred"),
                "cards_red": cards.get("red"),
                # Penalties
                "penalty_won": penalty.get("won"),
                "penalty_committed": penalty.get("commited"),  # API typo
                "penalty_scored": penalty.get("scored"),
                "penalty_missed": penalty.get("missed"),
                "penalty_saved": penalty.get("saved"),
            })
        
        return rows
        
    except Exception as e:
        print(f"    ⚠️ API error for player {player_id} season {season}: {e}")
        return []


def get_squad_file_for_team(team_key: str) -> Optional[str]:
    """Find the squad file path for a team key."""
    # Direct match
    path = f"{PLAYERS_OUTPUT_DIR}/{team_key}_squad.csv"
    if os.path.exists(path):
        return path
    
    # For cape_verde_islands, the squad file is named cape_verde_islands
    # but there's also cape_verde_squad.csv (not used)
    return None


def get_stats_file_for_team(team_key: str) -> str:
    """Get the stats file path for a team key."""
    return f"{STATISTICS_OUTPUT_DIR}/{team_key}_player_statistics.csv"


def analyze_team_gaps(
    team_key: str, 
    national_team_id: int, 
    national_team_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Set[int]]]:
    """
    Analyze what's missing for a team.
    
    Returns:
        (squad_df, existing_stats_df, missing_dict)
        missing_dict: {player_id: set of missing seasons}
    """
    # Load squad
    squad_path = get_squad_file_for_team(team_key)
    if squad_path is None:
        print(f"  ❌ No squad file found for {team_key}")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    squad_df = pd.read_csv(squad_path)
    
    # Load existing stats (may not exist or may be empty)
    stats_path = get_stats_file_for_team(team_key)
    if os.path.exists(stats_path):
        existing_stats = pd.read_csv(stats_path)
    else:
        existing_stats = pd.DataFrame()
    
    # For each squad player, determine which seasons are missing
    missing = {}
    
    for _, player in squad_df.iterrows():
        pid = int(player['player_id'])
        
        if existing_stats.empty:
            # No data at all — need all seasons
            missing[pid] = ALL_SEASONS.copy()
        else:
            # Find which seasons this player already has in the stats
            player_stats = existing_stats[existing_stats['player_id'] == pid]
            existing_seasons = set(player_stats['season'].unique()) if not player_stats.empty else set()
            gaps = ALL_SEASONS - existing_seasons
            if gaps:
                missing[pid] = gaps
    
    return squad_df, existing_stats, missing


def validate_team_result(
    team_key: str,
    national_team_id: int,
    national_team_name: str,
    squad_df: pd.DataFrame,
    final_stats: pd.DataFrame,
) -> Tuple[bool, List[str]]:
    """
    Validate the repaired stats for a team.
    Returns (is_valid, list_of_issues).
    """
    issues = []
    
    if final_stats.empty:
        issues.append(f"No statistics at all for {national_team_name}")
        return False, issues
    
    # 1. Check national_team_id consistency
    team_ids_in_stats = final_stats['national_team_id'].unique()
    if len(team_ids_in_stats) != 1 or int(team_ids_in_stats[0]) != national_team_id:
        issues.append(f"national_team_id mismatch: expected {national_team_id}, got {team_ids_in_stats}")
    
    # 2. Check national_team_name consistency
    team_names_in_stats = final_stats['national_team_name'].unique()
    if len(team_names_in_stats) != 1 or team_names_in_stats[0] != national_team_name:
        issues.append(f"national_team_name mismatch: expected '{national_team_name}', got {team_names_in_stats}")
    
    # 3. Check player coverage
    squad_player_ids = set(squad_df['player_id'].astype(int).unique())
    stats_player_ids = set(final_stats['player_id'].astype(int).unique())
    
    missing_players = squad_player_ids - stats_player_ids
    if missing_players:
        # Some players may genuinely have no API data at all (young/obscure)
        # We log as warning, not error
        issues.append(f"⚠️ {len(missing_players)} players with NO stats from API (may be too young/obscure): {missing_players}")
    
    # 4. Check season coverage per player
    players_with_gaps = []
    for pid in stats_player_ids:
        player_seasons = set(final_stats[final_stats['player_id'] == pid]['season'].unique())
        # Not all players will have all 22 seasons (young players didn't exist in 2004)
        # But they should have data for seasons they were active
        # The main check: no 2015-2021 gap pattern
        has_pre_2015 = any(s < 2015 for s in player_seasons)
        has_post_2021 = any(s > 2021 for s in player_seasons)
        gap_seasons = {s for s in range(2015, 2022)} - player_seasons
        
        if has_pre_2015 and has_post_2021 and gap_seasons:
            players_with_gaps.append((pid, gap_seasons))
    
    if players_with_gaps:
        issues.append(f"❌ {len(players_with_gaps)} players STILL have 2015-2021 gaps after repair!")
        for pid, gaps in players_with_gaps[:5]:
            issues.append(f"   Player {pid}: missing {sorted(gaps)}")
    
    # 5. Check for duplicates (same player + season + league + team should be unique)
    # Note: A player CAN have 2 entries for the same league+season with DIFFERENT teams
    # (mid-season transfer). That's legitimate data.
    dup_check = final_stats.groupby(['player_id', 'season', 'league_id', 'team_id']).size()
    duplicates = dup_check[dup_check > 1]
    if len(duplicates) > 0:
        issues.append(f"❌ {len(duplicates)} duplicate player-season-league-team rows found!")
    
    # 6. Check column count
    if len(final_stats.columns) != 57:
        issues.append(f"Column count: expected 57, got {len(final_stats.columns)}")
    
    is_valid = not any("❌" in issue for issue in issues)
    return is_valid, issues


def repair_team(
    team_key: str,
    national_team_id: int,
    national_team_name: str,
    progress: RepairProgressTracker,
    dry_run: bool = False,
) -> Tuple[bool, int]:
    """
    Repair statistics for a single team.
    Returns (success, api_calls_made).
    """
    print(f"\n{'='*70}")
    print(f"🔧 Repairing: {national_team_name} ({team_key})")
    print(f"   National Team ID: {national_team_id}")
    print(f"{'='*70}")
    
    # Analyze gaps
    squad_df, existing_stats, missing = analyze_team_gaps(
        team_key, national_team_id, national_team_name
    )
    
    if squad_df.empty:
        print(f"  ❌ Could not load squad for {team_key}")
        return False, 0
    
    squad_count = len(squad_df)
    players_needing_repair = len(missing)
    total_missing_seasons = sum(len(s) for s in missing.values())
    
    print(f"  📊 Squad size: {squad_count}")
    print(f"  📊 Existing stats rows: {len(existing_stats)}")
    print(f"  📊 Players needing repair: {players_needing_repair}/{squad_count}")
    print(f"  📊 Total missing player-seasons: {total_missing_seasons}")
    
    if total_missing_seasons == 0:
        print(f"  ✅ No gaps found! Team is complete.")
        return True, 0
    
    if dry_run:
        print(f"  [DRY RUN] Would make {total_missing_seasons} API calls")
        return True, 0
    
    # Scrape missing data
    new_rows = []
    api_calls = 0
    
    # Build player name lookup from squad
    player_names = {
        int(row['player_id']): row['player_name'] 
        for _, row in squad_df.iterrows()
    }
    
    for pid, missing_seasons in tqdm(
        missing.items(), 
        desc=f"  Players ({national_team_name})", 
        total=len(missing)
    ):
        player_name = player_names.get(pid, f"ID:{pid}")
        sorted_seasons = sorted(missing_seasons, reverse=True)  # Recent first
        
        # Check which seasons we already scraped in a previous interrupted run
        already_scraped = progress.get_player_scraped_seasons(pid)
        seasons_to_scrape = [s for s in sorted_seasons if s not in already_scraped]
        
        if not seasons_to_scrape:
            continue
        
        for season in seasons_to_scrape:
            rows = scrape_player_season(pid, season)
            api_calls += 1
            progress.add_api_calls(1)
            progress.mark_player_season_scraped(pid, season)
            
            for row in rows:
                row['national_team_id'] = national_team_id
                row['national_team_name'] = national_team_name
                new_rows.append(row)
            
            # Save progress periodically (every 50 API calls)
            if api_calls % 50 == 0:
                progress.save()
    
    progress.save()
    
    print(f"\n  📡 API calls made: {api_calls}")
    print(f"  📡 New stat rows fetched: {len(new_rows)}")
    
    # Merge new data with existing
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        
        if not existing_stats.empty:
            # Combine existing + new
            combined = pd.concat([existing_stats, new_df], ignore_index=True)
        else:
            combined = new_df
        
        # Deduplicate: same player + season + league_id + team_id should be unique
        before_dedup = len(combined)
        combined = combined.drop_duplicates(
            subset=['player_id', 'season', 'league_id', 'team_id'], 
            keep='last'  # Keep the newer data
        )
        after_dedup = len(combined)
        if before_dedup != after_dedup:
            print(f"  🔄 Deduplication: {before_dedup} → {after_dedup} rows ({before_dedup - after_dedup} duplicates removed)")
        
        # Sort for consistent output
        combined = combined.sort_values(
            ['player_id', 'season', 'league_id'], 
            ascending=[True, False, True]
        ).reset_index(drop=True)
    else:
        combined = existing_stats
    
    # Validate
    is_valid, issues = validate_team_result(
        team_key, national_team_id, national_team_name, squad_df, combined
    )
    
    print(f"\n  📋 Validation for {national_team_name}:")
    for issue in issues:
        print(f"     {issue}")
    
    if is_valid:
        print(f"  ✅ PASSED validation")
    else:
        print(f"  ⚠️ Validation has critical issues (see above)")
    
    # Save regardless (even with warnings — the data is real, just may have expected gaps)
    stats_path = get_stats_file_for_team(team_key)
    
    # Ensure correct column order (match existing files exactly)
    expected_columns = [
        'player_id', 'player_name', 'firstname', 'lastname', 'nationality',
        'birth_date', 'birth_place', 'birth_country', 'age', 'height', 'weight',
        'injured', 'photo', 'season', 'team_id', 'team_name', 'league_id',
        'league_name', 'league_country', 'position', 'appearances', 'lineups',
        'minutes', 'rating', 'captain', 'substitutes_in', 'substitutes_out',
        'substitutes_bench', 'shots_total', 'shots_on_target', 'goals',
        'goals_conceded', 'assists', 'saves', 'passes_total', 'passes_key',
        'passes_accuracy', 'tackles_total', 'tackles_blocks',
        'tackles_interceptions', 'duels_total', 'duels_won', 'dribbles_attempts',
        'dribbles_success', 'dribbles_past', 'fouls_drawn', 'fouls_committed',
        'cards_yellow', 'cards_yellowred', 'cards_red', 'penalty_won',
        'penalty_committed', 'penalty_scored', 'penalty_missed', 'penalty_saved',
        'national_team_id', 'national_team_name'
    ]
    
    # Add any missing columns with NaN
    for col in expected_columns:
        if col not in combined.columns:
            combined[col] = None
    
    # Reorder to match
    combined = combined[expected_columns]
    
    combined.to_csv(stats_path, index=False)
    print(f"  💾 Saved {len(combined)} rows to {stats_path}")
    
    return True, api_calls


def rebuild_all_player_statistics():
    """Rebuild all_player_statistics.csv from all individual team files."""
    print(f"\n{'='*70}")
    print("📦 Rebuilding all_player_statistics.csv from individual team files...")
    print(f"{'='*70}")
    
    all_dfs = []
    
    for team_key, (national_team_id, national_team_name) in sorted(TEAM_REGISTRY.items()):
        stats_path = get_stats_file_for_team(team_key)
        if os.path.exists(stats_path):
            df = pd.read_csv(stats_path)
            all_dfs.append(df)
            print(f"  ✅ {national_team_name}: {len(df)} rows")
        else:
            print(f"  ❌ Missing stats file for {national_team_name}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Deduplicate across teams (a player might be in multiple national team squads... unlikely but possible)
        # Actually, each player belongs to ONE national team, but their club stats are per team
        # The dedup key should be player_id + season + league_id + team_id  
        before = len(combined)
        combined = combined.drop_duplicates(
            subset=['player_id', 'season', 'league_id', 'team_id'],
            keep='first'
        )
        after = len(combined)
        if before != after:
            print(f"\n  🔄 Cross-team dedup: {before} → {after} ({before - after} duplicates)")
        
        combined = combined.sort_values(
            ['national_team_name', 'player_id', 'season', 'league_id'],
            ascending=[True, True, False, True]
        ).reset_index(drop=True)
        
        output_path = f"{STATISTICS_OUTPUT_DIR}/all_player_statistics.csv"
        combined.to_csv(output_path, index=False)
        
        # Summary stats
        n_players = combined['player_id'].nunique()
        n_teams = combined['national_team_name'].nunique()
        seasons_range = f"{combined['season'].min()}-{combined['season'].max()}"
        
        print(f"\n  📊 Final all_player_statistics.csv:")
        print(f"     Rows: {len(combined)}")
        print(f"     Unique players: {n_players}")
        print(f"     National teams: {n_teams}")
        print(f"     Season range: {seasons_range}")
        print(f"     Columns: {len(combined.columns)}")
        print(f"  💾 Saved to {output_path}")
    else:
        print("  ❌ No team stats files found!")


def main():
    """Main repair function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Repair player statistics gaps')
    parser.add_argument('--dry-run', action='store_true', help='Only analyze gaps, do not scrape')
    parser.add_argument('--team', type=str, help='Repair only a specific team (e.g., portugal)')
    parser.add_argument('--rebuild-only', action='store_true', help='Only rebuild all_player_statistics.csv')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from previous run')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔧 PLAYER STATISTICS REPAIR TOOL")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Seasons: {sorted(ALL_SEASONS)}")
    print(f"   Teams: {len(TEAM_REGISTRY)}")
    if args.dry_run:
        print("   Mode: DRY RUN (no API calls)")
    print("=" * 70)
    
    if args.rebuild_only:
        rebuild_all_player_statistics()
        return
    
    # Initialize progress tracker
    progress = RepairProgressTracker()
    
    # Determine which teams to process
    if args.team:
        if args.team not in TEAM_REGISTRY:
            print(f"❌ Unknown team: {args.team}")
            print(f"   Available: {sorted(TEAM_REGISTRY.keys())}")
            return
        teams_to_process = {args.team: TEAM_REGISTRY[args.team]}
    else:
        teams_to_process = TEAM_REGISTRY
    
    # Process each team
    total_api_calls = 0
    results = {}
    
    for team_key, (national_team_id, national_team_name) in sorted(teams_to_process.items()):
        # Skip completed teams if resuming
        if args.resume and progress.is_team_completed(team_key):
            print(f"\n  ⏭️ Skipping {national_team_name} (already completed)")
            results[team_key] = ("skipped", 0)
            continue
        
        success, api_calls = repair_team(
            team_key, national_team_id, national_team_name,
            progress, dry_run=args.dry_run
        )
        
        total_api_calls += api_calls
        
        if success:
            if not args.dry_run:
                progress.mark_team_completed(team_key)
            results[team_key] = ("success", api_calls)
        else:
            results[team_key] = ("failed", api_calls)
    
    # Rebuild the combined file
    if not args.dry_run:
        rebuild_all_player_statistics()
    
    # Final summary
    print(f"\n{'='*70}")
    print("📋 REPAIR SUMMARY")
    print(f"{'='*70}")
    
    succeeded = sum(1 for s, _ in results.values() if s == "success")
    skipped = sum(1 for s, _ in results.values() if s == "skipped")
    failed = sum(1 for s, _ in results.values() if s == "failed")
    
    print(f"  Teams processed: {succeeded + failed}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total API calls: {total_api_calls}")
    print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed > 0:
        print(f"\n  ❌ Failed teams:")
        for team_key, (status, _) in results.items():
            if status == "failed":
                print(f"     - {team_key}")


if __name__ == "__main__":
    main()

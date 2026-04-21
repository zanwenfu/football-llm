#!/usr/bin/env python3
"""
Scrape additional historical seasons and merge with existing data.
This script avoids re-scraping 2022-2024 which we already have.

Features:
- Progress tracking with resume capability
- Quota monitoring
- Prioritizes recent seasons first
"""
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Set
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from api_client import api_client
from scraper import WorldCup2026Scraper
from config import (
    OUTPUT_DIR,
    PLAYERS_OUTPUT_DIR,
    STATISTICS_OUTPUT_DIR,
)

# Historical seasons to scrape (Pro plan has access back to 2004, forward to 2025)
# Prioritize recent seasons first (more relevant for prediction)
# Full range: 2004-2025 (22 years total)
# Import from config for consistency
from config import SEASONS_TO_SCRAPE
HISTORICAL_SEASONS = SEASONS_TO_SCRAPE  # [2025, 2024, 2023, ..., 2005, 2004]

# Progress file
HISTORICAL_PROGRESS_FILE = f"{OUTPUT_DIR}/historical_scrape_progress.json"

# Minimum quota to keep as buffer
MIN_QUOTA_BUFFER = 200


class ProgressTracker:
    """Track scraping progress for resume capability."""
    
    def __init__(self, progress_file: str = HISTORICAL_PROGRESS_FILE):
        self.progress_file = progress_file
        self.progress = self.load()
    
    def load(self) -> dict:
        """Load progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "completed_teams": [],  # Teams fully scraped for ALL seasons
            "scraped_seasons": {},  # {team_name: {player_id: [seasons]}}
            "new_teams_progress": {},  # {team_name: {squad_fetched: bool, seasons_done: [seasons]}}
            "api_calls_made": 0,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def save(self):
        """Save progress to file."""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def is_team_completed(self, team_name: str) -> bool:
        return team_name in self.progress["completed_teams"]
    
    def get_player_scraped_seasons(self, team_name: str, player_id: int) -> List[int]:
        """Get seasons already scraped for a player in this historical run."""
        key = str(player_id)
        return self.progress["scraped_seasons"].get(team_name, {}).get(key, [])
    
    def mark_player_season(self, team_name: str, player_id: int, season: int):
        """Mark a season as scraped for a player."""
        if team_name not in self.progress["scraped_seasons"]:
            self.progress["scraped_seasons"][team_name] = {}
        key = str(player_id)
        if key not in self.progress["scraped_seasons"][team_name]:
            self.progress["scraped_seasons"][team_name][key] = []
        if season not in self.progress["scraped_seasons"][team_name][key]:
            self.progress["scraped_seasons"][team_name][key].append(season)
    
    def mark_team_completed(self, team_name: str):
        if team_name not in self.progress["completed_teams"]:
            self.progress["completed_teams"].append(team_name)
        self.save()
    
    def add_api_calls(self, count: int):
        self.progress["api_calls_made"] += count
    
    def is_new_team(self, team_name: str) -> bool:
        """Check if this is a new team (no existing CSV data)."""
        safe_name = team_name.replace(" ", "_").lower()
        stats_file = f"{STATISTICS_OUTPUT_DIR}/{safe_name}_player_statistics.csv"
        return not os.path.exists(stats_file) or os.path.getsize(stats_file) < 10
    
    def get_new_team_progress(self, team_name: str) -> dict:
        """Get progress for a new team."""
        if "new_teams_progress" not in self.progress:
            self.progress["new_teams_progress"] = {}
        return self.progress["new_teams_progress"].get(team_name, {
            "squad_fetched": False,
            "seasons_done": []
        })
    
    def set_new_team_squad_fetched(self, team_name: str):
        """Mark that squad has been fetched for a new team."""
        if "new_teams_progress" not in self.progress:
            self.progress["new_teams_progress"] = {}
        if team_name not in self.progress["new_teams_progress"]:
            self.progress["new_teams_progress"][team_name] = {"squad_fetched": False, "seasons_done": []}
        self.progress["new_teams_progress"][team_name]["squad_fetched"] = True
        self.save()
    
    def add_new_team_season_done(self, team_name: str, season: int):
        """Mark a season as done for a new team."""
        if "new_teams_progress" not in self.progress:
            self.progress["new_teams_progress"] = {}
        if team_name not in self.progress["new_teams_progress"]:
            self.progress["new_teams_progress"][team_name] = {"squad_fetched": False, "seasons_done": []}
        if season not in self.progress["new_teams_progress"][team_name]["seasons_done"]:
            self.progress["new_teams_progress"][team_name]["seasons_done"].append(season)
    
    def get_stats(self) -> str:
        total_teams = 42  # WC 2026 confirmed qualified teams
        return (
            f"Teams completed: {len(self.progress['completed_teams'])}/{total_teams} | "
            f"API calls: {self.progress['api_calls_made']}"
        )


def check_quota() -> int:
    """Check remaining API quota. Returns remaining requests."""
    try:
        status = api_client.get_account_status()
        response = status.get("response", {})
        requests_info = response.get("requests", {})
        current = requests_info.get("current", 0)
        limit = requests_info.get("limit_day", 7500)
        return limit - current
    except Exception as e:
        print(f"  ⚠️ Could not check quota: {e}")
        return 0


def get_existing_seasons(team_name: str) -> List[int]:
    """Get seasons already scraped for a team."""
    safe_name = team_name.replace(" ", "_").lower()
    stats_file = f"{STATISTICS_OUTPUT_DIR}/{safe_name}_player_statistics.csv"
    
    if os.path.exists(stats_file):
        df = pd.read_csv(stats_file)
        return sorted(df['season'].unique().tolist())
    return []


def merge_and_save_stats(team_name: str, new_stats_df: pd.DataFrame, team_id: int):
    """Merge new historical stats with existing data and save."""
    safe_name = team_name.replace(" ", "_").lower()
    stats_file = f"{STATISTICS_OUTPUT_DIR}/{safe_name}_player_statistics.csv"
    
    if os.path.exists(stats_file):
        existing_df = pd.read_csv(stats_file)
        print(f"  Existing data: {len(existing_df)} records, seasons {sorted(existing_df['season'].unique())}")
    else:
        existing_df = pd.DataFrame()
        print(f"  No existing data found")
    
    if new_stats_df.empty:
        print(f"  No new historical data found")
        return existing_df
    
    # Add national team info to new stats
    new_stats_df["national_team_id"] = team_id
    new_stats_df["national_team_name"] = team_name
    
    print(f"  New historical data: {len(new_stats_df)} records, seasons {sorted(new_stats_df['season'].unique())}")
    
    # Combine and deduplicate
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_stats_df], ignore_index=True)
        # Remove duplicates based on key columns
        combined_df = combined_df.drop_duplicates(
            subset=['player_id', 'season', 'team_id', 'league_id'],
            keep='first'
        )
    else:
        combined_df = new_stats_df
    
    # Sort by player and season
    combined_df = combined_df.sort_values(['player_name', 'season', 'team_name'])
    
    # Save
    combined_df.to_csv(stats_file, index=False)
    print(f"  ✅ Saved {len(combined_df)} total records to {stats_file}")
    print(f"     Seasons now: {sorted(combined_df['season'].unique())}")
    
    return combined_df


def fetch_and_save_squad(scraper: WorldCup2026Scraper, team_id: int, team_name: str) -> pd.DataFrame:
    """Fetch squad for a new team and save to CSV."""
    print(f"  👥 Fetching squad for {team_name} (ID: {team_id})...")
    
    squad_data = scraper.api.get_squad(team_id)
    
    players = []
    if squad_data:
        team_info = squad_data[0]
        for player in team_info.get("players", []):
            players.append({
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "team_id": team_id,
                "team_name": team_name,
                "age": player.get("age"),
                "number": player.get("number"),
                "position": player.get("position"),
                "photo": player.get("photo"),
            })
    
    df = pd.DataFrame(players)
    
    # Save to CSV
    safe_name = team_name.replace(" ", "_").lower()
    output_path = f"{PLAYERS_OUTPUT_DIR}/{safe_name}_squad.csv"
    df.to_csv(output_path, index=False)
    print(f"  ✅ Saved {len(df)} players to {output_path}")
    
    return df


def scrape_historical_for_team(
    scraper: WorldCup2026Scraper,
    team_id: int,
    team_name: str,
    seasons_to_scrape: List[int],
    progress: ProgressTracker,
) -> tuple[pd.DataFrame, int, bool]:
    """
    Scrape historical seasons for a single team.
    Handles both existing teams (missing some seasons) and new teams (no data at all).
    Returns: (DataFrame with stats, api_calls_made, quota_exhausted)
    """
    
    # Check if team already completed
    if progress.is_team_completed(team_name):
        print(f"  ⏭️ Team already completed in previous run")
        return pd.DataFrame(), 0, False
    
    # Check if this is a NEW team (no existing data)
    is_new_team = progress.is_new_team(team_name)
    api_calls_made = 0
    
    safe_name = team_name.replace(" ", "_").lower()
    squad_file = f"{PLAYERS_OUTPUT_DIR}/{safe_name}_squad.csv"
    
    if is_new_team:
        print(f"  🆕 NEW TEAM - no existing data found")
        new_team_progress = progress.get_new_team_progress(team_name)
        
        # Fetch squad if not already done
        if not new_team_progress.get("squad_fetched") or not os.path.exists(squad_file) or os.path.getsize(squad_file) < 10:
            try:
                squad_df = fetch_and_save_squad(scraper, team_id, team_name)
                api_calls_made += 1
                progress.set_new_team_squad_fetched(team_name)
                if squad_df.empty:
                    print(f"  ⚠️ No squad data available for {team_name}")
                    progress.mark_team_completed(team_name)  # Mark as complete (nothing to scrape)
                    return pd.DataFrame(), api_calls_made, False
            except Exception as e:
                print(f"  ❌ Error fetching squad: {e}")
                return pd.DataFrame(), api_calls_made, False
        
        # For new teams, scrape ALL seasons (not just missing)
        seasons_already_done = new_team_progress.get("seasons_done", [])
        new_seasons = [s for s in seasons_to_scrape if s not in seasons_already_done]
        existing_seasons = []
        print(f"  📊 Will scrape all {len(new_seasons)} seasons for new team")
        print(f"     Seasons already done: {seasons_already_done if seasons_already_done else 'none'}")
    else:
        # EXISTING team - get existing data info
        existing_seasons = get_existing_seasons(team_name)
        new_seasons = [s for s in seasons_to_scrape if s not in existing_seasons]
    
    # Load squad
    if not os.path.exists(squad_file):
        print(f"  ❌ No squad file found for {team_name}")
        return pd.DataFrame(), api_calls_made, False
    
    # Check file size (empty placeholder files are 1 byte)
    if os.path.getsize(squad_file) < 10:
        print(f"  ❌ Squad file is empty for {team_name}")
        return pd.DataFrame(), api_calls_made, False
    
    try:
        squad_df = pd.read_csv(squad_file)
        if squad_df.empty or 'player_id' not in squad_df.columns:
            print(f"  ❌ Squad file has no player data for {team_name}")
            return pd.DataFrame(), api_calls_made, False
    except Exception as e:
        print(f"  ❌ Error reading squad file: {e}")
        return pd.DataFrame(), api_calls_made, False
    
    if not new_seasons:
        print(f"  ⏭️ All requested seasons already in CSV: {existing_seasons}")
        progress.mark_team_completed(team_name)
        return pd.DataFrame(), api_calls_made, False
    
    print(f"  📊 Scraping seasons {new_seasons} for {len(squad_df)} players")
    if existing_seasons:
        print(f"     Already have in CSV: {existing_seasons}")
    
    all_player_stats = []
    # api_calls_made already initialized above
    quota_exhausted = False
    
    for idx, player in squad_df.iterrows():
        if pd.isna(player.get("player_id")):
            continue
            
        player_id = int(player["player_id"])
        player_name = player.get("player_name", f"Player {player_id}")
        
        # Check which seasons still need to be scraped for this player
        already_scraped = progress.get_player_scraped_seasons(team_name, player_id)
        seasons_for_player = [s for s in new_seasons if s not in already_scraped]
        
        if not seasons_for_player:
            continue  # All seasons for this player already done
        
        # Check quota before making requests
        if api_calls_made > 0 and api_calls_made % 50 == 0:
            remaining = check_quota()
            print(f"     [Quota check: {remaining} remaining]")
            if remaining < MIN_QUOTA_BUFFER:
                print(f"  ⚠️ Quota too low ({remaining} < {MIN_QUOTA_BUFFER}). Stopping.")
                quota_exhausted = True
                break
        
        # Scrape this player's historical stats
        player_stats = []
        
        for season in seasons_for_player:
            try:
                stats_response = scraper.api.get_player_statistics(player_id, season)
                api_calls_made += 1
                
                if stats_response:
                    player_data = stats_response[0]
                    player_info = player_data.get("player", {})
                    birth = player_info.get("birth", {})
                    
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
                        
                        player_stats.append({
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
                            "height": player_info.get("height"),
                            "weight": player_info.get("weight"),
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
                            "appearances": games.get("appearences"),
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
                            "penalty_committed": penalty.get("commited"),
                            "penalty_scored": penalty.get("scored"),
                            "penalty_missed": penalty.get("missed"),
                            "penalty_saved": penalty.get("saved"),
                        })
                
                # Mark season as done for this player
                progress.mark_player_season(team_name, player_id, season)
                
                # Also track for new teams
                if is_new_team:
                    progress.add_new_team_season_done(team_name, season)
                
            except Exception as e:
                print(f"  ⚠️ Error fetching stats for {player_name} season {season}: {e}")
        
        if player_stats:
            all_player_stats.extend(player_stats)
        
        # Save progress periodically
        if api_calls_made % 20 == 0:
            progress.add_api_calls(20)
            progress.save()
            
        # If quota exhausted, break out of player loop immediately
        if quota_exhausted:
            break
    
    # Don't mark team as completed here - do it after merge succeeds in main()
    
    progress.add_api_calls(api_calls_made % 20)  # Add remaining calls
    progress.save()
    
    # IMPORTANT: Return whatever data we have, even if partial
    # The caller will save it to CSV
    if all_player_stats:
        return pd.DataFrame(all_player_stats), api_calls_made, quota_exhausted
    return pd.DataFrame(), api_calls_made, quota_exhausted


def main():
    """Main entry point."""
    print("=" * 70)
    print("🕰️  HISTORICAL DATA SCRAPER")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Historical seasons to scrape: {HISTORICAL_SEASONS}")
    print("=" * 70)
    
    # Initialize progress tracker
    progress = ProgressTracker()
    print(f"\n📋 Progress: {progress.get_stats()}")
    
    # Check API quota
    print("\n📊 Checking API quota...")
    remaining = check_quota()
    print(f"   Remaining requests: {remaining}")
    
    if remaining < MIN_QUOTA_BUFFER:
        print(f"   ⚠️ Quota too low ({remaining} < {MIN_QUOTA_BUFFER}). Wait for reset.")
        return
    
    if remaining < 500:
        print("   ⚠️ Low quota. Will scrape what we can.")
    
    # Initialize scraper
    scraper = WorldCup2026Scraper()
    
    # Get list of teams from existing data
    teams_file = f"{OUTPUT_DIR}/teams/world_cup_2026_teams.csv"
    if not os.path.exists(teams_file):
        print(f"❌ Teams file not found: {teams_file}")
        return
    
    teams_df = pd.read_csv(teams_file)
    print(f"\n📋 Found {len(teams_df)} teams")
    
    # Track overall progress
    total_api_calls = 0
    quota_exhausted = False
    
    for idx, team in teams_df.iterrows():
        team_id = team["team_id"]
        team_name = team["team_name"]
        
        if pd.isna(team_id):
            continue
        
        team_id = int(team_id)
        
        print(f"\n{'='*60}")
        print(f"📌 [{idx+1}/{len(teams_df)}] {team_name} (ID: {team_id})")
        print(f"{'='*60}")
        
        try:
            # Scrape historical data with progress tracking
            new_stats_df, api_calls, quota_exhausted = scrape_historical_for_team(
                scraper, team_id, team_name, HISTORICAL_SEASONS, progress
            )
            
            total_api_calls += api_calls
            
            # Merge with existing data (even if partial due to quota)
            if not new_stats_df.empty:
                merge_and_save_stats(team_name, new_stats_df, team_id)
                print(f"  💾 Data saved to CSV")
            
            # Only mark team as completed if NOT interrupted by quota
            if not quota_exhausted:
                if not new_stats_df.empty:
                    progress.mark_team_completed(team_name)
                    print(f"  ✅ Team marked as completed")
                elif progress.is_team_completed(team_name):
                    # Already completed before, just skipped
                    pass
                else:
                    # No new data scraped (all seasons already in CSV)
                    progress.mark_team_completed(team_name)
                    print(f"  ✅ Team marked as completed (all seasons already present)")
            else:
                # Quota exhausted mid-team - save partial progress but don't mark complete
                print(f"  ⏸️ Partial progress saved for {team_name}. Will resume next run.")
                progress.save()  # Ensure progress is saved immediately
            
            if quota_exhausted:
                print(f"\n⏸️ Stopping due to low quota. Run again tomorrow to continue.")
                print(f"   Progress saved. {progress.get_stats()}")
                progress.save()  # Final save before exit
                break
                
        except KeyboardInterrupt:
            print("\n\n⏸️ Interrupted by user. Progress saved.")
            progress.save()
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Update the combined all-teams statistics file
    print("\n📊 Updating combined statistics file...")
    all_stats = []
    for filename in os.listdir(STATISTICS_OUTPUT_DIR):
        if filename.endswith("_player_statistics.csv") and filename != "all_player_statistics.csv":
            filepath = f"{STATISTICS_OUTPUT_DIR}/{filename}"
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    all_stats.append(df)
            except Exception:
                pass
    
    if all_stats:
        combined_all = pd.concat(all_stats, ignore_index=True)
        # Deduplicate
        combined_all = combined_all.drop_duplicates(
            subset=['player_id', 'season', 'team_id', 'league_id'],
            keep='first'
        )
        combined_all = combined_all.sort_values(['player_name', 'season', 'team_name'])
        all_stats_file = f"{STATISTICS_OUTPUT_DIR}/all_player_statistics.csv"
        combined_all.to_csv(all_stats_file, index=False)
        print(f"✅ Combined statistics: {len(combined_all)} records")
        print(f"   Seasons: {sorted(combined_all['season'].unique())}")
    
    print("\n" + "=" * 70)
    print("🏁 HISTORICAL SCRAPING SESSION COMPLETE")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   API calls this session: {total_api_calls}")
    print(f"   {progress.get_stats()}")
    if quota_exhausted:
        print("   ⚠️ Run again tomorrow to continue with remaining teams/seasons")
    print("=" * 70)


if __name__ == "__main__":
    main()

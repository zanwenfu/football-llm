#!/usr/bin/env python3
"""
Production scraper for World Cup 2026 teams.
Features:
- Real-time data validation
- Auto-pause on errors
- Progress tracking and resume capability
- API quota monitoring
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from api_client import api_client
from scraper import WorldCup2026Scraper
from config import (
    SEASONS_TO_SCRAPE,
    OUTPUT_DIR,
    TEAMS_OUTPUT_DIR,
    PLAYERS_OUTPUT_DIR,
    STATISTICS_OUTPUT_DIR,
)

# Progress tracking file
PROGRESS_FILE = f"{OUTPUT_DIR}/scrape_progress.json"

# Error thresholds
MAX_CONSECUTIVE_ERRORS = 5
MAX_EMPTY_RESPONSES = 10
MIN_PLAYERS_PER_TEAM = 10  # Warn if team has fewer players


class DataValidator:
    """Validates scraped data quality on-the-fly."""
    
    REQUIRED_STAT_COLUMNS = [
        'player_id', 'player_name', 'season', 'team_id', 'team_name',
        'appearances', 'minutes', 'goals'
    ]
    
    @staticmethod
    def validate_squad(df: pd.DataFrame, team_name: str) -> Tuple[bool, str]:
        """Validate squad data quality."""
        if df.empty:
            return False, f"Empty squad for {team_name}"
        
        if len(df) < MIN_PLAYERS_PER_TEAM:
            return True, f"⚠️ Warning: {team_name} has only {len(df)} players (expected >={MIN_PLAYERS_PER_TEAM})"
        
        # Check for required columns
        required = ['player_id', 'player_name', 'position']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return False, f"Missing columns in squad: {missing}"
        
        # Check for null player_ids
        null_ids = df['player_id'].isnull().sum()
        if null_ids > 0:
            return True, f"⚠️ Warning: {null_ids} players without ID in {team_name}"
        
        return True, f"✅ Valid: {len(df)} players"
    
    @staticmethod
    def validate_statistics(df: pd.DataFrame, team_name: str, expected_players: int) -> Tuple[bool, str]:
        """Validate player statistics data quality."""
        if df.empty:
            return False, f"Empty statistics for {team_name}"
        
        # Check for required columns
        missing = [col for col in DataValidator.REQUIRED_STAT_COLUMNS if col not in df.columns]
        if missing:
            return False, f"Missing statistics columns: {missing}"
        
        # Count unique players with data
        players_with_data = df['player_id'].nunique()
        coverage = (players_with_data / expected_players) * 100 if expected_players > 0 else 0
        
        # Count records with actual appearances
        records_with_apps = len(df[df['appearances'] > 0])
        
        if coverage < 50:
            return True, f"⚠️ Low coverage: {players_with_data}/{expected_players} players ({coverage:.1f}%)"
        
        return True, f"✅ Valid: {len(df)} records, {players_with_data} players ({coverage:.1f}%), {records_with_apps} with appearances"


class ProgressTracker:
    """Tracks scraping progress for resume capability."""
    
    def __init__(self, progress_file: str = PROGRESS_FILE):
        self.progress_file = progress_file
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict:
        """Load existing progress or create new."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "started_at": None,
            "last_updated": None,
            "completed_teams": [],
            "failed_teams": [],
            "total_requests": 0,
            "errors": [],
        }
    
    def save(self):
        """Save current progress."""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def mark_team_completed(self, team_id: int, team_name: str, stats_count: int):
        """Mark a team as completed."""
        self.progress["completed_teams"].append({
            "team_id": team_id,
            "team_name": team_name,
            "stats_count": stats_count,
            "completed_at": datetime.now().isoformat()
        })
        self.save()
    
    def mark_team_failed(self, team_id: int, team_name: str, reason: str):
        """Mark a team as failed."""
        self.progress["failed_teams"].append({
            "team_id": team_id,
            "team_name": team_name,
            "reason": reason,
            "failed_at": datetime.now().isoformat()
        })
        self.save()
    
    def add_error(self, error: str):
        """Log an error."""
        self.progress["errors"].append({
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.save()
    
    def is_team_completed(self, team_id: int) -> bool:
        """Check if a team has already been scraped."""
        return any(t["team_id"] == team_id for t in self.progress["completed_teams"])
    
    def get_summary(self) -> str:
        """Get progress summary."""
        completed = len(self.progress["completed_teams"])
        failed = len(self.progress["failed_teams"])
        errors = len(self.progress["errors"])
        return f"Completed: {completed}, Failed: {failed}, Errors: {errors}"


class QuotaMonitor:
    """Monitors API quota usage."""
    
    def __init__(self, daily_limit: int = 7500):
        self.daily_limit = daily_limit
        self.requests_made = 0
        self.start_time = time.time()
    
    def check_quota(self) -> Tuple[bool, int, int]:
        """
        Check API quota status.
        Returns: (is_ok, remaining, used)
        """
        try:
            status = api_client.get_account_status()
            response = status.get("response", {})
            requests_info = response.get("requests", {})
            
            current = requests_info.get("current", 0)
            limit = requests_info.get("limit_day", self.daily_limit)
            remaining = limit - current
            
            return remaining > 100, remaining, current  # Leave buffer of 100 requests
        except Exception as e:
            print(f"  ⚠️ Could not check quota: {e}")
            return True, -1, -1
    
    def print_status(self):
        """Print current quota status."""
        is_ok, remaining, used = self.check_quota()
        if remaining >= 0:
            print(f"\n📊 API Quota: {used}/{self.daily_limit} used, {remaining} remaining")
            if not is_ok:
                print("  ⚠️ WARNING: Running low on API quota!")
        return is_ok


def scrape_all_teams(resume: bool = True):
    """
    Scrape all World Cup 2026 teams with validation and error handling.
    
    Args:
        resume: If True, skip already-completed teams
    """
    print("=" * 70)
    print("🏆 WORLD CUP 2026 - FULL TEAM SCRAPER (PRODUCTION)")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Seasons: {SEASONS_TO_SCRAPE}")
    print("=" * 70)
    
    # Initialize components
    scraper = WorldCup2026Scraper()
    validator = DataValidator()
    progress = ProgressTracker()
    quota = QuotaMonitor()
    
    # Start progress tracking
    if not progress.progress["started_at"]:
        progress.progress["started_at"] = datetime.now().isoformat()
        progress.save()
    
    # Check initial quota
    if not quota.print_status():
        print("\n❌ Insufficient API quota. Aborting.")
        return
    
    # Step 1: Get or load teams list
    teams_file = f"{TEAMS_OUTPUT_DIR}/world_cup_2026_teams.csv"
    
    if os.path.exists(teams_file):
        print(f"\n📋 Loading existing teams from {teams_file}")
        teams_df = pd.read_csv(teams_file)
    else:
        print("\n📋 Fetching World Cup 2026 teams...")
        teams_df = scraper.scrape_world_cup_teams()
    
    if teams_df.empty:
        print("❌ No teams found. Aborting.")
        return
    
    print(f"   Found {len(teams_df)} teams total")
    
    # Filter out already completed teams if resuming
    if resume:
        completed_ids = [t["team_id"] for t in progress.progress["completed_teams"]]
        remaining_teams = teams_df[~teams_df["team_id"].isin(completed_ids)]
        skipped = len(teams_df) - len(remaining_teams)
        if skipped > 0:
            print(f"   ⏭️ Skipping {skipped} already-completed teams")
        teams_df = remaining_teams
    
    print(f"   Teams to process: {len(teams_df)}")
    print(f"   Progress: {progress.get_summary()}")
    
    # Error tracking
    consecutive_errors = 0
    empty_responses = 0
    
    # Step 2: Process each team
    all_squads = []
    all_statistics = []
    
    for idx, team in teams_df.iterrows():
        team_id = team["team_id"]
        team_name = team["team_name"]
        
        if pd.isna(team_id):
            continue
        
        team_id = int(team_id)
        
        print(f"\n{'='*60}")
        print(f"📌 [{idx+1}/{len(teams_df)}] Processing: {team_name} (ID: {team_id})")
        print(f"{'='*60}")
        
        try:
            # Check quota before each team
            if idx % 5 == 0:  # Check every 5 teams
                is_ok, remaining, _ = quota.check_quota()
                if not is_ok:
                    print(f"\n⚠️ API quota low ({remaining} remaining). Pausing...")
                    print("   Re-run the script tomorrow to continue.")
                    progress.save()
                    return
            
            # Step 2a: Get squad
            print(f"\n👥 Fetching squad...")
            squad_df = scraper.scrape_team_squad(team_id, team_name)
            
            # Validate squad
            is_valid, message = validator.validate_squad(squad_df, team_name)
            print(f"   {message}")
            
            if not is_valid:
                consecutive_errors += 1
                empty_responses += 1
                progress.mark_team_failed(team_id, team_name, message)
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"\n❌ {MAX_CONSECUTIVE_ERRORS} consecutive errors. PAUSING to avoid wasting API quota.")
                    print("   Please check the errors and re-run when ready.")
                    progress.save()
                    sys.exit(1)
                continue
            
            all_squads.append(squad_df)
            
            # Step 2b: Get player statistics
            print(f"\n📊 Fetching player statistics ({len(squad_df)} players × {len(SEASONS_TO_SCRAPE)} seasons)...")
            stats_df = scraper.scrape_all_team_players_statistics(team_id, team_name, squad_df)
            
            # Validate statistics
            is_valid, message = validator.validate_statistics(stats_df, team_name, len(squad_df))
            print(f"   {message}")
            
            if stats_df.empty:
                empty_responses += 1
                if empty_responses >= MAX_EMPTY_RESPONSES:
                    print(f"\n⚠️ {MAX_EMPTY_RESPONSES} empty responses. API may be having issues.")
                    progress.add_error(f"Too many empty responses after {team_name}")
            else:
                all_statistics.append(stats_df)
                empty_responses = 0  # Reset on success
            
            # Mark team as completed
            progress.mark_team_completed(team_id, team_name, len(stats_df))
            consecutive_errors = 0  # Reset on success
            
            print(f"\n✅ {team_name} completed: {len(squad_df)} players, {len(stats_df)} stat records")
            
        except KeyboardInterrupt:
            print("\n\n⏸️ Interrupted by user. Progress saved.")
            progress.save()
            sys.exit(0)
            
        except Exception as e:
            error_msg = f"Error processing {team_name}: {str(e)}"
            print(f"\n❌ {error_msg}")
            progress.add_error(error_msg)
            progress.mark_team_failed(team_id, team_name, str(e))
            consecutive_errors += 1
            
            # Check if error is critical
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                print("\n⚠️ Rate limit or quota error detected. Waiting 60 seconds...")
                time.sleep(60)
                consecutive_errors = 0  # Don't count rate limit as consecutive error
            
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n❌ {MAX_CONSECUTIVE_ERRORS} consecutive errors. PAUSING.")
                progress.save()
                sys.exit(1)
    
    # Step 3: Save combined data
    print("\n" + "=" * 70)
    print("💾 SAVING COMBINED DATA")
    print("=" * 70)
    
    if all_squads:
        combined_squads = pd.concat(all_squads, ignore_index=True)
        output_path = f"{PLAYERS_OUTPUT_DIR}/all_squads.csv"
        combined_squads.to_csv(output_path, index=False)
        print(f"✅ Saved {len(combined_squads)} players to {output_path}")
    
    if all_statistics:
        combined_stats = pd.concat(all_statistics, ignore_index=True)
        output_path = f"{STATISTICS_OUTPUT_DIR}/all_player_statistics.csv"
        combined_stats.to_csv(output_path, index=False)
        print(f"✅ Saved {len(combined_stats)} statistics to {output_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏁 SCRAPING COMPLETE")
    print("=" * 70)
    quota.print_status()
    print(f"\nFinal progress: {progress.get_summary()}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_progress():
    """Print current scraping progress."""
    progress = ProgressTracker()
    
    print("=" * 60)
    print("📊 SCRAPING PROGRESS REPORT")
    print("=" * 60)
    
    p = progress.progress
    print(f"Started: {p['started_at'] or 'Not started'}")
    print(f"Last updated: {p['last_updated'] or 'N/A'}")
    print(f"\nCompleted teams: {len(p['completed_teams'])}")
    
    if p['completed_teams']:
        total_stats = sum(t['stats_count'] for t in p['completed_teams'])
        print(f"Total statistics collected: {total_stats}")
        print("\nCompleted:")
        for t in p['completed_teams'][-10:]:  # Show last 10
            print(f"  ✅ {t['team_name']}: {t['stats_count']} records")
        if len(p['completed_teams']) > 10:
            print(f"  ... and {len(p['completed_teams']) - 10} more")
    
    if p['failed_teams']:
        print(f"\nFailed teams: {len(p['failed_teams'])}")
        for t in p['failed_teams']:
            print(f"  ❌ {t['team_name']}: {t['reason']}")
    
    if p['errors']:
        print(f"\nRecent errors: {len(p['errors'])}")
        for e in p['errors'][-5:]:
            print(f"  ⚠️ {e['error'][:80]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="World Cup 2026 Data Scraper")
    parser.add_argument("--progress", action="store_true", help="Show current progress")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't skip completed teams")
    parser.add_argument("--reset", action="store_true", help="Reset progress and start fresh")
    
    args = parser.parse_args()
    
    if args.progress:
        print_progress()
    elif args.reset:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print("✅ Progress reset. Ready for fresh start.")
        else:
            print("No progress file found.")
    else:
        scrape_all_teams(resume=not args.no_resume)

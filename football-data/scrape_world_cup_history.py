"""
Scraper for historical FIFA World Cup match-level data.

Scrapes match details, lineups, and events for all World Cup tournaments
available in API-Football (2010, 2014, 2018, 2022).

Data is stored in:
  data/world_cup_history/world_cup_matches.csv    - Match results & scores
  data/world_cup_history/world_cup_lineups.csv    - Starting XI & substitutes per match
  data/world_cup_history/world_cup_events.csv     - Goals, cards, substitutions per match

API calls budget:
  - 4 calls for fixtures (1 per season)
  - Up to 256 calls for lineups (1 per fixture)
  - Up to 256 calls for events (1 per fixture)
  Total: ~516 calls (well within daily limits)
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from api_client import api_client
from config import WORLD_CUP_LEAGUE_ID


# =============================================================================
# Configuration
# =============================================================================

WORLD_CUP_SEASONS = [2010, 2014, 2018, 2022]

OUTPUT_DIR = "data/world_cup_history"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "scrape_progress.json")

MATCHES_CSV = os.path.join(OUTPUT_DIR, "world_cup_matches.csv")
LINEUPS_CSV = os.path.join(OUTPUT_DIR, "world_cup_lineups.csv")
EVENTS_CSV = os.path.join(OUTPUT_DIR, "world_cup_events.csv")


# =============================================================================
# Progress tracking for resume capability
# =============================================================================

def load_progress() -> Dict[str, Any]:
    """Load scraping progress from disk."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {
        "fixtures_fetched": [],       # seasons whose fixtures have been fetched
        "lineups_fetched": [],        # fixture IDs whose lineups have been fetched
        "events_fetched": [],         # fixture IDs whose events have been fetched
        "completed": False,
    }


def save_progress(progress: Dict[str, Any]):
    """Save scraping progress to disk."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# =============================================================================
# Known World Cup teams validation
# Reference: https://en.wikipedia.org/wiki/FIFA_World_Cup
# =============================================================================

# Expected number of teams and matches per World Cup
EXPECTED_TEAMS_PER_WC = 32
EXPECTED_MATCHES_PER_WC = 64


def validate_fixtures(fixtures: List[Dict], season: int) -> bool:
    """
    Validate that we got the expected number of fixtures and teams for a WC.
    
    Args:
        fixtures: List of fixture dicts from API
        season: World Cup year
        
    Returns:
        True if validation passes
    """
    if len(fixtures) != EXPECTED_MATCHES_PER_WC:
        print(f"  ⚠️  WARNING: Expected {EXPECTED_MATCHES_PER_WC} matches for WC {season}, "
              f"got {len(fixtures)}")
        return False

    teams = set()
    for f in fixtures:
        t = f.get("teams", {})
        home = t.get("home", {})
        away = t.get("away", {})
        if home.get("id"):
            teams.add((home["id"], home["name"]))
        if away.get("id"):
            teams.add((away["id"], away["name"]))

    if len(teams) != EXPECTED_TEAMS_PER_WC:
        print(f"  ⚠️  WARNING: Expected {EXPECTED_TEAMS_PER_WC} teams for WC {season}, "
              f"got {len(teams)}")
        # Print teams for debugging
        for tid, tname in sorted(teams, key=lambda x: x[1]):
            print(f"    Team ID {tid}: {tname}")
        return False

    print(f"  ✅ Validated: {len(fixtures)} matches, {len(teams)} teams")
    return True


# =============================================================================
# Step 1: Fetch all fixtures (match results)
# =============================================================================

def fetch_all_fixtures(progress: Dict) -> pd.DataFrame:
    """
    Fetch all World Cup fixtures for all seasons.
    
    Returns:
        DataFrame with all match data
    """
    all_matches = []
    all_raw_fixtures = {}  # season -> list of raw fixture dicts (for lineup/event fetching)

    for season in WORLD_CUP_SEASONS:
        print(f"\n{'='*60}")
        print(f"📋 Fetching fixtures for World Cup {season}...")
        print(f"{'='*60}")

        fixtures = api_client.get_fixtures(league_id=WORLD_CUP_LEAGUE_ID, season=season)

        if not fixtures:
            print(f"  ❌ ERROR: No fixtures returned for WC {season}!")
            continue

        # Validate
        validate_fixtures(fixtures, season)

        # Print all teams found
        teams = set()
        for f in fixtures:
            t = f.get("teams", {})
            teams.add((t["home"]["id"], t["home"]["name"]))
            teams.add((t["away"]["id"], t["away"]["name"]))
        print(f"  Teams in WC {season}:")
        for tid, tname in sorted(teams, key=lambda x: x[1]):
            print(f"    {tid}: {tname}")

        all_raw_fixtures[season] = fixtures

        for f in fixtures:
            fixture = f.get("fixture", {})
            league = f.get("league", {})
            teams_data = f.get("teams", {})
            goals = f.get("goals", {})
            score = f.get("score", {})
            status = fixture.get("status", {})

            # Determine if match went to extra time or penalties
            et = score.get("extratime", {})
            pen = score.get("penalty", {})
            went_to_extra_time = et.get("home") is not None
            went_to_penalties = pen.get("home") is not None
            penalty_status = status.get("short") == "PEN"

            match_row = {
                "fixture_id": fixture.get("id"),
                "world_cup_year": season,
                "date": fixture.get("date"),
                "round": league.get("round"),
                "venue_name": fixture.get("venue", {}).get("name"),
                "venue_city": fixture.get("venue", {}).get("city"),
                "referee": fixture.get("referee"),
                "status": status.get("short"),
                "status_long": status.get("long"),
                "elapsed_minutes": status.get("elapsed"),
                # Home team
                "home_team_id": teams_data.get("home", {}).get("id"),
                "home_team_name": teams_data.get("home", {}).get("name"),
                "home_team_winner": teams_data.get("home", {}).get("winner"),
                # Away team
                "away_team_id": teams_data.get("away", {}).get("id"),
                "away_team_name": teams_data.get("away", {}).get("name"),
                "away_team_winner": teams_data.get("away", {}).get("winner"),
                # Score
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                # Halftime
                "halftime_home": score.get("halftime", {}).get("home"),
                "halftime_away": score.get("halftime", {}).get("away"),
                # Fulltime
                "fulltime_home": score.get("fulltime", {}).get("home"),
                "fulltime_away": score.get("fulltime", {}).get("away"),
                # Extra time
                "extratime_home": et.get("home"),
                "extratime_away": et.get("away"),
                "went_to_extra_time": went_to_extra_time,
                # Penalties
                "penalty_home": pen.get("home"),
                "penalty_away": pen.get("away"),
                "went_to_penalties": went_to_penalties or penalty_status,
            }
            all_matches.append(match_row)

        progress["fixtures_fetched"].append(season)
        save_progress(progress)

    df = pd.DataFrame(all_matches)
    df = df.sort_values(["world_cup_year", "date"]).reset_index(drop=True)
    return df, all_raw_fixtures


# =============================================================================
# Step 2: Fetch lineups for each fixture
# =============================================================================

def fetch_all_lineups(
    fixtures_by_season: Dict[int, List[Dict]],
    progress: Dict,
) -> pd.DataFrame:
    """
    Fetch lineups for all fixtures across all World Cup seasons.
    
    Args:
        fixtures_by_season: Dict mapping season -> list of fixture dicts
        progress: Progress tracking dict
        
    Returns:
        DataFrame with all lineup data
    """
    all_lineups = []
    fetched_set: Set[int] = set(progress.get("lineups_fetched", []))

    # Collect all fixture IDs we need to process
    all_fixture_ids = []
    fixture_meta = {}  # fixture_id -> (season, round, home_name, away_name)
    for season, fixtures in fixtures_by_season.items():
        for f in fixtures:
            fid = f["fixture"]["id"]
            all_fixture_ids.append(fid)
            fixture_meta[fid] = (
                season,
                f["league"]["round"],
                f["teams"]["home"]["name"],
                f["teams"]["away"]["name"],
            )

    remaining = [fid for fid in all_fixture_ids if fid not in fetched_set]
    print(f"\n{'='*60}")
    print(f"👕 Fetching lineups: {len(remaining)} fixtures remaining "
          f"(of {len(all_fixture_ids)} total)")
    print(f"{'='*60}")

    if not remaining:
        print("  All lineups already fetched!")

    for i, fid in enumerate(tqdm(remaining, desc="Fetching lineups")):
        season, round_name, home, away = fixture_meta[fid]

        try:
            response = api_client._make_request("fixtures/lineups", {"fixture": fid})
            lineup_data = response.get("response", [])
        except Exception as e:
            print(f"\n  ❌ Error fetching lineup for fixture {fid} ({home} vs {away}): {e}")
            continue

        if not lineup_data:
            print(f"\n  ⚠️  No lineup data for fixture {fid} ({home} vs {away}, WC {season})")

        for team_lineup in lineup_data:
            team = team_lineup.get("team", {})
            team_id = team.get("id")
            team_name = team.get("name")
            formation = team_lineup.get("formation")
            coach = team_lineup.get("coach", {})
            coach_id = coach.get("id")
            coach_name = coach.get("name")

            # Starting XI
            for player_entry in team_lineup.get("startXI", []):
                p = player_entry.get("player", {})
                all_lineups.append({
                    "fixture_id": fid,
                    "world_cup_year": season,
                    "round": round_name,
                    "team_id": team_id,
                    "team_name": team_name,
                    "formation": formation,
                    "coach_id": coach_id,
                    "coach_name": coach_name,
                    "player_id": p.get("id"),
                    "player_name": p.get("name"),
                    "player_number": p.get("number"),
                    "player_position": p.get("pos"),
                    "player_grid": p.get("grid"),
                    "is_starter": True,
                    "is_substitute": False,
                })

            # Substitutes
            for player_entry in team_lineup.get("substitutes", []):
                p = player_entry.get("player", {})
                all_lineups.append({
                    "fixture_id": fid,
                    "world_cup_year": season,
                    "round": round_name,
                    "team_id": team_id,
                    "team_name": team_name,
                    "formation": formation,
                    "coach_id": coach_id,
                    "coach_name": coach_name,
                    "player_id": p.get("id"),
                    "player_name": p.get("name"),
                    "player_number": p.get("number"),
                    "player_position": p.get("pos"),
                    "player_grid": p.get("grid"),
                    "is_starter": False,
                    "is_substitute": True,
                })

        fetched_set.add(fid)

        # Save progress every 20 fixtures
        if (i + 1) % 20 == 0:
            progress["lineups_fetched"] = list(fetched_set)
            save_progress(progress)

    # Final progress save
    progress["lineups_fetched"] = list(fetched_set)
    save_progress(progress)

    df = pd.DataFrame(all_lineups)
    return df


# =============================================================================
# Step 3: Fetch events for each fixture (goals, cards, subs)
# =============================================================================

def fetch_all_events(
    fixtures_by_season: Dict[int, List[Dict]],
    progress: Dict,
) -> pd.DataFrame:
    """
    Fetch events (goals, cards, substitutions) for all fixtures.
    
    Args:
        fixtures_by_season: Dict mapping season -> list of fixture dicts
        progress: Progress tracking dict
        
    Returns:
        DataFrame with all event data
    """
    all_events = []
    fetched_set: Set[int] = set(progress.get("events_fetched", []))

    # Collect all fixture IDs
    all_fixture_ids = []
    fixture_meta = {}
    for season, fixtures in fixtures_by_season.items():
        for f in fixtures:
            fid = f["fixture"]["id"]
            all_fixture_ids.append(fid)
            fixture_meta[fid] = (
                season,
                f["league"]["round"],
                f["teams"]["home"]["name"],
                f["teams"]["away"]["name"],
            )

    remaining = [fid for fid in all_fixture_ids if fid not in fetched_set]
    print(f"\n{'='*60}")
    print(f"⚽ Fetching events: {len(remaining)} fixtures remaining "
          f"(of {len(all_fixture_ids)} total)")
    print(f"{'='*60}")

    if not remaining:
        print("  All events already fetched!")

    for i, fid in enumerate(tqdm(remaining, desc="Fetching events")):
        season, round_name, home, away = fixture_meta[fid]

        try:
            response = api_client._make_request("fixtures/events", {"fixture": fid})
            event_data = response.get("response", [])
        except Exception as e:
            print(f"\n  ❌ Error fetching events for fixture {fid} ({home} vs {away}): {e}")
            continue

        for event in event_data:
            time_data = event.get("time", {})
            team = event.get("team", {})
            player = event.get("player", {})
            assist = event.get("assist", {})

            all_events.append({
                "fixture_id": fid,
                "world_cup_year": season,
                "round": round_name,
                "time_elapsed": time_data.get("elapsed"),
                "time_extra": time_data.get("extra"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "assist_id": assist.get("id"),
                "assist_name": assist.get("name"),
                "event_type": event.get("type"),
                "event_detail": event.get("detail"),
                "comments": event.get("comments"),
            })

        fetched_set.add(fid)

        # Save progress every 20 fixtures
        if (i + 1) % 20 == 0:
            progress["events_fetched"] = list(fetched_set)
            save_progress(progress)

    # Final progress save
    progress["events_fetched"] = list(fetched_set)
    save_progress(progress)

    df = pd.DataFrame(all_events)
    return df


# =============================================================================
# Validation & Summary
# =============================================================================

def print_summary(matches_df: pd.DataFrame, lineups_df: pd.DataFrame, events_df: pd.DataFrame):
    """Print a detailed summary of scraped data."""
    print(f"\n{'='*60}")
    print(f"📊 SCRAPING SUMMARY")
    print(f"{'='*60}")

    print(f"\n--- Matches ---")
    print(f"Total matches: {len(matches_df)}")
    for season in WORLD_CUP_SEASONS:
        season_df = matches_df[matches_df["world_cup_year"] == season]
        teams = set(
            list(season_df["home_team_name"].unique()) +
            list(season_df["away_team_name"].unique())
        )
        pens = season_df["went_to_penalties"].sum()
        et = season_df["went_to_extra_time"].sum()
        print(f"  WC {season}: {len(season_df)} matches, {len(teams)} teams, "
              f"{int(et)} went to ET, {int(pens)} went to penalties")

    if not lineups_df.empty:
        print(f"\n--- Lineups ---")
        print(f"Total lineup entries: {len(lineups_df)}")
        starters = lineups_df[lineups_df["is_starter"] == True]
        subs = lineups_df[lineups_df["is_substitute"] == True]
        print(f"  Starters: {len(starters)}")
        print(f"  Substitutes: {len(subs)}")
        print(f"  Unique players: {lineups_df['player_id'].nunique()}")
        for season in WORLD_CUP_SEASONS:
            s_df = lineups_df[lineups_df["world_cup_year"] == season]
            fixtures_with_lineups = s_df["fixture_id"].nunique()
            print(f"  WC {season}: {fixtures_with_lineups} fixtures with lineups, "
                  f"{s_df['player_id'].nunique()} unique players")

    if not events_df.empty:
        print(f"\n--- Events ---")
        print(f"Total events: {len(events_df)}")
        for etype in events_df["event_type"].unique():
            count = len(events_df[events_df["event_type"] == etype])
            print(f"  {etype}: {count}")
        for season in WORLD_CUP_SEASONS:
            s_df = events_df[events_df["world_cup_year"] == season]
            goals = len(s_df[s_df["event_type"] == "Goal"])
            print(f"  WC {season}: {s_df['fixture_id'].nunique()} fixtures with events, "
                  f"{goals} goals")

    print(f"\n--- Files saved ---")
    print(f"  {MATCHES_CSV}")
    print(f"  {LINEUPS_CSV}")
    print(f"  {EVENTS_CSV}")
    print()


def validate_completeness(
    matches_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> bool:
    """
    Validate that we have complete data.
    
    Returns:
        True if all validations pass
    """
    print(f"\n{'='*60}")
    print(f"🔍 VALIDATION")
    print(f"{'='*60}")

    all_ok = True

    # Check match counts
    for season in WORLD_CUP_SEASONS:
        season_matches = matches_df[matches_df["world_cup_year"] == season]
        if len(season_matches) != EXPECTED_MATCHES_PER_WC:
            print(f"  ❌ WC {season}: Expected {EXPECTED_MATCHES_PER_WC} matches, "
                  f"got {len(season_matches)}")
            all_ok = False

        # Check team counts
        teams = set(
            list(season_matches["home_team_name"].unique()) +
            list(season_matches["away_team_name"].unique())
        )
        if len(teams) != EXPECTED_TEAMS_PER_WC:
            print(f"  ❌ WC {season}: Expected {EXPECTED_TEAMS_PER_WC} teams, "
                  f"got {len(teams)}")
            all_ok = False

    # Check lineup completeness (should have lineups for most fixtures)
    if not lineups_df.empty:
        fixtures_with_lineups = lineups_df["fixture_id"].nunique()
        total_fixtures = matches_df["fixture_id"].nunique()
        coverage = fixtures_with_lineups / total_fixtures * 100
        print(f"  Lineup coverage: {fixtures_with_lineups}/{total_fixtures} "
              f"fixtures ({coverage:.1f}%)")
        if coverage < 90:
            print(f"  ⚠️  Lineup coverage below 90%")

        # Every fixture with lineups should have exactly 22 starters (11 per team)
        starters_per_fixture = lineups_df[lineups_df["is_starter"]].groupby("fixture_id").size()
        bad_fixtures = starters_per_fixture[starters_per_fixture != 22]
        if len(bad_fixtures) > 0:
            print(f"  ⚠️  {len(bad_fixtures)} fixtures don't have exactly 22 starters")
        else:
            print(f"  ✅ All fixtures with lineups have exactly 22 starters")

    # Check events completeness
    if not events_df.empty:
        fixtures_with_events = events_df["fixture_id"].nunique()
        total_fixtures = matches_df["fixture_id"].nunique()
        coverage = fixtures_with_events / total_fixtures * 100
        print(f"  Events coverage: {fixtures_with_events}/{total_fixtures} "
              f"fixtures ({coverage:.1f}%)")

    # Validate match scores make sense
    score_issues = 0
    for _, match in matches_df.iterrows():
        ft_home = match.get("fulltime_home")
        ft_away = match.get("fulltime_away")
        if ft_home is None or ft_away is None:
            score_issues += 1
    if score_issues > 0:
        print(f"  ⚠️  {score_issues} matches with missing fulltime scores")
    else:
        print(f"  ✅ All matches have fulltime scores")

    if all_ok:
        print(f"\n  ✅ ALL VALIDATIONS PASSED")
    else:
        print(f"\n  ⚠️  SOME VALIDATIONS FAILED - review above")

    return all_ok


# =============================================================================
# Main execution
# =============================================================================

def main():
    """Main scraping workflow."""
    print(f"\n{'='*60}")
    print(f"🏆 FIFA WORLD CUP HISTORICAL DATA SCRAPER")
    print(f"   Seasons: {WORLD_CUP_SEASONS}")
    print(f"   Output: {OUTPUT_DIR}/")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load progress
    progress = load_progress()

    if progress.get("completed"):
        print("\n✅ Scraping was already completed! Delete progress file to re-run.")
        print(f"   Progress file: {PROGRESS_FILE}")

        # Load and show summary
        if os.path.exists(MATCHES_CSV):
            matches_df = pd.read_csv(MATCHES_CSV)
            lineups_df = pd.read_csv(LINEUPS_CSV) if os.path.exists(LINEUPS_CSV) else pd.DataFrame()
            events_df = pd.read_csv(EVENTS_CSV) if os.path.exists(EVENTS_CSV) else pd.DataFrame()
            print_summary(matches_df, lineups_df, events_df)
        return

    # =========================================================================
    # Step 1: Fetch all fixtures
    # =========================================================================
    print("\n\n" + "="*60)
    print("STEP 1/3: Fetching match fixtures...")
    print("="*60)

    matches_df, raw_fixtures = fetch_all_fixtures(progress)

    # Save matches CSV immediately
    matches_df.to_csv(MATCHES_CSV, index=False)
    print(f"\n✅ Saved {len(matches_df)} matches to {MATCHES_CSV}")

    # =========================================================================
    # Step 2: Fetch lineups for each fixture
    # =========================================================================
    print("\n\n" + "="*60)
    print("STEP 2/3: Fetching match lineups...")
    print("="*60)

    lineups_df = fetch_all_lineups(raw_fixtures, progress)

    if not lineups_df.empty:
        lineups_df.to_csv(LINEUPS_CSV, index=False)
        print(f"\n✅ Saved {len(lineups_df)} lineup entries to {LINEUPS_CSV}")
    else:
        print("\n⚠️  No lineup data collected!")

    # =========================================================================
    # Step 3: Fetch events for each fixture
    # =========================================================================
    print("\n\n" + "="*60)
    print("STEP 3/3: Fetching match events...")
    print("="*60)

    events_df = fetch_all_events(raw_fixtures, progress)

    if not events_df.empty:
        events_df.to_csv(EVENTS_CSV, index=False)
        print(f"\n✅ Saved {len(events_df)} events to {EVENTS_CSV}")
    else:
        print("\n⚠️  No event data collected!")

    # =========================================================================
    # Mark completed & validate
    # =========================================================================
    progress["completed"] = True
    save_progress(progress)

    # Validate & summarize
    validate_completeness(matches_df, lineups_df, events_df)
    print_summary(matches_df, lineups_df, events_df)

    print(f"\n🏁 Scraping completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Total API requests made: {api_client.total_requests}")


if __name__ == "__main__":
    main()

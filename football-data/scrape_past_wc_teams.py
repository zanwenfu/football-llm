#!/usr/bin/env python3
"""
Scrape player statistics for teams that went to past World Cups (2010-2022)
but are NOT in the World Cup 2026 dataset.

Approach:
1. For each of the 19 delta teams, discover all players who represented them
   in their WC year(s) using /players?team={national_team_id}&season={wc_year}
2. Deduplicate players across multiple WC appearances for the same team
3. For each unique player, scrape full historical stats across all seasons (2004-2025)
4. Save per-team CSVs and a combined all_player_statistics.csv
5. Output goes to data/past_wc_statistics/ (separate from the 2026 data)

Progress tracking allows safe resume if interrupted.
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Set, Optional
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from api_client import api_client

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# All 22 seasons to scrape per player (same as the 2026 dataset)
ALL_SEASONS = list(range(2025, 2003, -1))  # [2025, 2024, ..., 2004]

# Output directory (separate from the 2026 data)
OUTPUT_DIR = "data/past_wc_statistics"

# Progress file
PROGRESS_FILE = "data/past_wc_scrape_progress.json"

# The 19 teams that went to WC 2010-2022 but are NOT in the 2026 dataset
# Each entry: team_id -> (display_name, file_key, [wc_years])
PAST_WC_TEAMS = {
    4:    ("Russia",               "russia",                  [2018, 2014]),
    5:    ("Sweden",               "sweden",                  [2018]),
    14:   ("Serbia",               "serbia",                  [2022, 2018, 2010]),
    18:   ("Iceland",              "iceland",                 [2018]),
    19:   ("Nigeria",              "nigeria",                 [2018, 2014, 2010]),
    21:   ("Denmark",              "denmark",                 [2022, 2018, 2010]),
    24:   ("Poland",               "poland",                  [2022, 2018]),
    29:   ("Costa Rica",           "costa_rica",              [2022, 2018, 2014]),
    30:   ("Peru",                 "peru",                    [2018]),
    767:  ("Wales",                "wales",                   [2022]),
    768:  ("Italy",                "italy",                   [2014, 2010]),
    773:  ("Slovakia",             "slovakia",                [2010]),
    1091: ("Slovenia",             "slovenia",                [2010]),
    1113: ("Bosnia & Herzegovina", "bosnia_and_herzegovina",  [2014]),
    1117: ("Greece",               "greece",                  [2014, 2010]),
    1530: ("Cameroon",             "cameroon",                [2022, 2014, 2010]),
    1561: ("North Korea",          "north_korea",             [2010]),
    2383: ("Chile",                "chile",                   [2014, 2010]),
    4672: ("Honduras",             "honduras",                [2014, 2010]),
}

# Expected column order (same 57 columns as the 2026 dataset)
EXPECTED_COLUMNS = [
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


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """Track scraping progress for safe resume."""

    def __init__(self, progress_file: str = PROGRESS_FILE):
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
            "phase": "not_started",  # not_started, discovering, scraping, done
            "discovered_players": {},  # team_id_str -> {player_id_str: player_name}
            "completed_teams": [],     # list of team_id_str that are fully scraped
            "player_progress": {},     # team_id_str -> {player_id_str: [scraped_seasons]}
            "api_calls": 0,
            "started_at": datetime.now().isoformat(),
            "last_updated": None,
        }

    def save(self):
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    @property
    def phase(self) -> str:
        return self.progress["phase"]

    @phase.setter
    def phase(self, value: str):
        self.progress["phase"] = value
        self.save()

    def is_team_discovered(self, team_id: int) -> bool:
        return str(team_id) in self.progress["discovered_players"]

    def set_team_players(self, team_id: int, players: Dict[int, str]):
        """Store discovered players for a team."""
        self.progress["discovered_players"][str(team_id)] = {
            str(pid): pname for pid, pname in players.items()
        }
        self.save()

    def get_team_players(self, team_id: int) -> Dict[int, str]:
        """Get discovered players for a team."""
        raw = self.progress["discovered_players"].get(str(team_id), {})
        return {int(pid): pname for pid, pname in raw.items()}

    def is_team_completed(self, team_id: int) -> bool:
        return str(team_id) in self.progress["completed_teams"]

    def mark_team_completed(self, team_id: int):
        tid_str = str(team_id)
        if tid_str not in self.progress["completed_teams"]:
            self.progress["completed_teams"].append(tid_str)
        self.save()

    def get_player_scraped_seasons(self, team_id: int, player_id: int) -> Set[int]:
        tid_str = str(team_id)
        pid_str = str(player_id)
        team_prog = self.progress["player_progress"].get(tid_str, {})
        return set(team_prog.get(pid_str, []))

    def mark_player_season_scraped(self, team_id: int, player_id: int, season: int):
        tid_str = str(team_id)
        pid_str = str(player_id)
        if tid_str not in self.progress["player_progress"]:
            self.progress["player_progress"][tid_str] = {}
        if pid_str not in self.progress["player_progress"][tid_str]:
            self.progress["player_progress"][tid_str][pid_str] = []
        if season not in self.progress["player_progress"][tid_str][pid_str]:
            self.progress["player_progress"][tid_str][pid_str].append(season)

    def add_api_calls(self, count: int):
        self.progress["api_calls"] += count

    @property
    def api_calls(self) -> int:
        return self.progress["api_calls"]


# ═══════════════════════════════════════════════════════════════════════════════
# API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def discover_team_players(team_id: int, wc_years: List[int],
                          tracker: ProgressTracker) -> Dict[int, str]:
    """
    Discover all players who represented a national team across their WC years.
    Uses /players?team={team_id}&season={year} with pagination.
    Returns dict of {player_id: player_name}.
    """
    all_players = {}

    for year in wc_years:
        page = 1
        while True:
            resp = api_client._make_request(
                "players", {"team": team_id, "season": year, "page": page}
            )
            tracker.add_api_calls(1)
            players = resp.get("response", [])
            paging = resp.get("paging", {})

            for p in players:
                pi = p.get("player", {})
                pid = pi.get("id")
                pname = pi.get("name")
                if pid and pid not in all_players:
                    all_players[pid] = pname

            total_pages = paging.get("total", 1)
            if page >= total_pages:
                break
            page += 1

    return all_players


def scrape_player_season(player_id: int, season: int) -> List[dict]:
    """
    Scrape a single player's stats for a single season.
    Returns a list of stat row dicts (one per league/team the player played for).
    Uses identical parsing logic to scraper.py and repair_statistics.py.
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

            # Parse rating to float
            rating_raw = games.get("rating")
            rating = None
            if rating_raw is not None:
                try:
                    rating = round(float(rating_raw), 2)
                except (ValueError, TypeError):
                    rating = None

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
                "appearances": games.get("appearences"),  # API typo is intentional
                "lineups": games.get("lineups"),
                "minutes": games.get("minutes"),
                "rating": rating,
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
        print(f"    ⚠️ Error scraping player {player_id} season {season}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCRAPING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def phase_discover(tracker: ProgressTracker):
    """Phase 1: Discover all players for all 19 teams via their WC years."""
    print("\n" + "=" * 70)
    print("PHASE 1: PLAYER DISCOVERY")
    print("=" * 70)

    total_players = 0
    for team_id, (team_name, file_key, wc_years) in sorted(
        PAST_WC_TEAMS.items(), key=lambda x: x[1][0]
    ):
        if tracker.is_team_discovered(team_id):
            players = tracker.get_team_players(team_id)
            print(f"  ✅ {team_name}: {len(players)} players (cached)")
            total_players += len(players)
            continue

        print(f"  🔍 Discovering {team_name} (ID: {team_id}) "
              f"WC: {', '.join(str(y) for y in wc_years)}...")
        players = discover_team_players(team_id, wc_years, tracker)
        tracker.set_team_players(team_id, players)
        print(f"     → {len(players)} unique players found")
        total_players += len(players)

    print(f"\n  📊 Total unique players across all 19 teams: {total_players}")
    tracker.phase = "discovering_done"


def phase_scrape(tracker: ProgressTracker, dry_run: bool = False):
    """Phase 2: Scrape full historical stats for all discovered players.
    Returns True if completed successfully, False if interrupted."""
    print("\n" + "=" * 70)
    print(f"PHASE 2: SCRAPE PLAYER STATISTICS {'(DRY RUN)' if dry_run else ''}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_interval = 5  # Save progress every N players

    teams_done = 0
    teams_total = len(PAST_WC_TEAMS)

    for team_id, (team_name, file_key, wc_years) in sorted(
        PAST_WC_TEAMS.items(), key=lambda x: x[1][0]
    ):
        teams_done += 1

        if tracker.is_team_completed(team_id):
            print(f"\n  ✅ [{teams_done}/{teams_total}] {team_name}: already completed")
            continue

        players = tracker.get_team_players(team_id)
        if not players:
            print(f"\n  ⚠️ [{teams_done}/{teams_total}] {team_name}: no players discovered, skipping")
            continue

        print(f"\n  🏟️ [{teams_done}/{teams_total}] {team_name} "
              f"({len(players)} players, WC: {', '.join(str(y) for y in wc_years)})")

        if dry_run:
            # Count how many API calls would be needed
            calls_needed = 0
            for pid in players:
                scraped = tracker.get_player_scraped_seasons(team_id, pid)
                remaining = set(ALL_SEASONS) - scraped
                calls_needed += len(remaining)
            print(f"     Would make {calls_needed} API calls")
            continue

        # Scrape all players for this team
        all_team_rows = []
        players_with_data = 0
        players_no_data = 0
        player_list = list(players.items())
        interrupted = False

        try:
            for i, (player_id, player_name) in enumerate(
                tqdm(player_list, desc=f"  {team_name}", unit="player", leave=True)
            ):
                already_scraped = tracker.get_player_scraped_seasons(team_id, player_id)
                seasons_to_scrape = [s for s in ALL_SEASONS if s not in already_scraped]

                player_rows = []
                for season in seasons_to_scrape:
                    rows = scrape_player_season(player_id, season)
                    tracker.add_api_calls(1)
                    tracker.mark_player_season_scraped(team_id, player_id, season)
                    if rows:
                        player_rows.extend(rows)

                if player_rows:
                    # Add national team context
                    for row in player_rows:
                        row["national_team_id"] = team_id
                        row["national_team_name"] = team_name
                    all_team_rows.extend(player_rows)
                    players_with_data += 1
                else:
                    # Check if we got data in a previous partial run
                    if already_scraped:
                        players_with_data += 1  # Had data from before
                    else:
                        players_no_data += 1

                # Save progress periodically
                if (i + 1) % save_interval == 0:
                    tracker.save()
        except KeyboardInterrupt:
            print(f"\n     ⚠️ Interrupted! Saving partial progress...")
            tracker.save()
            interrupted = True

        # Also load any data from a previous partial run for this team
        output_path = f"{OUTPUT_DIR}/{file_key}_player_statistics.csv"
        if os.path.exists(output_path) and all_team_rows:
            # Merge with existing partial data
            existing_df = pd.read_csv(output_path)
            new_df = pd.DataFrame(all_team_rows)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            # Deduplicate
            combined = combined.drop_duplicates(
                subset=['player_id', 'season', 'league_name', 'team_name'],
                keep='first'
            )
        elif all_team_rows:
            combined = pd.DataFrame(all_team_rows)
            # Deduplicate just in case
            combined = combined.drop_duplicates(
                subset=['player_id', 'season', 'league_name', 'team_name'],
                keep='first'
            )
        elif os.path.exists(output_path):
            combined = pd.read_csv(output_path)
        else:
            combined = pd.DataFrame(columns=EXPECTED_COLUMNS)

        # Ensure correct column order
        if not combined.empty:
            combined = combined[EXPECTED_COLUMNS]

        # Save team file (partial or complete)
        combined.to_csv(output_path, index=False)

        if interrupted:
            print(f"     💾 Saved partial data: {len(combined)} rows → {output_path}")
            print(f"     Total API calls so far: {tracker.api_calls:,}")
            print(f"     Re-run to resume from where we left off.")
            return False  # Signal interruption
        else:
            tracker.mark_team_completed(team_id)
            print(f"     ✅ Saved {len(combined)} rows → {output_path}")
            print(f"     Players with data: {players_with_data}, no API data: {players_no_data}")
            print(f"     Total API calls so far: {tracker.api_calls:,}")

    tracker.save()
    return True  # Completed successfully


def phase_combine(tracker: ProgressTracker):
    """Phase 3: Combine all individual team files into one master CSV."""
    print("\n" + "=" * 70)
    print("PHASE 3: COMBINE INTO MASTER CSV")
    print("=" * 70)

    all_dfs = []
    for team_id, (team_name, file_key, wc_years) in sorted(
        PAST_WC_TEAMS.items(), key=lambda x: x[1][0]
    ):
        path = f"{OUTPUT_DIR}/{file_key}_player_statistics.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_dfs.append(df)
            print(f"  {team_name:<25} {len(df):>6} rows, "
                  f"{df['player_id'].nunique():>4} players")
        else:
            print(f"  ⚠️ {team_name}: file not found at {path}")

    if all_dfs:
        master = pd.concat(all_dfs, ignore_index=True)
        master_path = f"{OUTPUT_DIR}/all_player_statistics.csv"
        master.to_csv(master_path, index=False)

        print(f"\n  📊 Master CSV: {len(master):,} rows, "
              f"{master['player_id'].nunique():,} unique players, "
              f"{master['national_team_name'].nunique()} teams")
        print(f"  💾 Saved to {master_path}")
    else:
        print("  ❌ No team files found!")


def phase_validate():
    """Phase 4: Validate the scraped data."""
    print("\n" + "=" * 70)
    print("PHASE 4: VALIDATION")
    print("=" * 70)

    issues = []

    for team_id, (team_name, file_key, wc_years) in sorted(
        PAST_WC_TEAMS.items(), key=lambda x: x[1][0]
    ):
        path = f"{OUTPUT_DIR}/{file_key}_player_statistics.csv"
        if not os.path.exists(path):
            issues.append(f"CRITICAL: {team_name} file missing: {path}")
            continue

        df = pd.read_csv(path)

        # Column check
        if list(df.columns) != EXPECTED_COLUMNS:
            issues.append(f"CRITICAL: {team_name} column mismatch")

        # national_team_id consistency
        if df['national_team_id'].nunique() != 1 or int(df['national_team_id'].iloc[0]) != team_id:
            issues.append(f"CRITICAL: {team_name} national_team_id inconsistent")

        # national_team_name consistency
        if df['national_team_name'].nunique() != 1 or df['national_team_name'].iloc[0] != team_name:
            issues.append(f"CRITICAL: {team_name} national_team_name inconsistent")

        # True duplicate check (NaN-safe using league_name+team_name)
        dups = df.duplicated(
            subset=['player_id', 'season', 'league_name', 'team_name'], keep=False
        ).sum()
        if dups > 0:
            issues.append(f"WARNING: {team_name} has {dups} duplicate rows")

        # Season range
        if not df.empty:
            if df['season'].min() < 2004 or df['season'].max() > 2025:
                issues.append(f"WARNING: {team_name} season out of range "
                              f"[{df['season'].min()}-{df['season'].max()}]")

        print(f"  {team_name:<25} {len(df):>6} rows, "
              f"{df['player_id'].nunique():>4} players, "
              f"seasons {int(df['season'].min())}-{int(df['season'].max())}, "
              f"dups={dups}")

    # Validate master CSV
    master_path = f"{OUTPUT_DIR}/all_player_statistics.csv"
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        # Check master = sum of parts
        total_from_files = 0
        for team_id, (team_name, file_key, wc_years) in PAST_WC_TEAMS.items():
            path = f"{OUTPUT_DIR}/{file_key}_player_statistics.csv"
            if os.path.exists(path):
                total_from_files += len(pd.read_csv(path))

        if len(master) != total_from_files:
            issues.append(f"CRITICAL: Master CSV ({len(master)}) != "
                          f"sum of team files ({total_from_files})")
        else:
            print(f"\n  ✅ Master CSV matches sum of team files: {len(master):,} rows")

        # Cross-team player uniqueness
        player_teams = master.groupby('player_id')['national_team_id'].nunique()
        multi_team = (player_teams > 1).sum()
        if multi_team > 0:
            issues.append(f"WARNING: {multi_team} players assigned to multiple national teams")
            # Show which players
            multi_pids = player_teams[player_teams > 1].index.tolist()
            for pid in multi_pids[:10]:
                rows = master[master['player_id'] == pid][
                    ['player_id', 'player_name', 'national_team_id', 'national_team_name']
                ].drop_duplicates()
                teams_str = ', '.join(
                    f"{r['national_team_name']}(ID:{int(r['national_team_id'])})"
                    for _, r in rows.iterrows()
                )
                print(f"    ⚠️ Player {pid} ({rows.iloc[0]['player_name']}): {teams_str}")

    print(f"\n  {'='*50}")
    print(f"  ISSUES: {len(issues)}")
    if issues:
        for iss in issues:
            is_critical = 'CRITICAL' in iss
            print(f"    {'❌' if is_critical else '⚠️'} {iss}")
    else:
        print("    ✅ ALL CHECKS PASSED!")

    return len(issues) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape past WC teams player statistics"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Count API calls without actually scraping")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation on existing data")
    parser.add_argument("--combine-only", action="store_true",
                        help="Only combine existing team files into master CSV")
    parser.add_argument("--team", type=int, default=None,
                        help="Scrape only a specific team ID (for testing)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset progress and start fresh")
    args = parser.parse_args()

    if args.validate_only:
        phase_validate()
        return

    if args.combine_only:
        tracker = ProgressTracker()
        phase_combine(tracker)
        return

    if args.reset and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("🗑️ Progress reset.")

    tracker = ProgressTracker()

    print("=" * 70)
    print("PAST WORLD CUP TEAMS — PLAYER STATISTICS SCRAPER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Teams: {len(PAST_WC_TEAMS)}")
    print(f"Seasons per player: {len(ALL_SEASONS)} ({min(ALL_SEASONS)}-{max(ALL_SEASONS)})")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"API calls so far: {tracker.api_calls:,}")
    print("=" * 70)

    # If only scraping one team (for testing), filter
    if args.team:
        if args.team not in PAST_WC_TEAMS:
            print(f"❌ Team ID {args.team} not in PAST_WC_TEAMS")
            sys.exit(1)
        filtered = {args.team: PAST_WC_TEAMS[args.team]}
        PAST_WC_TEAMS.clear()
        PAST_WC_TEAMS.update(filtered)
        print(f"  🎯 Single team mode: {PAST_WC_TEAMS[args.team][0]}")

    # Phase 1: Discover players
    phase_discover(tracker)

    # Phase 2: Scrape stats
    completed = phase_scrape(tracker, dry_run=args.dry_run)

    if args.dry_run:
        print("\n✋ Dry run complete. No data was scraped.")
        return

    if not completed:
        print("\n⚠️ Scraping was interrupted. Re-run to resume.")
        sys.exit(0)

    # Phase 3: Combine
    phase_combine(tracker)

    # Phase 4: Validate
    phase_validate()

    print("\n" + "=" * 70)
    print("SCRAPING COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total API calls: {tracker.api_calls:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()

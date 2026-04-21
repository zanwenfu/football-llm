#!/usr/bin/env python3
"""
Scrape complete career statistics for every player who appeared in
World Cup 2010, 2014, 2018, or 2022 match lineups.

Source of truth: data/world_cup_history/world_cup_lineups.csv
    — Contains 2,378 unique player IDs across 53 teams and 4 WC years.

For each player we:
  1. Call /players/seasons to discover which seasons have data.
  2. For each season, call /players?id={id}&season={year} to get all
     statistics (every league/team the player was at that season).

Output:
  data/wc_player_career_stats/
      all_wc_player_career_stats.csv          (master file)
      {team_name}_player_stats.csv            (per WC-team files)
      scrape_progress.json                    (resume state)

Design:
  - Player IDs come ONLY from world_cup_lineups.csv (no hardcoding).
  - Processes players one-by-one (deduplicated across WC years).
  - Full resume capability via JSON progress file.
  - Saves after every player for crash safety.
  - Tracks which WC year(s) and team(s) each player appeared for.

Usage:
    python scrape_wc_player_stats.py              # full run (resume-aware)
    python scrape_wc_player_stats.py --test 5     # test with 5 players
    python scrape_wc_player_stats.py --reset      # wipe progress, start fresh
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from api_client import api_client

# ─── Configuration ───────────────────────────────────────────────────────────

LINEUPS_FILE = "data/world_cup_history/world_cup_lineups.csv"
OUTPUT_DIR = "data/wc_player_career_stats"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "scrape_progress.json")
MASTER_CSV = os.path.join(OUTPUT_DIR, "all_wc_player_career_stats.csv")

# Seasons range to consider  (API-Football has data roughly 2004–2025)
# We let the API tell us which seasons exist per player, but cap at this range.
MIN_SEASON = 2004
MAX_SEASON = 2025

# Save master CSV every N players for crash safety
SAVE_INTERVAL = 10


# ─── Progress tracker ────────────────────────────────────────────────────────

class ProgressTracker:
    """Tracks which players have been fully scraped for resume capability."""

    def __init__(self, path: str = PROGRESS_FILE):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "completed_player_ids": [],
            "failed_player_ids": [],   # players where API returned no seasons
            "api_calls": 0,
            "stats_rows": 0,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    @property
    def completed(self) -> Set[int]:
        return set(self.data["completed_player_ids"])

    @property
    def failed(self) -> Set[int]:
        return set(self.data["failed_player_ids"])

    def mark_completed(self, player_id: int, api_calls: int, rows: int):
        if player_id not in self.data["completed_player_ids"]:
            self.data["completed_player_ids"].append(player_id)
        self.data["api_calls"] += api_calls
        self.data["stats_rows"] += rows

    def mark_failed(self, player_id: int, api_calls: int):
        if player_id not in self.data["failed_player_ids"]:
            self.data["failed_player_ids"].append(player_id)
        self.data["api_calls"] += api_calls

    def reset(self):
        self.data = {
            "completed_player_ids": [],
            "failed_player_ids": [],
            "api_calls": 0,
            "stats_rows": 0,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        self.save()


# ─── Data loading ────────────────────────────────────────────────────────────

def load_lineups() -> pd.DataFrame:
    """Load WC lineups — the single source of truth for player IDs."""
    if not os.path.exists(LINEUPS_FILE):
        raise FileNotFoundError(f"Lineups file not found: {LINEUPS_FILE}")
    df = pd.read_csv(LINEUPS_FILE)
    print(f"📋 Loaded lineups: {len(df)} rows, {df['player_id'].nunique()} unique players")
    return df


def build_player_registry(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a registry of unique players with their WC context.

    Returns DataFrame with columns:
        player_id, player_name, wc_years, wc_teams, wc_team_ids
    Each row = one unique player.  wc_years / wc_teams are comma-separated strings.
    """
    records = []
    grouped = lineups_df.groupby("player_id")

    for player_id, grp in grouped:
        # Take the most recent name (API sometimes changes transliteration)
        name = grp.sort_values("world_cup_year", ascending=False)["player_name"].iloc[0]
        wc_years = sorted(grp["world_cup_year"].unique())
        # team per WC year (player may represent different teams — very rare)
        team_info = grp.drop_duplicates(subset=["world_cup_year", "team_id"])[
            ["world_cup_year", "team_id", "team_name"]
        ].sort_values("world_cup_year")
        wc_team_ids = list(team_info["team_id"].unique())
        wc_teams = list(team_info["team_name"].unique())

        records.append({
            "player_id": int(player_id),
            "player_name": name,
            "wc_years": ",".join(str(y) for y in wc_years),
            "wc_teams": ",".join(wc_teams),
            "wc_team_ids": ",".join(str(t) for t in wc_team_ids),
        })

    registry = pd.DataFrame(records)
    print(f"📝 Player registry: {len(registry)} unique players")
    return registry


# ─── Scraping logic ──────────────────────────────────────────────────────────

def scrape_player_all_seasons(player_id: int, player_name: str) -> tuple:
    """
    Scrape all career statistics for a single player.

    Steps:
        1. GET /players/seasons?player={id}  →  list of season years
        2. For each season: GET /players?id={id}&season={year}
           → parse all league/team entries

    Returns:
        (list_of_stat_dicts, api_call_count)
    """
    api_calls = 0
    all_rows: List[Dict[str, Any]] = []

    # Step 1 — discover available seasons
    try:
        seasons = api_client.get_player_seasons(player_id)
        api_calls += 1
    except Exception as e:
        print(f"    ❌ Error getting seasons for {player_name} (ID {player_id}): {e}")
        return [], 1

    if not seasons:
        # Player exists in lineups but API has no season data
        return [], api_calls

    # Filter to our range
    seasons = [s for s in seasons if MIN_SEASON <= s <= MAX_SEASON]
    seasons.sort()

    # Step 2 — fetch stats for each season
    for season in seasons:
        try:
            response = api_client.get_player_statistics(player_id, season)
            api_calls += 1
        except Exception as e:
            print(f"    ⚠️  Error for {player_name} season {season}: {e}")
            continue

        if not response:
            continue

        player_data = response[0]
        player_info = player_data.get("player", {})
        birth = player_info.get("birth", {})

        for stat_block in player_data.get("statistics", []):
            league = stat_block.get("league", {})
            team = stat_block.get("team", {})
            games = stat_block.get("games", {})
            substitutes = stat_block.get("substitutes", {})
            shots = stat_block.get("shots", {})
            goals_data = stat_block.get("goals", {})
            passes = stat_block.get("passes", {})
            tackles = stat_block.get("tackles", {})
            duels = stat_block.get("duels", {})
            dribbles = stat_block.get("dribbles", {})
            fouls = stat_block.get("fouls", {})
            cards = stat_block.get("cards", {})
            penalty = stat_block.get("penalty", {})

            all_rows.append({
                # ── Player bio ──
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
                # ── Season / Team / League ──
                "season": season,
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "league_country": league.get("country"),
                # ── Games ──
                "position": games.get("position"),
                "appearances": games.get("appearences"),  # API typo
                "lineups": games.get("lineups"),
                "minutes": games.get("minutes"),
                "rating": games.get("rating"),
                "captain": games.get("captain"),
                # ── Substitutes ──
                "substitutes_in": substitutes.get("in"),
                "substitutes_out": substitutes.get("out"),
                "substitutes_bench": substitutes.get("bench"),
                # ── Shooting ──
                "shots_total": shots.get("total"),
                "shots_on_target": shots.get("on"),
                # ── Goals ──
                "goals": goals_data.get("total"),
                "goals_conceded": goals_data.get("conceded"),
                "assists": goals_data.get("assists"),
                "saves": goals_data.get("saves"),
                # ── Passing ──
                "passes_total": passes.get("total"),
                "passes_key": passes.get("key"),
                "passes_accuracy": passes.get("accuracy"),
                # ── Defensive ──
                "tackles_total": tackles.get("total"),
                "tackles_blocks": tackles.get("blocks"),
                "tackles_interceptions": tackles.get("interceptions"),
                # ── Duels ──
                "duels_total": duels.get("total"),
                "duels_won": duels.get("won"),
                # ── Dribbles ──
                "dribbles_attempts": dribbles.get("attempts"),
                "dribbles_success": dribbles.get("success"),
                "dribbles_past": dribbles.get("past"),
                # ── Fouls ──
                "fouls_drawn": fouls.get("drawn"),
                "fouls_committed": fouls.get("committed"),
                # ── Cards ──
                "cards_yellow": cards.get("yellow"),
                "cards_yellowred": cards.get("yellowred"),
                "cards_red": cards.get("red"),
                # ── Penalties ──
                "penalty_won": penalty.get("won"),
                "penalty_committed": penalty.get("commited"),  # API typo
                "penalty_scored": penalty.get("scored"),
                "penalty_missed": penalty.get("missed"),
                "penalty_saved": penalty.get("saved"),
            })

    return all_rows, api_calls


# ─── Saving helpers ──────────────────────────────────────────────────────────

def save_master_csv(all_rows: List[dict], registry: pd.DataFrame):
    """Save the master CSV with WC context columns joined from registry."""
    if not all_rows:
        print("  ⚠️  No rows to save.")
        return

    df = pd.DataFrame(all_rows)

    # Join WC context from registry
    ctx = registry[["player_id", "wc_years", "wc_teams", "wc_team_ids"]].copy()
    df = df.merge(ctx, on="player_id", how="left")

    # De-duplicate just in case (same player_id + season + league_id + team_id)
    before = len(df)
    df.drop_duplicates(
        subset=["player_id", "season", "league_id", "team_id"],
        keep="first",
        inplace=True,
    )
    after = len(df)
    if before != after:
        print(f"  ⚠️  Removed {before - after} duplicate rows")

    df.to_csv(MASTER_CSV, index=False)
    print(f"  💾 Saved master CSV: {len(df)} rows → {MASTER_CSV}")


def save_per_team_csvs(registry: pd.DataFrame):
    """
    Split the master CSV into per-WC-team files.
    A player appears in every team file for teams they represented.
    """
    if not os.path.exists(MASTER_CSV):
        return

    master = pd.read_csv(MASTER_CSV)

    # Build mapping: team_name → set of player_ids
    team_players: Dict[str, Set[int]] = {}
    for _, row in registry.iterrows():
        for team in str(row["wc_teams"]).split(","):
            team = team.strip()
            if team:
                team_players.setdefault(team, set()).add(int(row["player_id"]))

    for team_name, pids in sorted(team_players.items()):
        subset = master[master["player_id"].isin(pids)]
        if subset.empty:
            continue
        safe = team_name.replace(" ", "_").replace("&", "and").lower()
        path = os.path.join(OUTPUT_DIR, f"{safe}_player_stats.csv")
        subset.to_csv(path, index=False)
        print(f"  📁 {team_name}: {len(subset)} rows, {subset['player_id'].nunique()} players → {path}")


# ─── Main orchestrator ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape WC player career stats")
    parser.add_argument("--test", type=int, default=0,
                        help="Only scrape N players (for testing)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset progress and start fresh")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load source of truth ──
    lineups_df = load_lineups()
    registry = build_player_registry(lineups_df)

    # ── Progress ──
    tracker = ProgressTracker()
    if args.reset:
        print("🔄 Resetting progress...")
        tracker.reset()

    # ── Determine work list ──
    all_player_ids = registry["player_id"].tolist()
    already_done = tracker.completed | tracker.failed
    remaining = [pid for pid in all_player_ids if pid not in already_done]

    if args.test > 0:
        remaining = remaining[: args.test]
        print(f"\n🧪 TEST MODE: processing only {len(remaining)} players")

    total_players = len(all_player_ids)
    done_count = len(already_done)

    print(f"\n{'='*70}")
    print(f"🏆 World Cup Player Career Stats Scraper")
    print(f"{'='*70}")
    print(f"  Total unique players : {total_players}")
    print(f"  Already completed    : {len(tracker.completed)}")
    print(f"  Previously failed    : {len(tracker.failed)}")
    print(f"  Remaining to scrape  : {len(remaining)}")
    print(f"  API calls so far     : {tracker.data['api_calls']}")
    print(f"  Stats rows so far    : {tracker.data['stats_rows']}")
    print(f"{'='*70}\n")

    if not remaining:
        print("✅ All players already scraped! Generating final files...")
        # Reload all completed data from per-player accumulation
        if os.path.exists(MASTER_CSV):
            save_per_team_csvs(registry)
            print("\n🎉 Done!")
        return

    # ── Check API status ──
    try:
        status = api_client.get_account_status()
        req_info = status.get("response", {}).get("requests", {})
        current = req_info.get("current", "?")
        limit = req_info.get("limit_day", "?")
        print(f"📡 API status: {current}/{limit} requests used today\n")
    except Exception as e:
        print(f"⚠️  Could not check API status: {e}\n")

    # ── Load existing rows if resuming ──
    existing_rows: List[dict] = []
    if os.path.exists(MASTER_CSV) and not args.reset:
        existing_df = pd.read_csv(MASTER_CSV)
        # Remove WC context columns (will re-join on save)
        drop_cols = [c for c in ["wc_years", "wc_teams", "wc_team_ids"] if c in existing_df.columns]
        if drop_cols:
            existing_df.drop(columns=drop_cols, inplace=True)
        existing_rows = existing_df.to_dict("records")
        print(f"📂 Loaded {len(existing_rows)} existing rows from master CSV\n")

    # ── Scrape loop ──
    new_rows: List[dict] = []
    players_since_save = 0
    scrape_start = time.time()

    for i, player_id in enumerate(remaining):
        player_row = registry[registry["player_id"] == player_id].iloc[0]
        player_name = player_row["player_name"]
        wc_info = player_row["wc_years"]

        progress_pct = (done_count + i + 1) / total_players * 100
        print(
            f"[{done_count + i + 1}/{total_players}] ({progress_pct:.1f}%) "
            f"{player_name} (ID {player_id}, WC {wc_info})"
        )

        rows, calls = scrape_player_all_seasons(player_id, player_name)

        if rows:
            new_rows.extend(rows)
            tracker.mark_completed(player_id, calls, len(rows))
            print(f"    ✅ {len(rows)} stat rows across {len(set(r['season'] for r in rows))} seasons ({calls} API calls)")
        else:
            tracker.mark_failed(player_id, calls)
            print(f"    ⚠️  No data returned ({calls} API calls)")

        players_since_save += 1

        # Periodic save
        if players_since_save >= SAVE_INTERVAL:
            print(f"\n  💾 Auto-saving ({len(existing_rows) + len(new_rows)} total rows)...")
            save_master_csv(existing_rows + new_rows, registry)
            tracker.save()
            players_since_save = 0
            elapsed = time.time() - scrape_start
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            eta_min = (len(remaining) - i - 1) / rate if rate > 0 else 0
            print(f"  ⏱️  Rate: {rate:.1f} players/min | ETA: {eta_min:.0f} min\n")

    # ── Final save ──
    all_rows = existing_rows + new_rows
    print(f"\n{'='*70}")
    print(f"📊 Scraping complete!")
    print(f"  New rows collected    : {len(new_rows)}")
    print(f"  Total rows (all time) : {len(all_rows)}")
    print(f"  Total API calls       : {tracker.data['api_calls']}")
    print(f"{'='*70}\n")

    save_master_csv(all_rows, registry)
    tracker.save()

    # ── Per-team CSVs ──
    print("\n📁 Generating per-team CSV files...")
    save_per_team_csvs(registry)

    # ── Summary ──
    if os.path.exists(MASTER_CSV):
        final = pd.read_csv(MASTER_CSV)
        print(f"\n{'='*70}")
        print(f"✅ FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"  Master CSV rows       : {len(final)}")
        print(f"  Unique players        : {final['player_id'].nunique()}")
        print(f"  Seasons covered       : {sorted(final['season'].unique())}")
        print(f"  Unique teams/clubs    : {final['team_name'].nunique()}")
        print(f"  Unique leagues        : {final['league_name'].nunique()}")
        print(f"  Players completed     : {len(tracker.completed)}")
        print(f"  Players failed (no data): {len(tracker.failed)}")
        print(f"{'='*70}")

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()

"""
Step 2: Build team profiles for each WC match.

For each match, looks up the starting XI, aggregates their prior-season
league stats into a team profile, and enriches with prior WC performance.
"""

import logging
import os

import numpy as np
import pandas as pd

from football_llm.paths import PROCESSED_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    """Load all required datasets."""
    matches = pd.read_csv(os.path.join(RAW_DIR, "world_cup_matches.csv"))
    lineups = pd.read_csv(os.path.join(RAW_DIR, "world_cup_lineups.csv"))
    events = pd.read_csv(os.path.join(RAW_DIR, "world_cup_events.csv"))
    team_stats = pd.read_csv(os.path.join(RAW_DIR, "world_cup_team_match_stats.csv"))
    player_stats = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))

    logger.info(
        f"Loaded: {len(matches)} matches, {len(lineups)} lineups, "
        f"{len(events)} events, {len(team_stats)} team_stats, "
        f"{len(player_stats)} player_season_stats"
    )
    return matches, lineups, events, team_stats, player_stats


def get_starters_for_match(lineups: pd.DataFrame, fixture_id: int, team_id: int) -> pd.DataFrame:
    """Get starting XI for a team in a specific match."""
    mask = (
        (lineups["fixture_id"] == fixture_id)
        & (lineups["team_id"] == team_id)
        & (lineups["is_starter"] == True)  # noqa: E712 — pandas boolean mask idiom
    )
    return lineups[mask]


def get_player_prior_stats(
    player_stats: pd.DataFrame, player_id: int, wc_year: int, lookback_years: int = 3
) -> pd.Series:
    """
    Get a player's aggregated stats from seasons PRIOR to the WC year.
    Uses a lookback window (default 3 years) to capture recent form.
    E.g., for WC 2018, uses 2015-2017 season data.
    """
    min_season = wc_year - lookback_years
    mask = (
        (player_stats["player_id"] == player_id)
        & (player_stats["season"] >= min_season)
        & (player_stats["season"] < wc_year)
    )
    prior = player_stats[mask]

    if prior.empty:
        return pd.Series(dtype=float)

    # Aggregate across the lookback window
    numeric_cols = prior.select_dtypes(include=[np.number]).columns
    # Sum counting stats, average rate stats
    rate_cols = [
        "rating",
        "passes_accuracy",
        "goals_per_90",
        "assists_per_90",
        "goal_contributions_per_90",
        "shot_accuracy",
        "duel_win_pct",
        "tackles_per_90",
        "dribble_success_pct",
    ]
    result = {}
    for col in numeric_cols:
        if col in ("player_id", "season"):
            continue
        if col in rate_cols:
            result[col] = prior[col].mean()
        else:
            result[col] = prior[col].sum()

    result["seasons_with_data"] = len(prior)
    result["player_name"] = prior.iloc[-1].get("player_name", "Unknown")
    result["position"] = prior.iloc[-1].get("position", "Unknown")
    result["nationality"] = prior.iloc[-1].get("nationality", "Unknown")

    return pd.Series(result)


def build_team_profile(
    lineups: pd.DataFrame, player_stats: pd.DataFrame, fixture_id: int, team_id: int, wc_year: int
) -> dict:
    """
    Build a team profile by aggregating the starting XI's prior-season stats.
    Returns a dict with team-level aggregated statistics.
    """
    starters = get_starters_for_match(lineups, fixture_id, team_id)
    if starters.empty:
        return {}

    player_profiles = []
    for _, player in starters.iterrows():
        stats = get_player_prior_stats(player_stats, player["player_id"], wc_year)
        if not stats.empty:
            stats["player_name"] = player["player_name"]
            # Use lineup position if available, otherwise fall back to career stats position
            lineup_pos = player.get("player_position", None)
            if pd.isna(lineup_pos) or not lineup_pos or str(lineup_pos) == "nan":
                lineup_pos = stats.get("position", "Unknown")
            stats["position"] = lineup_pos if lineup_pos and str(lineup_pos) != "nan" else "Unknown"
            stats["player_number"] = player.get("player_number", "")
            player_profiles.append(stats)

    if not player_profiles:
        return {}

    profiles_df = pd.DataFrame(player_profiles)

    # Aggregate team-level stats
    profile = {
        "team_id": team_id,
        "team_name": starters.iloc[0]["team_name"],
        "formation": starters.iloc[0].get("formation", "Unknown"),
        "coach": starters.iloc[0].get("coach_name", "Unknown"),
        "num_starters_with_data": len(profiles_df),
        "total_starters": len(starters),
    }

    # Team aggregates from player stats
    stat_keys = {
        "goals": "sum",
        "assists": "sum",
        "appearances": "sum",
        "minutes": "sum",
        "rating": "mean",
        "passes_accuracy": "mean",
        "goals_per_90": "mean",
        "assists_per_90": "mean",
        "goal_contributions_per_90": "mean",
        "tackles_per_90": "mean",
        "duel_win_pct": "mean",
        "shots_total": "sum",
        "shots_on_target": "sum",
        "cards_yellow": "sum",
        "cards_red": "sum",
        "dribble_success_pct": "mean",
    }

    for col, agg_fn in stat_keys.items():
        if col in profiles_df.columns:
            vals = profiles_df[col].dropna()
            if not vals.empty:
                profile[f"team_avg_{col}" if agg_fn == "mean" else f"team_total_{col}"] = (
                    round(vals.mean(), 2) if agg_fn == "mean" else round(vals.sum(), 1)
                )

    # Position breakdown
    positions = (
        profiles_df["position"].value_counts().to_dict()
        if "position" in profiles_df.columns
        else {}
    )
    profile["position_breakdown"] = positions

    # Individual player summaries (top contributors)
    profile["player_summaries"] = []
    for _, p in profiles_df.iterrows():
        summary = {
            "name": p.get("player_name", "Unknown"),
            "position": p.get("position", "Unknown"),
        }
        for col in [
            "goals",
            "assists",
            "appearances",
            "rating",
            "minutes",
            "goals_per_90",
            "passes_accuracy",
        ]:
            if col in p.index and pd.notna(p[col]):
                summary[col] = round(float(p[col]), 2)
        profile["player_summaries"].append(summary)

    return profile


def get_team_prior_wc_stats(team_stats: pd.DataFrame, team_name: str, wc_year: int) -> list:
    """
    Get a team's prior WC match statistics (from previous tournaments
    and earlier matches in the current tournament).
    """
    prior = team_stats[
        (team_stats["team_name"] == team_name) & (team_stats["world_cup_year"] < wc_year)
    ]

    if prior.empty:
        return []

    records = []
    for _, row in prior.iterrows():
        records.append(
            {
                "wc_year": row["world_cup_year"],
                "opponent": row["opponent_name"],
                "stage": row.get("competition_stage", "Unknown"),
                "goals_scored": row.get("goals_scored", 0),
                "goals_conceded": row.get("goals_conceded", 0),
                "result": row.get("match_result", "Unknown"),
                "xg": row.get("xg", None),
                "possession": row.get("ball_possession_pct", None),
                "total_shots": row.get("total_shots", None),
            }
        )

    return records


def get_match_events(events: pd.DataFrame, fixture_id: int) -> dict:
    """Get structured match events (goals, cards, subs) for a match."""
    match_events = events[events["fixture_id"] == fixture_id]
    if match_events.empty:
        return {}

    result = {"goals": [], "cards": [], "substitutions": []}

    for _, e in match_events.iterrows():
        event_info = {
            "team": e["team_name"],
            "player": e["player_name"],
            "time": e["time_elapsed"],
            "time_extra": e.get("time_extra", ""),
        }

        if e["event_type"] == "Goal":
            event_info["type"] = e.get("goal_type", "Normal")
            event_info["assist"] = e.get("assist_name", "")
            result["goals"].append(event_info)
        elif e["event_type"] == "Card":
            event_info["card_type"] = e.get("card_type", e.get("event_detail", ""))
            result["cards"].append(event_info)
        elif e["event_type"] == "Substitution":
            event_info["player_in"] = e.get("player_in", "")
            event_info["player_out"] = e.get("player_out", "")
            result["substitutions"].append(event_info)

    return result


def get_h2h_record(matches: pd.DataFrame, team1: str, team2: str, before_year: int) -> dict:
    """Get historical head-to-head record between two teams in WC matches."""
    h2h = matches[
        (matches["world_cup_year"] < before_year)
        & (
            ((matches["home_team_name"] == team1) & (matches["away_team_name"] == team2))
            | ((matches["home_team_name"] == team2) & (matches["away_team_name"] == team1))
        )
    ]

    if h2h.empty:
        return {"matches": 0}

    team1_wins = 0
    team2_wins = 0
    draws = 0

    for _, m in h2h.iterrows():
        if m["match_result"] == "draw":
            draws += 1
        elif m["match_result"] == "home_win":
            if m["home_team_name"] == team1:
                team1_wins += 1
            else:
                team2_wins += 1
        elif m["match_result"] == "away_win":
            if m["away_team_name"] == team1:
                team1_wins += 1
            else:
                team2_wins += 1

    return {
        "matches": len(h2h),
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "draws": draws,
    }


def build_match_context(
    match_row: pd.Series,
    lineups: pd.DataFrame,
    player_stats: pd.DataFrame,
    team_stats: pd.DataFrame,
    events: pd.DataFrame,
    matches: pd.DataFrame,
) -> dict:
    """
    Build complete context for a single match, including both teams' profiles,
    prior WC performance, H2H record, and actual match outcome.
    """
    fixture_id = match_row["fixture_id"]
    wc_year = match_row["world_cup_year"]
    home_team = match_row["home_team_name"]
    away_team = match_row["away_team_name"]
    home_team_id = match_row["home_team_id"]
    away_team_id = match_row["away_team_id"]

    # Build team profiles from starting XI's prior stats
    home_profile = build_team_profile(lineups, player_stats, fixture_id, home_team_id, wc_year)
    away_profile = build_team_profile(lineups, player_stats, fixture_id, away_team_id, wc_year)

    # Prior WC stats (only available for 2018+ in team_stats)
    home_prior_wc = get_team_prior_wc_stats(team_stats, home_team, wc_year)
    away_prior_wc = get_team_prior_wc_stats(team_stats, away_team, wc_year)

    # Head-to-head
    h2h = get_h2h_record(matches, home_team, away_team, wc_year)

    # Match events (actual outcome)
    match_events = get_match_events(events, fixture_id)

    # Match result
    home_goals = int(match_row["home_goals"])
    away_goals = int(match_row["away_goals"])
    if home_goals > away_goals:
        result = "home_win"
    elif home_goals < away_goals:
        result = "away_win"
    else:
        result = "draw"

    context = {
        "fixture_id": fixture_id,
        "world_cup_year": wc_year,
        "date": match_row["date"],
        "round": match_row["round"],
        "venue": match_row.get("venue_name", "Unknown"),
        "venue_city": match_row.get("venue_city", "Unknown"),
        "referee": match_row.get("referee", "Unknown"),
        "home_team": home_team,
        "away_team": away_team,
        "home_profile": home_profile,
        "away_profile": away_profile,
        "home_prior_wc": home_prior_wc,
        "away_prior_wc": away_prior_wc,
        "h2h": h2h,
        # Outcome (ground truth)
        "result": result,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "halftime_home": match_row.get("halftime_home", ""),
        "halftime_away": match_row.get("halftime_away", ""),
        "went_to_extra_time": match_row.get("went_to_extra_time", False),
        "went_to_penalties": match_row.get("went_to_penalties", False),
        "events": match_events,
    }

    return context


def run():
    """Build team profiles for all matches and save."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    matches, lineups, events, team_stats, player_stats = load_data()

    all_contexts = []
    for idx, match in matches.iterrows():
        context = build_match_context(match, lineups, player_stats, team_stats, events, matches)
        all_contexts.append(context)

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(matches)} matches")

    logger.info(f"Built contexts for {len(all_contexts)} matches")

    # Save as JSON
    import json

    out_path = os.path.join(PROCESSED_DIR, "match_contexts.json")
    with open(out_path, "w") as f:
        json.dump(all_contexts, f, indent=2, default=str)
    logger.info(f"Saved match contexts to {out_path}")

    return all_contexts


if __name__ == "__main__":
    run()

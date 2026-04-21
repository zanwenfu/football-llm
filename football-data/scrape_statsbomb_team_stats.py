#!/usr/bin/env python3
"""
Scrape team-level match statistics for FIFA World Cup 2018 & 2022
from StatsBomb open data.

StatsBomb provides extremely granular event-level data (29 event types,
~4000 events per match). This script aggregates those events into
team-level per-match statistics with ~60+ columns including:
- Standard stats: shots, passes, fouls, corners, cards
- Detailed stats: long passes, goal kicks, free kicks, throw-ins, handballs
- Advanced stats: xG, clearances, interceptions, ball recoveries, pressures
- ML-engineered features: shot accuracy, pass completion, xG differential, etc.

Data source: https://github.com/statsbomb/open-data (free, CC BY 4.0)
Output: data/world_cup_history/world_cup_team_match_stats_v2.csv
"""

import json
import urllib.request
import pandas as pd
import numpy as np
import time
import os
from collections import Counter, defaultdict
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID = 43  # FIFA World Cup
SEASON_IDS = {2018: 3, 2022: 106}
OUTPUT_DIR = "/Users/zanwenfu/IdeaProject/football-data/data/world_cup_history"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "world_cup_team_match_stats_v2.csv")
LONG_PASS_THRESHOLD = 32.0  # meters — standard threshold for "long pass"
RATE_LIMIT_DELAY = 0.1  # seconds between requests (be polite to GitHub)


def fetch_json(url):
    """Fetch JSON from URL with retry logic."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'football-data-scraper/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < 2:
                print(f"  Retry {attempt+1} for {url}: {e}")
                time.sleep(2 ** attempt)
            else:
                raise
    return None


def fetch_matches(competition_id, season_id):
    """Fetch all matches for a competition/season."""
    url = f"{STATSBOMB_BASE}/matches/{competition_id}/{season_id}.json"
    return fetch_json(url)


def fetch_events(match_id):
    """Fetch all events for a match."""
    url = f"{STATSBOMB_BASE}/events/{match_id}.json"
    return fetch_json(url)


def aggregate_team_stats(events, team_name, opponent_name, match_info):
    """
    Aggregate event-level data into team-level statistics.
    
    Returns a dict with ~50+ raw statistics for the given team.
    """
    # Filter events by team
    team_events = [e for e in events if e.get('team', {}).get('name') == team_name]
    opp_events = [e for e in events if e.get('team', {}).get('name') == opponent_name]
    
    stats = {}
    
    # ── MATCH INFO ──
    stats['statsbomb_match_id'] = match_info['match_id']
    stats['world_cup_year'] = match_info['world_cup_year']
    stats['date'] = match_info['match_date']
    stats['kick_off'] = match_info.get('kick_off', '')
    stats['competition_stage'] = match_info.get('competition_stage', {}).get('name', '')
    stats['stadium'] = match_info.get('stadium', {}).get('name', '')
    stats['referee'] = match_info.get('referee', {}).get('name', '')
    stats['team_name'] = team_name
    stats['opponent_name'] = opponent_name
    
    # Determine home/away
    home_team = match_info['home_team']['home_team_name']
    away_team = match_info['away_team']['away_team_name']
    stats['is_home'] = 1 if team_name == home_team else 0
    
    # Goals
    home_score = match_info.get('home_score', 0)
    away_score = match_info.get('away_score', 0)
    if team_name == home_team:
        stats['goals_scored'] = home_score
        stats['goals_conceded'] = away_score
    else:
        stats['goals_scored'] = away_score
        stats['goals_conceded'] = home_score
    
    # ── SHOTS ──
    team_shots = [e for e in team_events if e['type']['name'] == 'Shot']
    opp_shots = [e for e in opp_events if e['type']['name'] == 'Shot']
    
    stats['total_shots'] = len(team_shots)
    stats['shots_on_target'] = len([s for s in team_shots 
        if s.get('shot', {}).get('outcome', {}).get('name') in ('Goal', 'Saved')])
    stats['shots_off_target'] = len([s for s in team_shots 
        if s.get('shot', {}).get('outcome', {}).get('name') in ('Off T', 'Wayward', 'Post')])
    stats['shots_blocked'] = len([s for s in team_shots 
        if s.get('shot', {}).get('outcome', {}).get('name') == 'Blocked'])
    
    # Shot types
    stats['shots_open_play'] = len([s for s in team_shots 
        if s.get('shot', {}).get('type', {}).get('name') == 'Open Play'])
    stats['shots_free_kick'] = len([s for s in team_shots 
        if s.get('shot', {}).get('type', {}).get('name') == 'Free Kick'])
    stats['shots_penalty'] = len([s for s in team_shots 
        if s.get('shot', {}).get('type', {}).get('name') == 'Penalty'])
    
    # Shot body parts
    stats['shots_right_foot'] = len([s for s in team_shots 
        if s.get('shot', {}).get('body_part', {}).get('name') == 'Right Foot'])
    stats['shots_left_foot'] = len([s for s in team_shots 
        if s.get('shot', {}).get('body_part', {}).get('name') == 'Left Foot'])
    stats['shots_header'] = len([s for s in team_shots 
        if s.get('shot', {}).get('body_part', {}).get('name') == 'Head'])
    
    # xG (expected goals)
    team_xg = sum(s.get('shot', {}).get('statsbomb_xg', 0) for s in team_shots)
    opp_xg = sum(s.get('shot', {}).get('statsbomb_xg', 0) for s in opp_shots)
    stats['xg'] = round(team_xg, 4)
    stats['xg_conceded'] = round(opp_xg, 4)
    
    # ── PASSES ──
    team_passes = [e for e in team_events if e['type']['name'] == 'Pass']
    
    # Total passes and completion
    complete_passes = [p for p in team_passes if 'outcome' not in p.get('pass', {})]
    incomplete_passes = [p for p in team_passes if 'outcome' in p.get('pass', {})]
    stats['total_passes'] = len(team_passes)
    stats['passes_completed'] = len(complete_passes)
    stats['passes_incomplete'] = len(incomplete_passes)
    
    # Pass lengths
    pass_lengths = [p['pass'].get('length', 0) for p in team_passes if 'pass' in p]
    stats['avg_pass_length'] = round(np.mean(pass_lengths), 2) if pass_lengths else 0
    
    # Long passes (>32m)
    long_passes = [p for p in team_passes if p.get('pass', {}).get('length', 0) > LONG_PASS_THRESHOLD]
    long_complete = [p for p in long_passes if 'outcome' not in p.get('pass', {})]
    stats['long_passes'] = len(long_passes)
    stats['long_passes_completed'] = len(long_complete)
    
    # Short passes (<=32m, excluding set pieces)
    short_passes = [p for p in team_passes 
                    if p.get('pass', {}).get('length', 0) <= LONG_PASS_THRESHOLD
                    and p.get('pass', {}).get('type', {}).get('name') not in 
                        ('Corner', 'Free Kick', 'Goal Kick', 'Throw-in', 'Kick Off')]
    stats['short_passes'] = len(short_passes)
    
    # Pass height
    stats['ground_passes'] = len([p for p in team_passes 
        if p.get('pass', {}).get('height', {}).get('name') == 'Ground Pass'])
    stats['low_passes'] = len([p for p in team_passes 
        if p.get('pass', {}).get('height', {}).get('name') == 'Low Pass'])
    stats['high_passes'] = len([p for p in team_passes 
        if p.get('pass', {}).get('height', {}).get('name') == 'High Pass'])
    
    # Through balls
    stats['through_balls'] = len([p for p in team_passes 
        if p.get('pass', {}).get('technique', {}).get('name') == 'Through Ball'])
    
    # Cross passes (passes into the box from wide areas)
    stats['crosses'] = len([p for p in team_passes 
        if p.get('pass', {}).get('cross', False)])
    
    # ── SET PIECES ──
    stats['corners'] = len([p for p in team_passes 
        if p.get('pass', {}).get('type', {}).get('name') == 'Corner'])
    stats['free_kicks'] = len([p for p in team_passes 
        if p.get('pass', {}).get('type', {}).get('name') == 'Free Kick'])
    stats['goal_kicks'] = len([p for p in team_passes 
        if p.get('pass', {}).get('type', {}).get('name') == 'Goal Kick'])
    stats['throw_ins'] = len([p for p in team_passes 
        if p.get('pass', {}).get('type', {}).get('name') == 'Throw-in'])
    stats['kick_offs'] = len([p for p in team_passes 
        if p.get('pass', {}).get('type', {}).get('name') == 'Kick Off'])
    
    # ── FOULS & DISCIPLINE ──
    team_fouls_committed = [e for e in team_events if e['type']['name'] == 'Foul Committed']
    team_fouls_won = [e for e in team_events if e['type']['name'] == 'Foul Won']
    
    stats['fouls_committed'] = len(team_fouls_committed)
    stats['fouls_won'] = len(team_fouls_won)
    
    # Handball fouls
    stats['handballs'] = len([f for f in team_fouls_committed 
        if f.get('foul_committed', {}).get('type', {}).get('name') == 'Handball'])
    
    # Cards
    yellow_cards = 0
    red_cards = 0
    second_yellow = 0
    for e in team_events:
        card = None
        if e['type']['name'] == 'Foul Committed' and 'foul_committed' in e:
            card = e['foul_committed'].get('card', {}).get('name', '')
        elif e['type']['name'] == 'Bad Behaviour' and 'bad_behaviour' in e:
            card = e['bad_behaviour'].get('card', {}).get('name', '')
        if card:
            if card == 'Yellow Card':
                yellow_cards += 1
            elif card == 'Red Card':
                red_cards += 1
            elif card == 'Second Yellow':
                second_yellow += 1
                yellow_cards += 1  # Count as yellow too
                red_cards += 1     # Results in red
    
    stats['yellow_cards'] = yellow_cards
    stats['red_cards'] = red_cards
    stats['second_yellows'] = second_yellow
    
    # ── OFFSIDES ──
    # Offsides are tracked via pass outcomes
    stats['offsides'] = len([p for p in team_passes 
        if p.get('pass', {}).get('outcome', {}).get('name') == 'Pass Offside'])
    
    # ── DEFENSIVE ACTIONS ──
    stats['clearances'] = len([e for e in team_events if e['type']['name'] == 'Clearance'])
    stats['interceptions'] = len([e for e in team_events if e['type']['name'] == 'Interception'])
    stats['blocks'] = len([e for e in team_events if e['type']['name'] == 'Block'])
    stats['ball_recoveries'] = len([e for e in team_events if e['type']['name'] == 'Ball Recovery'])
    
    # Tackles (from duels)
    team_duels = [e for e in team_events if e['type']['name'] == 'Duel']
    stats['tackles'] = len([d for d in team_duels 
        if d.get('duel', {}).get('type', {}).get('name') == 'Tackle'])
    stats['tackles_won'] = len([d for d in team_duels 
        if d.get('duel', {}).get('type', {}).get('name') == 'Tackle'
        and d.get('duel', {}).get('outcome', {}).get('name') in ('Won', 'Success In Play')])
    
    # Aerial duels
    stats['aerial_duels'] = len([d for d in team_duels 
        if d.get('duel', {}).get('type', {}).get('name') == 'Aerial Lost'])
    # Note: Aerial Lost is from the losing team's perspective in StatsBomb
    # We also need to count opponent's "Aerial Lost" as our aerial wins
    opp_duels = [e for e in opp_events if e['type']['name'] == 'Duel']
    stats['aerial_duels_won'] = len([d for d in opp_duels 
        if d.get('duel', {}).get('type', {}).get('name') == 'Aerial Lost'])
    stats['aerial_duels_lost'] = stats['aerial_duels']
    stats['aerial_duels_total'] = stats['aerial_duels_won'] + stats['aerial_duels_lost']
    
    # ── PRESSING & POSSESSION ──
    stats['pressures'] = len([e for e in team_events if e['type']['name'] == 'Pressure'])
    
    # Dribbles
    team_dribbles = [e for e in team_events if e['type']['name'] == 'Dribble']
    stats['dribbles_attempted'] = len(team_dribbles)
    stats['dribbles_completed'] = len([d for d in team_dribbles 
        if d.get('duel', {}).get('outcome', {}).get('name') in ('Won', 'Complete')])
    
    # Dispossessed
    stats['dispossessed'] = len([e for e in team_events if e['type']['name'] == 'Dispossessed'])
    
    # Miscontrols
    stats['miscontrols'] = len([e for e in team_events if e['type']['name'] == 'Miscontrol'])
    
    # ── GOALKEEPER ──
    team_gk = [e for e in team_events if e['type']['name'] == 'Goal Keeper']
    stats['gk_saves'] = len([g for g in team_gk 
        if g.get('goalkeeper', {}).get('type', {}).get('name') == 'Shot Saved'])
    stats['gk_punches'] = len([g for g in team_gk 
        if g.get('goalkeeper', {}).get('type', {}).get('name') == 'Punch'])
    stats['gk_claims'] = len([g for g in team_gk 
        if g.get('goalkeeper', {}).get('type', {}).get('name') == 'Collected'])
    stats['gk_sweeper'] = len([g for g in team_gk 
        if g.get('goalkeeper', {}).get('type', {}).get('name') == 'Keeper Sweeper'])
    
    # ── POSSESSION (calculated from event timestamps) ──
    # StatsBomb tracks possession_team on every event - we can compute possession %
    all_events_with_possession = [e for e in events if 'possession_team' in e]
    if all_events_with_possession:
        team_poss_events = len([e for e in all_events_with_possession 
            if e['possession_team']['name'] == team_name])
        total_poss_events = len(all_events_with_possession)
        stats['ball_possession_pct'] = round(100 * team_poss_events / total_poss_events, 1)
    else:
        stats['ball_possession_pct'] = 50.0
    
    # Count unique possession sequences
    possession_ids = set()
    for e in events:
        if e.get('possession_team', {}).get('name') == team_name:
            possession_ids.add(e.get('possession', 0))
    stats['possession_sequences'] = len(possession_ids)
    
    # ── SUBSTITUTIONS ──
    stats['substitutions'] = len([e for e in team_events if e['type']['name'] == 'Substitution'])
    
    # ── 50/50s ──
    fifty_fiftys = [e for e in team_events if e['type']['name'] == '50/50']
    stats['fifty_fiftys'] = len(fifty_fiftys)
    
    return stats


def compute_ml_features(df):
    """
    Add ML-engineered features on top of raw statistics.
    """
    # ── SHOOTING EFFICIENCY ──
    df['shot_accuracy'] = np.where(
        df['total_shots'] > 0,
        (df['shots_on_target'] / df['total_shots'] * 100).round(1),
        0.0
    )
    df['shot_conversion_rate'] = np.where(
        df['total_shots'] > 0,
        (df['goals_scored'] / df['total_shots'] * 100).round(1),
        0.0
    )
    df['xg_per_shot'] = np.where(
        df['total_shots'] > 0,
        (df['xg'] / df['total_shots']).round(4),
        0.0
    )
    df['xg_overperformance'] = (df['goals_scored'] - df['xg']).round(4)
    df['xg_differential'] = (df['xg'] - df['xg_conceded']).round(4)
    
    # ── PASSING EFFICIENCY ──
    df['pass_completion_pct'] = np.where(
        df['total_passes'] > 0,
        (df['passes_completed'] / df['total_passes'] * 100).round(1),
        0.0
    )
    df['long_pass_completion_pct'] = np.where(
        df['long_passes'] > 0,
        (df['long_passes_completed'] / df['long_passes'] * 100).round(1),
        0.0
    )
    df['long_pass_ratio'] = np.where(
        df['total_passes'] > 0,
        (df['long_passes'] / df['total_passes'] * 100).round(1),
        0.0
    )
    df['cross_ratio'] = np.where(
        df['total_passes'] > 0,
        (df['crosses'] / df['total_passes'] * 100).round(1),
        0.0
    )
    
    # ── DEFENSIVE EFFICIENCY ──
    df['tackle_success_rate'] = np.where(
        df['tackles'] > 0,
        (df['tackles_won'] / df['tackles'] * 100).round(1),
        0.0
    )
    df['aerial_win_pct'] = np.where(
        df['aerial_duels_total'] > 0,
        (df['aerial_duels_won'] / df['aerial_duels_total'] * 100).round(1),
        0.0
    )
    df['defensive_actions'] = (
        df['tackles'] + df['interceptions'] + df['clearances'] + df['blocks']
    )
    
    # ── PRESSING METRICS ──
    df['pressure_intensity'] = df['pressures']  # Can be compared across matches
    df['recoveries_per_possession_lost'] = np.where(
        df['dispossessed'] + df['miscontrols'] > 0,
        (df['ball_recoveries'] / (df['dispossessed'] + df['miscontrols'])).round(2),
        df['ball_recoveries'].astype(float)
    )
    
    # ── DRIBBLING ──
    df['dribble_success_rate'] = np.where(
        df['dribbles_attempted'] > 0,
        (df['dribbles_completed'] / df['dribbles_attempted'] * 100).round(1),
        0.0
    )
    
    # ── DISCIPLINE INDEX ──
    df['discipline_index'] = (
        df['yellow_cards'] * 1 + df['red_cards'] * 3 + df['fouls_committed'] * 0.1
    ).round(2)
    
    # ── MATCH OUTCOME FEATURES ──
    df['goal_difference'] = df['goals_scored'] - df['goals_conceded']
    df['match_result'] = np.where(
        df['goals_scored'] > df['goals_conceded'], 'W',
        np.where(df['goals_scored'] < df['goals_conceded'], 'L', 'D')
    )
    df['is_clean_sheet'] = (df['goals_conceded'] == 0).astype(int)
    
    # ── POSSESSION-RELATED ──
    df['possession_differential'] = (df['ball_possession_pct'] - 50).round(1)
    
    # ── SET PIECE DEPENDENCY ──
    df['set_piece_passes'] = df['corners'] + df['free_kicks'] + df['throw_ins'] + df['goal_kicks']
    df['set_piece_ratio'] = np.where(
        df['total_passes'] > 0,
        (df['set_piece_passes'] / df['total_passes'] * 100).round(1),
        0.0
    )
    
    # ── GK WORKLOAD ──
    df['gk_total_actions'] = df['gk_saves'] + df['gk_punches'] + df['gk_claims'] + df['gk_sweeper']
    
    return df


def main():
    print("=" * 70)
    print("StatsBomb World Cup Team-Level Match Statistics Scraper")
    print("=" * 70)
    print(f"Target: WC 2018 + 2022 (128 matches, 256 team-rows)")
    print(f"Source: StatsBomb Open Data (CC BY 4.0)")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    all_rows = []
    total_matches = 0
    
    for year, season_id in SEASON_IDS.items():
        print(f"\n{'─' * 50}")
        print(f"World Cup {year} (competition_id={COMPETITION_ID}, season_id={season_id})")
        print(f"{'─' * 50}")
        
        # Fetch match list
        matches = fetch_matches(COMPETITION_ID, season_id)
        print(f"Found {len(matches)} matches")
        
        for i, match in enumerate(matches):
            match_id = match['match_id']
            home_team = match['home_team']['home_team_name']
            away_team = match['away_team']['away_team_name']
            stage = match.get('competition_stage', {}).get('name', '')
            
            print(f"  [{i+1:2d}/{len(matches)}] {home_team} vs {away_team} "
                  f"({match['home_score']}-{match['away_score']}) [{stage}]", end='')
            
            # Fetch events
            events = fetch_events(match_id)
            
            # Add year to match info
            match['world_cup_year'] = year
            
            # Aggregate stats for home team
            home_stats = aggregate_team_stats(events, home_team, away_team, match)
            all_rows.append(home_stats)
            
            # Aggregate stats for away team
            away_stats = aggregate_team_stats(events, away_team, home_team, match)
            all_rows.append(away_stats)
            
            total_matches += 1
            print(f" ✓ ({len(events):,} events)")
            
            time.sleep(RATE_LIMIT_DELAY)
        
        print(f"\n  ✅ {year}: {len(matches)} matches processed")
    
    print(f"\n{'=' * 70}")
    print(f"Total: {total_matches} matches, {len(all_rows)} team-rows")
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    print(f"\nRaw stats shape: {df.shape}")
    
    # Add ML features
    print("Adding ML-engineered features...")
    df = compute_ml_features(df)
    print(f"Final shape with ML features: {df.shape}")
    
    # Sort by year, date, match_id
    df = df.sort_values(['world_cup_year', 'date', 'statsbomb_match_id', 'is_home'], 
                         ascending=[True, True, True, False]).reset_index(drop=True)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved to {OUTPUT_FILE}")
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Matches per year: {df.groupby('world_cup_year')['statsbomb_match_id'].nunique().to_dict()}")
    print(f"Unique teams: {df['team_name'].nunique()}")
    print(f"\nColumn categories:")
    
    match_cols = ['statsbomb_match_id', 'world_cup_year', 'date', 'kick_off', 
                  'competition_stage', 'stadium', 'referee', 'team_name', 
                  'opponent_name', 'is_home', 'goals_scored', 'goals_conceded']
    shot_cols = [c for c in df.columns if 'shot' in c or c.startswith('xg')]
    pass_cols = [c for c in df.columns if 'pass' in c or c in ['avg_pass_length', 'through_balls', 'crosses']]
    setpiece_cols = ['corners', 'free_kicks', 'goal_kicks', 'throw_ins', 'kick_offs']
    discipline_cols = [c for c in df.columns if 'card' in c or 'foul' in c or c in ['handballs', 'offsides']]
    defense_cols = [c for c in df.columns if any(x in c for x in ['clearance', 'interception', 'block', 'recovery', 'tackle', 'aerial', 'pressure'])]
    
    print(f"  Match info: {len(match_cols)} cols")
    print(f"  Shooting: {len(shot_cols)} cols")
    print(f"  Passing: {len(pass_cols)} cols")
    print(f"  Set pieces: {len(setpiece_cols)} cols")
    print(f"  Discipline: {len(discipline_cols)} cols")
    print(f"  Defensive: {len(defense_cols)} cols")
    
    # Print sample stats for verification
    print(f"\n{'─' * 50}")
    print("Sample verification (first match):")
    print(f"{'─' * 50}")
    row = df.iloc[0]
    print(f"  {row['team_name']} vs {row['opponent_name']} ({row['goals_scored']}-{row['goals_conceded']})")
    print(f"  Shots: {row['total_shots']} (on target: {row['shots_on_target']})")
    print(f"  xG: {row['xg']}")
    print(f"  Passes: {row['total_passes']} ({row['pass_completion_pct']}% complete)")
    print(f"  Long passes: {row['long_passes']} ({row['long_pass_completion_pct']}% complete)")
    print(f"  Corners: {row['corners']}, Free kicks: {row['free_kicks']}")
    print(f"  Goal kicks: {row['goal_kicks']}, Throw-ins: {row['throw_ins']}")
    print(f"  Fouls: {row['fouls_committed']}, Handballs: {row['handballs']}")
    print(f"  Possession: {row['ball_possession_pct']}%")
    print(f"  Cards: {row['yellow_cards']}Y {row['red_cards']}R")
    
    # Check for any null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"\n⚠️  Columns with null values:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"    {col}: {count} nulls")
    else:
        print(f"\n✅ No null values in any column")
    
    print(f"\nAll columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col} ({df[col].dtype})")


if __name__ == '__main__':
    main()

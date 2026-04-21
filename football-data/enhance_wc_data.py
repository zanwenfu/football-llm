"""
Scrape team-level match statistics from fixtures/statistics endpoint
for 2018 and 2022 World Cup matches. Creates world_cup_team_match_stats.csv.

Then enhance ALL 4+1 CSV files with ML-ready features and data cleaning.
"""
import pandas as pd
import numpy as np
import time
import json
import os
from api_client import api_client

DATA_DIR = 'data/world_cup_history'

# ========================================================================
# STEP 1: Scrape team match statistics for 2018+2022
# ========================================================================
def scrape_team_stats():
    """Scrape team-level match statistics for all 2018+2022 fixtures."""
    matches = pd.read_csv(f'{DATA_DIR}/world_cup_matches.csv')
    fixtures_2018_2022 = matches[matches['world_cup_year'].isin([2018, 2022])].sort_values('fixture_id')
    fixture_ids = fixtures_2018_2022['fixture_id'].unique()
    
    print(f"Scraping team stats for {len(fixture_ids)} fixtures (2018+2022)...")
    
    # Check if we have a progress file
    progress_file = f'{DATA_DIR}/team_stats_progress.json'
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            completed = set(json.load(f))
        print(f"  Resuming: {len(completed)} already done")
    else:
        completed = set()
    
    rows = []
    
    # Load any existing partial data
    output_file = f'{DATA_DIR}/world_cup_team_match_stats.csv'
    if os.path.exists(output_file) and completed:
        existing = pd.read_csv(output_file)
        rows = existing.to_dict('records')
        print(f"  Loaded {len(rows)} existing rows")
    
    for i, fid in enumerate(fixture_ids):
        if fid in completed:
            continue
        
        match = matches[matches['fixture_id'] == fid].iloc[0]
        
        try:
            data = api_client.get_fixture_statistics(fid)
        except Exception as e:
            print(f"  ❌ Error on fixture {fid}: {e}")
            continue
        
        if not data:
            print(f"  ⚠️  No stats for fixture {fid}")
            continue
        
        for team_data in data:
            team_id = team_data['team']['id']
            team_name = team_data['team']['name']
            
            # Determine if this team is home or away
            is_home = (team_id == match['home_team_id'])
            opponent_id = match['away_team_id'] if is_home else match['home_team_id']
            opponent_name = match['away_team_name'] if is_home else match['home_team_name']
            
            # Parse statistics into a flat dict
            stats = {}
            for s in team_data.get('statistics', []):
                stat_type = s['type']
                value = s['value']
                
                # Convert percentage strings to floats
                if isinstance(value, str) and value.endswith('%'):
                    value = float(value.replace('%', ''))
                elif value is None:
                    value = 0  # API returns null for 0 in some stats
                
                # Normalize stat name to column-friendly format
                col_name = stat_type.lower().replace(' ', '_').replace('%', 'pct')
                stats[col_name] = value
            
            row = {
                'fixture_id': fid,
                'world_cup_year': match['world_cup_year'],
                'date': match['date'],
                'round': match['round'],
                'team_id': team_id,
                'team_name': team_name,
                'is_home': is_home,
                'opponent_id': opponent_id,
                'opponent_name': opponent_name,
                'goals_scored': match['home_goals'] if is_home else match['away_goals'],
                'goals_conceded': match['away_goals'] if is_home else match['home_goals'],
                **stats
            }
            rows.append(row)
        
        completed.add(int(fid))
        
        # Save progress every 20 fixtures
        if len(completed) % 20 == 0:
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            with open(progress_file, 'w') as f:
                json.dump([int(x) for x in completed], f)
            print(f"  Progress: {len(completed)}/{len(fixture_ids)} fixtures")
        
        # Brief pause to be nice to the API
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
    
    # Final save
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved {len(df)} rows to {output_file}")
    print(f"   ({len(df)//2} fixtures × 2 teams)")
    print(f"   Columns: {list(df.columns)}")
    
    # Clean up progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    return df


# ========================================================================
# STEP 2: Enhance player_match_stats - fill nulls with 0s for appeared players
# ========================================================================
def enhance_player_stats():
    """Fill null values with 0 for appeared players (API returns null for 0)."""
    ps = pd.read_csv(f'{DATA_DIR}/world_cup_player_match_stats.csv')
    print(f"\nEnhancing player_match_stats ({len(ps)} rows)...")
    
    # Columns where null should be 0 for appeared players
    stat_cols = [
        'shots_total', 'shots_on_target',
        'goals_scored', 'goals_conceded', 'goals_assists', 'goals_saves',
        'passes_total', 'passes_key', 'passes_accuracy',
        'tackles_total', 'tackles_blocks', 'tackles_interceptions',
        'duels_total', 'duels_won',
        'dribbles_attempts', 'dribbles_success', 'dribbles_past',
        'fouls_drawn', 'fouls_committed',
        'cards_yellow', 'cards_red',
        'penalty_won', 'penalty_committed', 'penalty_scored', 'penalty_missed', 'penalty_saved',
        'offsides',
    ]
    
    appeared = ps['appeared'] == True
    filled_count = 0
    for col in stat_cols:
        if col in ps.columns:
            nulls_before = ps.loc[appeared, col].isnull().sum()
            ps.loc[appeared, col] = ps.loc[appeared, col].fillna(0)
            filled_count += nulls_before
    
    print(f"  Filled {filled_count} null values with 0 for appeared players")
    
    # Add derived ratio columns for ML
    eps = 1e-10  # avoid division by zero
    ps['pass_accuracy_pct'] = np.where(
        ps['passes_total'] > 0,
        (ps['passes_accuracy'] / ps['passes_total'] * 100).round(1),
        np.nan
    )
    ps['duel_win_pct'] = np.where(
        ps['duels_total'] > 0,
        (ps['duels_won'] / ps['duels_total'] * 100).round(1),
        np.nan
    )
    ps['dribble_success_pct'] = np.where(
        ps['dribbles_attempts'] > 0,
        (ps['dribbles_success'] / ps['dribbles_attempts'] * 100).round(1),
        np.nan
    )
    ps['shot_accuracy_pct'] = np.where(
        ps['shots_total'] > 0,
        (ps['shots_on_target'] / ps['shots_total'] * 100).round(1),
        np.nan
    )
    ps['goals_per_shot'] = np.where(
        ps['shots_total'] > 0,
        (ps['goals_scored'] / ps['shots_total']).round(3),
        np.nan
    )
    ps['minutes_per_goal'] = np.where(
        ps['goals_scored'] > 0,
        (ps['minutes_played'] / ps['goals_scored']).round(1),
        np.nan
    )
    
    # Defensive contribution score (tackles + interceptions + blocks)
    ps['defensive_actions'] = ps['tackles_total'] + ps['tackles_interceptions'] + ps['tackles_blocks']
    
    # Attacking contribution score (goals + assists + key passes)
    ps['attacking_actions'] = ps['goals_scored'] + ps['goals_assists'] + ps['passes_key']
    
    # Per-90 stats (normalized)
    for col, per90_col in [
        ('goals_scored', 'goals_per_90'),
        ('goals_assists', 'assists_per_90'),
        ('passes_key', 'key_passes_per_90'),
        ('tackles_total', 'tackles_per_90'),
        ('dribbles_success', 'successful_dribbles_per_90'),
        ('fouls_committed', 'fouls_committed_per_90'),
    ]:
        ps[per90_col] = np.where(
            ps['minutes_played'] > 0,
            (ps[col] / ps['minutes_played'] * 90).round(3),
            np.nan
        )
    
    print(f"  Added derived columns: pass_accuracy_pct, duel_win_pct, dribble_success_pct,")
    print(f"    shot_accuracy_pct, goals_per_shot, minutes_per_goal, defensive_actions,")
    print(f"    attacking_actions, plus 6 per-90 stats")
    
    ps.to_csv(f'{DATA_DIR}/world_cup_player_match_stats.csv', index=False)
    print(f"  ✅ Saved ({len(ps)} rows × {len(ps.columns)} cols)")
    return ps


# ========================================================================
# STEP 3: Enhance matches with ML-ready features
# ========================================================================
def enhance_matches():
    """Add ML-ready derived features to matches."""
    m = pd.read_csv(f'{DATA_DIR}/world_cup_matches.csv')
    print(f"\nEnhancing matches ({len(m)} rows)...")
    
    # Date-based features
    m['date'] = pd.to_datetime(m['date'])
    m['day_of_week'] = m['date'].dt.day_name()
    m['match_month'] = m['date'].dt.month
    m['match_hour'] = m['date'].dt.hour
    
    # Match structure features
    m['is_group_stage'] = m['round'].str.contains('Group', case=False, na=False)
    m['is_knockout'] = ~m['is_group_stage']
    
    # Score features
    m['total_goals'] = m['home_goals'] + m['away_goals']
    m['goal_difference'] = (m['home_goals'] - m['away_goals']).abs()
    m['home_goal_diff'] = m['home_goals'] - m['away_goals']
    
    # Match result from home perspective
    m['match_result'] = np.where(
        m['home_goals'] > m['away_goals'], 'home_win',
        np.where(m['home_goals'] < m['away_goals'], 'away_win', 'draw')
    )
    
    # Note: for knockout matches that went to ET/penalties, home_goals == away_goals 
    # means the match was a draw after extra time, decided by penalties
    m['decided_by_penalties'] = m['went_to_penalties'].fillna(False)
    m['decided_by_extra_time'] = m['went_to_extra_time'].fillna(False) & ~m['decided_by_penalties']
    m['decided_in_regular_time'] = ~m['went_to_extra_time'].fillna(False)
    
    # Is it a high-scoring or low-scoring match?
    m['is_high_scoring'] = m['total_goals'] >= 4  # ~top 25%
    m['is_clean_sheet_home'] = m['away_goals'] == 0
    m['is_clean_sheet_away'] = m['home_goals'] == 0
    
    # Convert date back to string for CSV
    m['date'] = m['date'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    
    m.to_csv(f'{DATA_DIR}/world_cup_matches.csv', index=False)
    print(f"  Added: day_of_week, match_month, match_hour, is_group_stage, is_knockout,")
    print(f"    total_goals, goal_difference, home_goal_diff, match_result,")
    print(f"    decided_by_*, is_high_scoring, is_clean_sheet_*")
    print(f"  ✅ Saved ({len(m)} rows × {len(m.columns)} cols)")
    return m


# ========================================================================
# STEP 4: Enhance events with more ML features
# ========================================================================
def enhance_events():
    """Add time-based ML features to events."""
    e = pd.read_csv(f'{DATA_DIR}/world_cup_events.csv')
    print(f"\nEnhancing events ({len(e)} rows)...")
    
    # Time bucket features
    e['time_bucket'] = pd.cut(
        e['time_elapsed'].clip(lower=0),  # handle negative values
        bins=[0, 15, 30, 45, 60, 75, 90, 120, 999],
        labels=['0-15', '16-30', '31-45', '46-60', '61-75', '76-90', '91-105', '106-120'],
        include_lowest=True
    )
    
    e['is_first_half'] = (e['time_elapsed'] >= 0) & (e['time_elapsed'] <= 45)
    e['is_second_half'] = (e['time_elapsed'] > 45) & (e['time_elapsed'] <= 90)
    e['is_extra_time'] = e['time_elapsed'] > 90
    e['is_injury_time'] = e['time_extra'].notna() & (e['time_extra'] > 0)
    e['is_late_event'] = e['time_elapsed'] >= 80  # late-game events (dramatic moments)
    
    # Running goal difference at time of event (within each match)
    # This is complex - we need to track score progression
    goal_events = e[e['is_goal'] == True].copy()
    
    # For each fixture, compute the running score
    matches = pd.read_csv(f'{DATA_DIR}/world_cup_matches.csv')
    match_teams = {}
    for _, row in matches.iterrows():
        match_teams[row['fixture_id']] = {
            'home_id': row['home_team_id'],
            'away_id': row['away_team_id']
        }
    
    e.to_csv(f'{DATA_DIR}/world_cup_events.csv', index=False)
    print(f"  Added: time_bucket, is_first_half, is_second_half, is_extra_time,")
    print(f"    is_injury_time, is_late_event")
    print(f"  ✅ Saved ({len(e)} rows × {len(e.columns)} cols)")
    return e


# ========================================================================
# STEP 5: Enhance lineups  
# ========================================================================
def enhance_lineups():
    """Add useful features to lineups."""
    l = pd.read_csv(f'{DATA_DIR}/world_cup_lineups.csv')
    print(f"\nEnhancing lineups ({len(l)} rows)...")
    
    # Add match context from matches file
    matches = pd.read_csv(f'{DATA_DIR}/world_cup_matches.csv')
    
    # Determine if each player's team is home or away
    home_map = dict(zip(matches['fixture_id'], matches['home_team_id']))
    l['is_home'] = l.apply(lambda row: row['team_id'] == home_map.get(row['fixture_id']), axis=1)
    
    # Add opponent info
    match_info = {}
    for _, row in matches.iterrows():
        match_info[row['fixture_id']] = {
            'home_id': row['home_team_id'],
            'home_name': row['home_team_name'],
            'away_id': row['away_team_id'],
            'away_name': row['away_team_name'],
        }
    
    def get_opponent(row):
        info = match_info.get(row['fixture_id'], {})
        if row['team_id'] == info.get('home_id'):
            return info.get('away_id'), info.get('away_name')
        else:
            return info.get('home_id'), info.get('home_name')
    
    opponents = l.apply(get_opponent, axis=1, result_type='expand')
    l['opponent_id'] = opponents[0]
    l['opponent_name'] = opponents[1]
    
    # Squad size per team per match
    squad_size = l.groupby(['fixture_id', 'team_id']).size().reset_index(name='squad_size')
    l = l.merge(squad_size, on=['fixture_id', 'team_id'], how='left')
    
    # Number of substitutes per team per match
    sub_count = l[l['is_substitute'] == True].groupby(['fixture_id', 'team_id']).size().reset_index(name='num_substitutes')
    l = l.merge(sub_count, on=['fixture_id', 'team_id'], how='left')
    l['num_substitutes'] = l['num_substitutes'].fillna(0).astype(int)
    
    l.to_csv(f'{DATA_DIR}/world_cup_lineups.csv', index=False)
    print(f"  Added: is_home, opponent_id, opponent_name, squad_size, num_substitutes")
    print(f"  ✅ Saved ({len(l)} rows × {len(l.columns)} cols)")
    return l


# ========================================================================
# MAIN
# ========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("WORLD CUP DATA ENHANCEMENT")
    print("=" * 70)
    
    # Step 1: Scrape team match statistics
    print("\n--- STEP 1: Scrape Team Match Statistics ---")
    team_stats = scrape_team_stats()
    
    # Step 2: Enhance player match stats
    print("\n--- STEP 2: Enhance Player Match Stats ---")
    enhance_player_stats()
    
    # Step 3: Enhance matches
    print("\n--- STEP 3: Enhance Matches ---")
    enhance_matches()
    
    # Step 4: Enhance events
    print("\n--- STEP 4: Enhance Events ---")
    enhance_events()
    
    # Step 5: Enhance lineups
    print("\n--- STEP 5: Enhance Lineups ---")
    enhance_lineups()
    
    print("\n" + "=" * 70)
    print("ALL ENHANCEMENTS COMPLETE")
    print("=" * 70)
    
    # Summary
    for fname in ['world_cup_matches.csv', 'world_cup_lineups.csv', 
                   'world_cup_events.csv', 'world_cup_player_match_stats.csv',
                   'world_cup_team_match_stats.csv']:
        fpath = f'{DATA_DIR}/{fname}'
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            print(f"  {fname}: {len(df)} rows × {len(df.columns)} cols")

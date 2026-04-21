#!/usr/bin/env python3
"""
Script to scrape full data for Ghana national team.
Run with: conda activate football-data && python scrape_ghana.py
"""
from scraper import WorldCup2026Scraper
import pandas as pd

def main():
    scraper = WorldCup2026Scraper()

    # Ghana team info
    team_id = 1504
    team_name = 'Ghana'

    print('='*70)
    print(f'FULL SCRAPE FOR {team_name.upper()} (Team ID: {team_id})')
    print('='*70)

    # Step 1: Get squad
    print('\n📋 Step 1: Fetching squad...')
    squad_df = scraper.scrape_team_squad(team_id, team_name)
    print(f'   Found {len(squad_df)} players')

    # Step 2: Get statistics for all players (seasons 2022-2024 only for free tier)
    print('\n📊 Step 2: Fetching player statistics for seasons 2022-2024...')
    print(f'   Will fetch stats for {len(squad_df)} players x 3 seasons = {len(squad_df) * 3} API calls')
    print(f'   At 10 req/min, estimated time: ~{(len(squad_df) * 3) // 10 + 1} minutes')
    print()

    all_stats = scraper.scrape_all_team_players_statistics(team_id, team_name, squad_df)

    print('\n' + '='*70)
    print('SCRAPING COMPLETE')
    print('='*70)
    print(f'Total statistics records: {len(all_stats)}')
    if not all_stats.empty:
        print(f'Columns collected: {len(all_stats.columns)}')
        print(f'\nColumn names:')
        for i, col in enumerate(all_stats.columns):
            print(f'  {i+1}. {col}')
        print(f'\nData saved to: data/statistics/ghana_player_statistics.csv')
        
        # Show sample data
        print('\n' + '='*70)
        print('SAMPLE DATA (first 5 rows)')
        print('='*70)
        print(all_stats[['player_name', 'season', 'team_name', 'league_name', 'appearances', 'goals', 'assists', 'rating']].head())

if __name__ == "__main__":
    main()

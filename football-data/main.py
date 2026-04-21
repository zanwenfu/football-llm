#!/usr/bin/env python3
"""
Football World Cup 2026 Data Collection Tool

This script collects and organizes football data for World Cup 2026 predictions.
It scrapes team rosters, player statistics, and match data from official APIs.

Usage:
    python main.py [command] [options]
    
Commands:
    scrape      - Run the full data scraping process
    teams       - Scrape only team information
    players     - Scrape player data for a specific team
    stats       - Generate statistics summary
    test        - Run a test scrape with limited data
    status      - Check API key status and rate limits

Examples:
    python main.py scrape
    python main.py teams
    python main.py players --team "Argentina"
    python main.py test --max-teams 3
    python main.py status
"""
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper import WorldCup2026Scraper
from utils import (
    load_all_player_statistics,
    calculate_player_aggregates,
    export_summary_report,
)
from config import API_FOOTBALL_KEYS, REQUESTS_PER_KEY_PER_MINUTE


def check_api_keys():
    """Check if API keys are configured and show status."""
    if not API_FOOTBALL_KEYS:
        print("\n" + "=" * 60)
        print("ERROR: No API Keys Configured")
        print("=" * 60)
        print("\nTo use this tool, you need API-Football API key(s).")
        print("\nSteps to get your API key:")
        print("1. Go to https://www.api-football.com/")
        print("2. Sign up for a free account (100 requests/day)")
        print("3. Or sign up via RapidAPI: https://rapidapi.com/api-sports/api/api-football")
        print("4. Copy your API key")
        print("5. Create a .env file in this directory with:")
        print("   API_FOOTBALL_KEY_1=your_first_api_key")
        print("   API_FOOTBALL_KEY_2=your_second_api_key  # (optional)")
        print("\nAlternatively, you can use the mock data for testing:")
        print("   python main.py test --mock")
        print("=" * 60 + "\n")
        return False
    
    print(f"\n✅ Found {len(API_FOOTBALL_KEYS)} API key(s)")
    print(f"📊 Rate limit: {REQUESTS_PER_KEY_PER_MINUTE} requests/min per key")
    print(f"📊 Total capacity: {len(API_FOOTBALL_KEYS) * REQUESTS_PER_KEY_PER_MINUTE} requests/min")
    return True


def cmd_scrape(args):
    """Run full scraping process."""
    if not check_api_keys():
        return
    
    scraper = WorldCup2026Scraper()
    scraper.run_full_scrape(max_teams=args.max_teams)


def cmd_teams(args):
    """Scrape only teams."""
    if not check_api_keys():
        return
    
    scraper = WorldCup2026Scraper()
    teams_df = scraper.scrape_world_cup_teams()
    print(f"\nScraped {len(teams_df)} teams")
    print(teams_df.to_string())


def cmd_players(args):
    """Scrape players for a specific team."""
    if not check_api_keys():
        return
    
    scraper = WorldCup2026Scraper()
    
    # First get teams to find the team ID
    teams_df = scraper.scrape_world_cup_teams()
    
    team_row = teams_df[teams_df["team_name"].str.contains(args.team, case=False)]
    
    if team_row.empty:
        print(f"Team '{args.team}' not found")
        print("Available teams:")
        print(teams_df["team_name"].tolist())
        return
    
    team_id = int(team_row.iloc[0]["team_id"])
    team_name = team_row.iloc[0]["team_name"]
    
    squad_df = scraper.scrape_team_squad(team_id, team_name)
    print(f"\nSquad for {team_name}:")
    print(squad_df.to_string())


def cmd_stats(args):
    """Generate statistics summary."""
    import pandas as pd
    
    # Load existing data
    stats_df = load_all_player_statistics()
    
    if stats_df.empty:
        print("No statistics data found. Run 'scrape' first.")
        return
    
    # Calculate aggregates
    aggregated = calculate_player_aggregates(stats_df)
    
    print("\nTop 20 Players by Goals:")
    print(aggregated.nlargest(20, "goals_sum")[
        ["player_name", "nationality", "goals_sum", "assists_sum", "rating_mean"]
    ].to_string())
    
    print("\nTop 20 Players by Rating:")
    print(aggregated.nlargest(20, "rating_mean")[
        ["player_name", "nationality", "rating_mean", "goals_sum", "appearances_sum"]
    ].to_string())


def cmd_status(args):
    """Check API key status and rate limits."""
    if not check_api_keys():
        return
    
    from api_client import api_client
    
    print("\n" + "=" * 60)
    print("API KEY STATUS")
    print("=" * 60)
    
    # Check each key's status via API
    statuses = api_client.get_all_keys_status()
    
    for status_info in statuses:
        key_idx = status_info.get("key_index")
        key_preview = status_info.get("key_preview")
        api_status = status_info.get("status", {})
        
        # Handle different response formats
        if isinstance(api_status, list) and len(api_status) > 0:
            api_status = api_status[0]
        elif isinstance(api_status, list):
            api_status = {}
            
        account = api_status.get("account", {})
        subscription = api_status.get("subscription", {})
        requests_info = api_status.get("requests", {})
        
        print(f"\n🔑 Key #{key_idx}: {key_preview}")
        print(f"   Account: {account.get('firstname', 'N/A')} {account.get('lastname', '')}")
        print(f"   Subscription: {subscription.get('plan', 'N/A')}")
        print(f"   Daily Requests: {requests_info.get('current', 'N/A')}/{requests_info.get('limit_day', 'N/A')}")
    
    # Show current rate limiter status
    print("\n" + "=" * 60)
    print("RATE LIMITER STATUS (Current Minute)")
    print("=" * 60)
    
    limiter_status = api_client.get_key_manager_status()
    for key_name, key_info in limiter_status.items():
        print(f"\n{key_name}:")
        print(f"   Requests used: {key_info['requests_used']}/{REQUESTS_PER_KEY_PER_MINUTE}")
        print(f"   Remaining: {key_info['requests_remaining']}")


def cmd_test(args):
    """Run test scrape with limited data."""
    if args.mock:
        print("\nRunning with mock data...")
        run_mock_test()
        return
    
    if not check_api_keys():
        return
    
    max_teams = args.max_teams or 3
    print(f"\nRunning test scrape with {max_teams} teams...")
    
    scraper = WorldCup2026Scraper()
    scraper.run_full_scrape(max_teams=max_teams)


def run_mock_test():
    """Run with mock data for testing without API."""
    import pandas as pd
    from config import OUTPUT_DIR, TEAMS_OUTPUT_DIR, PLAYERS_OUTPUT_DIR, STATISTICS_OUTPUT_DIR
    
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEAMS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLAYERS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATISTICS_OUTPUT_DIR, exist_ok=True)
    
    # Mock team data
    teams_data = [
        {"team_id": 1, "team_name": "Argentina", "country": "Argentina", "code": "ARG", "is_host": False},
        {"team_id": 2, "team_name": "Brazil", "country": "Brazil", "code": "BRA", "is_host": False},
        {"team_id": 3, "team_name": "France", "country": "France", "code": "FRA", "is_host": False},
        {"team_id": 4, "team_name": "USA", "country": "USA", "code": "USA", "is_host": True},
        {"team_id": 5, "team_name": "Germany", "country": "Germany", "code": "GER", "is_host": False},
    ]
    
    teams_df = pd.DataFrame(teams_data)
    teams_df.to_csv(f"{TEAMS_OUTPUT_DIR}/world_cup_2026_teams.csv", index=False)
    
    # Mock player data
    players_data = [
        {"player_id": 101, "player_name": "L. Messi", "team_id": 1, "team_name": "Argentina", "age": 38, "position": "Attacker"},
        {"player_id": 102, "player_name": "J. Alvarez", "team_id": 1, "team_name": "Argentina", "age": 26, "position": "Attacker"},
        {"player_id": 201, "player_name": "Vinicius Jr", "team_id": 2, "team_name": "Brazil", "age": 25, "position": "Attacker"},
        {"player_id": 202, "player_name": "Rodrygo", "team_id": 2, "team_name": "Brazil", "age": 25, "position": "Attacker"},
        {"player_id": 301, "player_name": "K. Mbappe", "team_id": 3, "team_name": "France", "age": 27, "position": "Attacker"},
    ]
    
    players_df = pd.DataFrame(players_data)
    players_df.to_csv(f"{PLAYERS_OUTPUT_DIR}/all_squads.csv", index=False)
    
    # Mock statistics
    stats_data = [
        {"player_id": 101, "player_name": "L. Messi", "season": 2024, "league_name": "MLS", "appearances": 30, "goals": 20, "assists": 15, "rating": 8.5},
        {"player_id": 101, "player_name": "L. Messi", "season": 2025, "league_name": "MLS", "appearances": 25, "goals": 15, "assists": 12, "rating": 8.3},
        {"player_id": 102, "player_name": "J. Alvarez", "season": 2024, "league_name": "Premier League", "appearances": 35, "goals": 18, "assists": 8, "rating": 7.8},
        {"player_id": 201, "player_name": "Vinicius Jr", "season": 2024, "league_name": "La Liga", "appearances": 33, "goals": 22, "assists": 10, "rating": 8.4},
        {"player_id": 301, "player_name": "K. Mbappe", "season": 2024, "league_name": "La Liga", "appearances": 32, "goals": 28, "assists": 12, "rating": 8.6},
    ]
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(f"{STATISTICS_OUTPUT_DIR}/all_player_statistics.csv", index=False)
    
    print("Mock data created successfully!")
    print(f"\nTeams: {len(teams_df)}")
    print(teams_df.to_string())
    print(f"\nPlayers: {len(players_df)}")
    print(players_df.to_string())
    print(f"\nStatistics: {len(stats_df)}")
    print(stats_df.to_string())


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Football World Cup 2026 Data Collection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Run full data scraping")
    scrape_parser.add_argument(
        "--max-teams", type=int, default=None,
        help="Maximum number of teams to scrape (for testing)"
    )
    
    # Teams command
    teams_parser = subparsers.add_parser("teams", help="Scrape team information only")
    
    # Players command
    players_parser = subparsers.add_parser("players", help="Scrape players for a team")
    players_parser.add_argument(
        "--team", type=str, required=True,
        help="Team name to scrape players for"
    )
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Generate statistics summary")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run test with limited data")
    test_parser.add_argument(
        "--max-teams", type=int, default=3,
        help="Maximum teams for test (default: 3)"
    )
    test_parser.add_argument(
        "--mock", action="store_true",
        help="Use mock data instead of API"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check API key status and rate limits")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n\nQuick Start:")
        print("  1. Get API key from https://www.api-football.com/")
        print("  2. Create .env file with:")
        print("     API_FOOTBALL_KEY_1=your_first_key")
        print("     API_FOOTBALL_KEY_2=your_second_key  # (optional)")
        print("  3. Run: python main.py scrape")
        print("\n  Or test with mock data: python main.py test --mock")
        print("  Check status: python main.py status")
        return
    
    # Execute command
    commands = {
        "scrape": cmd_scrape,
        "teams": cmd_teams,
        "players": cmd_players,
        "stats": cmd_stats,
        "status": cmd_status,
        "test": cmd_test,
    }
    
    if args.command in commands:
        commands[args.command](args)


if __name__ == "__main__":
    main()

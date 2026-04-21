"""
Main scraper module for collecting World Cup 2026 football data.
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from api_client import api_client
from config import (
    WORLD_CUP_LEAGUE_ID,
    WORLD_CUP_REFERENCE_SEASON,
    WORLD_CUP_QUALIFICATIONS,
    MAJOR_LEAGUES,
    INTERNATIONAL_COMPETITIONS,
    SEASONS_TO_SCRAPE,
    OUTPUT_DIR,
    TEAMS_OUTPUT_DIR,
    PLAYERS_OUTPUT_DIR,
    STATISTICS_OUTPUT_DIR,
    PLAYER_STATS_FIELDS,
)


class WorldCup2026Scraper:
    """Scraper for World Cup 2026 data collection."""

    def __init__(self):
        self.api = api_client
        self._ensure_output_dirs()

    def _ensure_output_dirs(self):
        """Create output directories if they don't exist."""
        for dir_path in [OUTPUT_DIR, TEAMS_OUTPUT_DIR, PLAYERS_OUTPUT_DIR, STATISTICS_OUTPUT_DIR]:
            os.makedirs(dir_path, exist_ok=True)

    def fetch_world_cup_teams_from_api(self, season: int = None) -> pd.DataFrame:
        """
        Fetch teams from World Cup using the API.
        
        For WC 2026 (not yet started), we use 2022 teams as a base reference,
        then supplement with qualification data.
        
        Args:
            season: World Cup season (default: 2022 reference)
            
        Returns:
            DataFrame with team information
        """
        if season is None:
            season = WORLD_CUP_REFERENCE_SEASON
        
        print(f"\n📋 Fetching teams from World Cup {season}...")
        
        teams_data = []
        
        # Get teams from World Cup
        teams = self.api.get_teams(WORLD_CUP_LEAGUE_ID, season)
        
        for team_info in teams:
            team = team_info.get("team", {})
            venue = team_info.get("venue", {})
            
            teams_data.append({
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "country": team.get("country"),
                "code": team.get("code"),
                "founded": team.get("founded"),
                "national": team.get("national"),
                "logo": team.get("logo"),
                "venue_name": venue.get("name") if venue else None,
                "venue_city": venue.get("city") if venue else None,
                "venue_capacity": venue.get("capacity") if venue else None,
            })
        
        df = pd.DataFrame(teams_data)
        print(f"  Found {len(df)} teams from World Cup {season}")
        
        return df

    def fetch_teams_from_qualifications(self, season: int = 2024) -> pd.DataFrame:
        """
        Fetch additional teams from World Cup qualification competitions.
        This captures teams that qualified for 2026 but weren't in 2022.
        
        Args:
            season: Qualification season to check
            
        Returns:
            DataFrame with additional qualified teams
        """
        print(f"\n📋 Fetching teams from WC qualifications (season {season})...")
        
        all_teams = {}
        
        for qual_name, qual_id in WORLD_CUP_QUALIFICATIONS.items():
            try:
                teams = self.api.get_teams(qual_id, season)
                print(f"  {qual_name}: {len(teams)} teams")
                
                for team_info in teams:
                    team = team_info.get("team", {})
                    team_id = team.get("id")
                    
                    # Only add national teams
                    if team.get("national") and team_id not in all_teams:
                        venue = team_info.get("venue", {})
                        all_teams[team_id] = {
                            "team_id": team_id,
                            "team_name": team.get("name"),
                            "country": team.get("country"),
                            "code": team.get("code"),
                            "founded": team.get("founded"),
                            "national": team.get("national"),
                            "logo": team.get("logo"),
                            "venue_name": venue.get("name") if venue else None,
                            "venue_city": venue.get("city") if venue else None,
                            "venue_capacity": venue.get("capacity") if venue else None,
                            "qualification": qual_name,
                        }
            except Exception as e:
                print(f"  ⚠️ Error fetching {qual_name}: {e}")
        
        df = pd.DataFrame(list(all_teams.values()))
        print(f"  Total unique national teams from qualifications: {len(df)}")
        
        return df

    def scrape_world_cup_teams(self, include_qualifications: bool = True) -> pd.DataFrame:
        """
        Scrape teams for World Cup 2026.
        
        Combines teams from:
        1. World Cup 2022 (as base reference)
        2. World Cup qualifications (for newly qualified teams)
        
        Args:
            include_qualifications: Whether to include qualification teams
            
        Returns:
            DataFrame with all team information
        """
        print("\n" + "=" * 60)
        print("🏆 Scraping World Cup 2026 Teams")
        print("=" * 60)
        
        # Get World Cup 2022 teams as base
        wc_teams_df = self.fetch_world_cup_teams_from_api(WORLD_CUP_REFERENCE_SEASON)
        
        if include_qualifications:
            # Get additional teams from qualifications
            qual_teams_df = self.fetch_teams_from_qualifications(2024)
            
            # Merge, avoiding duplicates
            if not qual_teams_df.empty:
                existing_ids = set(wc_teams_df["team_id"].tolist())
                new_teams = qual_teams_df[~qual_teams_df["team_id"].isin(existing_ids)]
                
                if not new_teams.empty:
                    print(f"  Adding {len(new_teams)} new teams from qualifications")
                    wc_teams_df = pd.concat([wc_teams_df, new_teams], ignore_index=True)
        
        # Mark host nations
        host_names = ["USA", "Canada", "Mexico"]
        wc_teams_df["is_host"] = wc_teams_df["team_name"].isin(host_names)
        
        # Save to CSV
        output_path = f"{TEAMS_OUTPUT_DIR}/world_cup_2026_teams.csv"
        wc_teams_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(wc_teams_df)} teams to {output_path}")
        
        return wc_teams_df

    def scrape_team_squad(self, team_id: int, team_name: str) -> pd.DataFrame:
        """
        Scrape current squad for a team using /players/squads endpoint.
        
        Args:
            team_id: Team ID
            team_name: Team name for file naming
            
        Returns:
            DataFrame with player information
        """
        print(f"\n👥 Fetching squad for {team_name} (ID: {team_id})...")
        
        squad_data = self.api.get_squad(team_id)
        
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

    def scrape_player_statistics(
        self,
        player_id: int,
        player_name: str,
        seasons: List[int] = None,
    ) -> pd.DataFrame:
        """
        Scrape comprehensive historical statistics for a player.
        
        Args:
            player_id: Player ID
            player_name: Player name for logging
            seasons: List of seasons to scrape (defaults to SEASONS_TO_SCRAPE)
            
        Returns:
            DataFrame with player statistics
        """
        if seasons is None:
            seasons = SEASONS_TO_SCRAPE
        
        all_stats = []
        
        for season in seasons:
            try:
                stats_response = self.api.get_player_statistics(player_id, season)
                
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
                        
                        all_stats.append({
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
                        
            except Exception as e:
                print(f"  ⚠️ Error fetching stats for {player_name} season {season}: {e}")
        
        return pd.DataFrame(all_stats)

    def scrape_all_team_players_statistics(
        self,
        team_id: int,
        team_name: str,
        squad_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scrape statistics for all players in a team's squad.
        
        Args:
            team_id: Team ID
            team_name: Team name
            squad_df: DataFrame with squad player information
            
        Returns:
            DataFrame with all player statistics
        """
        print(f"\n{'=' * 60}")
        print(f"Scraping player statistics for {team_name}")
        print(f"{'=' * 60}")
        
        all_player_stats = []
        
        for _, player in tqdm(squad_df.iterrows(), total=len(squad_df), desc=f"Players from {team_name}"):
            player_id = player["player_id"]
            player_name = player["player_name"]
            
            try:
                stats_df = self.scrape_player_statistics(player_id, player_name)
                if not stats_df.empty:
                    stats_df["national_team_id"] = team_id
                    stats_df["national_team_name"] = team_name
                    all_player_stats.append(stats_df)
            except Exception as e:
                print(f"  Error processing {player_name}: {e}")
        
        if all_player_stats:
            combined_df = pd.concat(all_player_stats, ignore_index=True)
            
            # Save to CSV
            safe_name = team_name.replace(" ", "_").lower()
            output_path = f"{STATISTICS_OUTPUT_DIR}/{safe_name}_player_statistics.csv"
            combined_df.to_csv(output_path, index=False)
            print(f"\nSaved {len(combined_df)} statistics records to {output_path}")
            
            return combined_df
        
        return pd.DataFrame()

    def scrape_international_competition_stats(
        self,
        competition_id: int,
        competition_name: str,
        team_ids: List[int],
        seasons: List[int] = None,
    ) -> pd.DataFrame:
        """
        Scrape statistics from international competitions.
        
        Args:
            competition_id: Competition/League ID
            competition_name: Competition name
            team_ids: List of team IDs to filter
            seasons: Seasons to scrape
            
        Returns:
            DataFrame with competition statistics
        """
        if seasons is None:
            seasons = SEASONS_TO_SCRAPE
        
        print(f"\nScraping {competition_name} statistics...")
        
        all_fixtures = []
        
        for season in seasons:
            try:
                fixtures = self.api.get_fixtures(competition_id, season)
                
                for fixture in fixtures:
                    fixture_info = fixture.get("fixture", {})
                    league = fixture.get("league", {})
                    teams = fixture.get("teams", {})
                    goals = fixture.get("goals", {})
                    score = fixture.get("score", {})
                    
                    home_team = teams.get("home", {})
                    away_team = teams.get("away", {})
                    
                    # Filter fixtures involving our teams
                    if home_team.get("id") in team_ids or away_team.get("id") in team_ids:
                        all_fixtures.append({
                            "fixture_id": fixture_info.get("id"),
                            "date": fixture_info.get("date"),
                            "venue": fixture_info.get("venue", {}).get("name"),
                            "competition_id": competition_id,
                            "competition_name": competition_name,
                            "season": season,
                            "round": league.get("round"),
                            "home_team_id": home_team.get("id"),
                            "home_team_name": home_team.get("name"),
                            "home_goals": goals.get("home"),
                            "away_team_id": away_team.get("id"),
                            "away_team_name": away_team.get("name"),
                            "away_goals": goals.get("away"),
                            "home_winner": home_team.get("winner"),
                            "away_winner": away_team.get("winner"),
                            "halftime_home": score.get("halftime", {}).get("home"),
                            "halftime_away": score.get("halftime", {}).get("away"),
                            "fulltime_home": score.get("fulltime", {}).get("home"),
                            "fulltime_away": score.get("fulltime", {}).get("away"),
                            "extratime_home": score.get("extratime", {}).get("home"),
                            "extratime_away": score.get("extratime", {}).get("away"),
                            "penalty_home": score.get("penalty", {}).get("home"),
                            "penalty_away": score.get("penalty", {}).get("away"),
                        })
                        
            except Exception as e:
                print(f"  Error fetching {competition_name} season {season}: {e}")
        
        df = pd.DataFrame(all_fixtures)
        
        if not df.empty:
            safe_name = competition_name.replace(" ", "_").lower()
            output_path = f"{STATISTICS_OUTPUT_DIR}/{safe_name}_fixtures.csv"
            df.to_csv(output_path, index=False)
            print(f"  Saved {len(df)} fixtures to {output_path}")
        
        return df

    def run_full_scrape(self, max_teams: int = None):
        """
        Run the complete data scraping process.
        
        Args:
            max_teams: Maximum number of teams to scrape (for testing)
        """
        print("\n" + "=" * 70)
        print("WORLD CUP 2026 DATA SCRAPER")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Check API status
        print("\nChecking API status...")
        try:
            status = self.api.get_account_status()
            account = status.get("response", {}).get("account", {})
            requests_info = status.get("response", {}).get("requests", {})
            print(f"  Account: {account.get('firstname', 'N/A')} {account.get('lastname', 'N/A')}")
            print(f"  Remaining requests: {requests_info.get('current', 'N/A')}/{requests_info.get('limit_day', 'N/A')}")
        except Exception as e:
            print(f"  Warning: Could not check API status: {e}")
        
        # Step 1: Scrape World Cup 2026 teams
        teams_df = self.scrape_world_cup_teams()
        
        if teams_df.empty:
            print("No teams found. Exiting.")
            return
        
        # Limit teams if specified (for testing)
        if max_teams:
            teams_df = teams_df.head(max_teams)
            print(f"\nLimiting to {max_teams} teams for testing")
        
        # Step 2: For each team, scrape squad and player statistics
        all_squads = []
        all_player_statistics = []
        team_ids = []
        
        for _, team in teams_df.iterrows():
            team_id = team["team_id"]
            team_name = team["team_name"]
            
            if pd.isna(team_id):
                continue
                
            team_id = int(team_id)
            team_ids.append(team_id)
            
            # Get squad
            squad_df = self.scrape_team_squad(team_id, team_name)
            if not squad_df.empty:
                all_squads.append(squad_df)
                
                # Get player statistics
                player_stats_df = self.scrape_all_team_players_statistics(
                    team_id, team_name, squad_df
                )
                if not player_stats_df.empty:
                    all_player_statistics.append(player_stats_df)
        
        # Save combined squad data
        if all_squads:
            combined_squads = pd.concat(all_squads, ignore_index=True)
            combined_squads.to_csv(f"{PLAYERS_OUTPUT_DIR}/all_squads.csv", index=False)
            print(f"\nSaved combined squads: {len(combined_squads)} players")
        
        # Save combined player statistics
        if all_player_statistics:
            combined_stats = pd.concat(all_player_statistics, ignore_index=True)
            combined_stats.to_csv(f"{STATISTICS_OUTPUT_DIR}/all_player_statistics.csv", index=False)
            print(f"Saved combined statistics: {len(combined_stats)} records")
        
        # Step 3: Scrape international competition fixtures
        print("\n" + "=" * 60)
        print("Scraping International Competition Data")
        print("=" * 60)
        
        for comp_name, comp_id in INTERNATIONAL_COMPETITIONS.items():
            try:
                self.scrape_international_competition_stats(
                    comp_id, comp_name, team_ids
                )
            except Exception as e:
                print(f"  Error scraping {comp_name}: {e}")
        
        print("\n" + "=" * 70)
        print("SCRAPING COMPLETE")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"\nData saved to: {os.path.abspath(OUTPUT_DIR)}")


def main():
    """Main entry point."""
    scraper = WorldCup2026Scraper()
    
    # Run full scrape (use max_teams for testing)
    # scraper.run_full_scrape(max_teams=3)  # For testing
    scraper.run_full_scrape()


if __name__ == "__main__":
    main()

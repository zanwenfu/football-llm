"""
Configuration settings for the Football Data Scraper.
"""
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_api_keys() -> List[str]:
    """
    Get all API keys from environment variables.
    Supports single key (API_FOOTBALL_KEY) or multiple keys (API_FOOTBALL_KEY_1, API_FOOTBALL_KEY_2, etc.)
    """
    keys = []
    
    # Check for single key first (most common case)
    key = os.getenv("API_FOOTBALL_KEY", "")
    if key and key != "your_api_key_here":
        keys.append(key)
    
    # Also check for numbered keys (API_FOOTBALL_KEY_1, API_FOOTBALL_KEY_2, etc.)
    for i in range(1, 10):
        key = os.getenv(f"API_FOOTBALL_KEY_{i}", "")
        if key and key != "your_api_key_here" and key not in keys:
            keys.append(key)
    
    return keys


# API-Football Configuration
API_FOOTBALL_KEYS = get_api_keys()
API_FOOTBALL_HOST = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")
API_FOOTBALL_BASE_URL = f"https://{API_FOOTBALL_HOST}"

# Rate limiting - Free tier is 10 requests per minute per key
# IMPORTANT: Adjust this based on your subscription plan
REQUESTS_PER_KEY_PER_MINUTE = int(os.getenv("REQUESTS_PER_KEY_PER_MINUTE", 10))

# World Cup Configuration
WORLD_CUP_LEAGUE_ID = 1  # FIFA World Cup league ID in API-Football
WORLD_CUP_2026_YEAR = 2026

# Reference season to get national team IDs (most recent World Cup)
WORLD_CUP_REFERENCE_SEASON = 2022

# World Cup qualifications league IDs
WORLD_CUP_QUALIFICATIONS = {
    "WC Qualification - Europe": 32,
    "WC Qualification - South America": 29,
    "WC Qualification - CONCACAF": 31,
    "WC Qualification - Africa": 33,
    "WC Qualification - Asia": 30,
    "WC Qualification - Oceania": 34,
}

# Major Domestic Leagues IDs (API-Football) - comprehensive list
MAJOR_LEAGUES = {
    # Top 5 European Leagues
    "Premier League": 39,
    "La Liga": 140,
    "Bundesliga": 78,
    "Serie A": 135,
    "Ligue 1": 61,
    # Other European Leagues
    "Primeira Liga": 94,
    "Eredivisie": 88,
    "Belgian Pro League": 144,
    "Swiss Super League": 207,
    "Austrian Bundesliga": 218,
    "Scottish Premiership": 179,
    "Turkish Super Lig": 203,
    "Russian Premier League": 235,
    "Ukrainian Premier League": 333,
    "Croatian First League": 210,
    "Serbian Super Liga": 286,
    "Danish Superliga": 119,
    "Polish Ekstraklasa": 106,
    "Czech First League": 345,
    "Greek Super League": 197,
    "Norwegian Eliteserien": 103,
    "Swedish Allsvenskan": 113,
    # Americas
    "Argentine Primera Division": 128,
    "Brazilian Serie A": 71,
    "MLS": 253,
    "Liga MX": 262,
    "Colombian Primera A": 239,
    "Chilean Primera Division": 265,
    "Uruguayan Primera Division": 268,
    "Peruvian Primera Division": 281,
    "Ecuadorian Serie A": 242,
    "Paraguayan Primera Division": 279,
    "Venezuelan Primera Division": 299,
    # Asia & Middle East
    "Saudi Pro League": 307,
    "J1 League": 98,
    "K League 1": 292,
    "Chinese Super League": 169,
    "UAE Pro League": 305,
    "Qatar Stars League": 306,
    "Iranian Pro League": 290,
    # Africa
    "Egyptian Premier League": 233,
    "Moroccan Botola Pro": 200,
    "South African Premier Division": 288,
    "Tunisian Ligue 1": 202,
    "Algerian Ligue 1": 352,
    "Nigerian NPFL": 357,
    "Ghanaian Premier League": 372,
    # Oceania
    "A-League": 188,
}

# International Competitions for historical data
INTERNATIONAL_COMPETITIONS = {
    "World Cup": 1,
    "WC Qualification - Europe": 32,
    "WC Qualification - South America": 29,
    "WC Qualification - CONCACAF": 31,
    "WC Qualification - Africa": 33,
    "WC Qualification - Asia": 30,
    "UEFA Euro": 4,
    "Copa America": 9,
    "CONCACAF Gold Cup": 22,
    "Africa Cup of Nations": 6,
    "AFC Asian Cup": 7,
    "UEFA Nations League": 5,
    "UEFA Champions League": 2,
    "UEFA Europa League": 3,
    "Copa Libertadores": 13,
}

# Historical seasons to scrape
# Extended to cover 22 years of data (2004-2025) for comprehensive historical analysis
SEASONS_TO_SCRAPE = [
    2025, 2024, 2023, 2022,  # Recent seasons
    2021, 2020, 2019, 2018, 2017, 2016, 2015,  # Mid-range
    2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004  # Historical
]

# Output directories
OUTPUT_DIR = "data"
TEAMS_OUTPUT_DIR = f"{OUTPUT_DIR}/teams"
PLAYERS_OUTPUT_DIR = f"{OUTPUT_DIR}/players"
STATISTICS_OUTPUT_DIR = f"{OUTPUT_DIR}/statistics"

# Player statistics fields - comprehensive list matching API response
PLAYER_STATS_FIELDS = [
    # Player info
    "player_id",
    "player_name",
    "firstname",
    "lastname",
    "nationality",
    "birth_date",
    "birth_place",
    "birth_country",
    "age",
    "height",
    "weight",
    "injured",
    "photo",
    # Team & League context
    "team_id",
    "team_name",
    "league_id",
    "league_name",
    "league_country",
    "season",
    # Games
    "position",
    "appearances",
    "lineups",
    "minutes",
    "rating",
    "captain",
    # Substitutes
    "substitutes_in",
    "substitutes_out",
    "substitutes_bench",
    # Shooting
    "shots_total",
    "shots_on_target",
    # Goals
    "goals",
    "goals_conceded",
    "assists",
    "saves",
    # Passing
    "passes_total",
    "passes_key",
    "passes_accuracy",
    # Defensive
    "tackles_total",
    "tackles_blocks",
    "tackles_interceptions",
    # Duels
    "duels_total",
    "duels_won",
    # Dribbles
    "dribbles_attempts",
    "dribbles_success",
    "dribbles_past",
    # Fouls
    "fouls_drawn",
    "fouls_committed",
    # Cards
    "cards_yellow",
    "cards_yellowred",
    "cards_red",
    # Penalties
    "penalty_won",
    "penalty_committed",
    "penalty_scored",
    "penalty_missed",
    "penalty_saved",
]

# Squad player fields (from /players/squads endpoint)
SQUAD_PLAYER_FIELDS = [
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "age",
    "number",
    "position",
    "photo",
]

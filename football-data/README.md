# Football Data Scraper (API-Football)

> **Relationship to the main project.** This is a **standalone scraper sub-project** that produces the raw CSVs used by the Football-LLM training pipeline. The parent repo's [`data/raw/`](../data/) directory contains pre-scraped 2010–2022 World Cup data; you only need to run this scraper if you want to **refresh** the data, extend to **new tournaments** (Euro 2024, Copa 2024, WC 2026), or **add new leagues**.
>
> For just reproducing the paper's numbers, you do not need to run this — the committed raw CSVs suffice. See [../README.md](../README.md#quickstart).

A Python tool for collecting and organizing football player data for World Cup prediction tasks.

## Overview

This tool scrapes data from the **API-Football** service to collect:
- Teams qualified for World Cup (fetched dynamically from API)
- Player rosters for each national team
- Historical player statistics from domestic leagues
- International competition match data

## Script index

There are many specialized scrapers in this directory because API-Football data is collected in pieces (teams → squads → per-player season stats → per-match events). Most users only need `main.py` and `check_api.py`.

| Script | Purpose |
|:---|:---|
| `main.py` | **Entry point.** Top-level CLI with `status` / `scrape` / `teams` / `players` / `stats` / `test` subcommands |
| `check_api.py` | Verify API keys, quota, and rate-limit state |
| `api_client.py` | Multi-key API client with automatic rotation + retry |
| `config.py` | Configuration constants — edit to add new leagues/tournaments |
| `utils.py` | Pandas helpers — aggregation, team-strength analysis |
| `scraper.py` | Top-level orchestrator for the full pipeline |
| `scrape_all_teams.py` | Production: scrape all teams qualified for a tournament |
| `scrape_past_wc_teams.py` | Scrape player stats for teams in past World Cups (2010–2022) |
| `scrape_wc_player_stats.py` | Complete career statistics for every player who appeared in a WC |
| `scrape_historical.py` | Scrape additional historical seasons and merge with existing data |
| `scrape_world_cup_history.py` | Historical FIFA World Cup match-level data |
| `scrape_player_match_stats.py` | Per-match player statistics from FIFA World Cup |
| `scrape_statsbomb_team_stats.py` | Team-level match statistics for FIFA WC 2018 & 2022 |
| `enhance_wc_data.py` | Team-level stats from fixtures/statistics endpoint |
| `repair_statistics.py` | Repair script for player statistics completeness |
| `align_2026_data.py` | Align WC 2026 feature set with the historical schema |
| `analyze_ghana.py` / `scrape_ghana.py` | Team-specific scripts (kept as examples of the single-team workflow) |

If you're adding a new script, prefer extending `main.py` with a subcommand over adding another top-level file.

## Features

- **Multi-API Key Support**: Use multiple API keys with automatic rotation to maximize throughput
- **Intelligent Rate Limiting**: Respects rate limits per key (10 req/min on free tier)
- **Automatic Key Failover**: Detects suspended/invalid keys and disables them automatically
- **Dynamic Team Fetching**: Gets teams from World Cup 2022 as reference for 2026
- **Comprehensive Statistics**: Collects 50+ statistics fields per player per season

## Data Sources

The primary data source is [API-Football](https://www.api-football.com/), one of the most comprehensive football APIs available. It provides:
- Coverage of 900+ leagues and competitions
- Real-time statistics and historical data
- Player career statistics across multiple seasons
- Match-level statistics and lineups

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

1. Sign up at [API-Football](https://www.api-football.com/) or via [RapidAPI](https://rapidapi.com/api-sports/api/api-football)
2. Free tier: **100 requests/day, 10 requests/minute** per key
3. Consider creating multiple free accounts for higher throughput
4. Paid plans available for more requests

### 3. Configure Environment

Create a `.env` file in the project root:

```env
# Multiple API keys for rate limit rotation (recommended)
API_FOOTBALL_KEY_1=your_first_api_key
API_FOOTBALL_KEY_2=your_second_api_key

# Or single key (less throughput)
# API_FOOTBALL_KEY=your_api_key_here

API_FOOTBALL_HOST=v3.football.api-sports.io

# Optional: Override default rate limit (10 req/min for free tier)
# REQUESTS_PER_KEY_PER_MINUTE=10
```

## Usage

### Basic Commands

```bash
# Check API key status and rate limits
python main.py status

# Run full data scraping
python main.py scrape

# Scrape only team information
python main.py teams

# Scrape players for a specific team
python main.py players --team "Argentina"

# Generate statistics summary
python main.py stats

# Test with mock data (no API key needed)
python main.py test --mock

# Test with limited API calls
python main.py test --max-teams 3
```

### Full Scrape

The full scrape process:
1. Fetches teams from World Cup 2022 (32 teams) via API
2. Optionally fetches additional teams from qualification competitions
3. For each team, gets the current squad roster
4. For each player, fetches historical statistics (last 5 seasons)
5. Collects international competition match data
6. Saves everything to CSV files

```bash
python main.py scrape
```

**Note:** A full scrape may require thousands of API calls. Consider the following:
- Free tier: 100 requests/day - use `--max-teams` for testing
- Built-in rate limiting prevents API throttling
- Progress is saved incrementally

## Output Structure

```
data/
├── teams/
│   └── world_cup_2026_teams.csv       # All qualified teams
├── players/
│   ├── argentina_squad.csv            # Per-team squad files
│   ├── brazil_squad.csv
│   ├── ...
│   └── all_squads.csv                 # Combined squad data
└── statistics/
    ├── argentina_player_statistics.csv # Per-team player stats
    ├── brazil_player_statistics.csv
    ├── ...
    ├── all_player_statistics.csv      # Combined statistics
    └── world_cup_fixtures.csv         # Match data
```

## Data Fields

### Team Data
- `team_id`, `team_name`, `country`, `code`
- `founded`, `logo`
- `venue_name`, `venue_city`, `venue_capacity`
- `is_host` (for USA, Mexico, Canada)

### Player Data
- `player_id`, `player_name`
- `team_id`, `team_name`, `nationality`
- `age`, `position`, `number`

### Player Statistics
- **Basic**: `appearances`, `lineups`, `minutes`, `rating`
- **Attacking**: `goals`, `assists`, `shots_total`, `shots_on_target`
- **Passing**: `passes_total`, `passes_key`, `passes_accuracy`
- **Defensive**: `tackles_total`, `tackles_blocks`, `tackles_interceptions`
- **Physical**: `duels_total`, `duels_won`, `dribbles_attempts`, `dribbles_success`
- **Discipline**: `cards_yellow`, `cards_red`, `fouls_drawn`, `fouls_committed`
- **Penalties**: `penalty_scored`, `penalty_missed`

## World Cup 2026 Details

- **Host Countries**: USA, Mexico, Canada (automatically qualified)
- **Total Teams**: 48 (expanded from 32)
- **Format**: 12 groups of 4 teams
- **Dates**: June 11 - July 19, 2026

### Qualification Slots
- UEFA (Europe): 16 teams
- CAF (Africa): 9 teams
- AFC (Asia): 8 teams
- CONMEBOL (South America): 6 teams
- CONCACAF (North/Central America): 6 teams (including 3 hosts)
- OFC (Oceania): 1 team
- Playoffs: 2 teams

## Data Processing

### Aggregating Statistics

```python
from utils import load_all_player_statistics, calculate_player_aggregates

# Load all statistics
stats_df = load_all_player_statistics()

# Calculate career aggregates
aggregated = calculate_player_aggregates(stats_df)

# Top scorers
print(aggregated.nlargest(20, "goals_sum"))
```

### Team Strength Analysis

```python
from utils import get_team_strength_analysis
import pandas as pd

squads_df = pd.read_csv("data/players/all_squads.csv")
stats_df = pd.read_csv("data/statistics/all_player_statistics.csv")

analysis = get_team_strength_analysis(squads_df, stats_df)
print(analysis.sort_values("strength_score", ascending=False))
```

## Rate Limiting & Best Practices

- Built-in rate limiting: 30 requests/minute (configurable)
- Automatic retry with exponential backoff
- Progress saved incrementally to prevent data loss
- Use `--max-teams` flag for testing

## Covered Leagues

### Major Domestic Leagues
- Premier League (England)
- La Liga (Spain)
- Bundesliga (Germany)
- Serie A (Italy)
- Ligue 1 (France)
- And 20+ more leagues

### International Competitions
- FIFA World Cup
- UEFA Euro
- Copa America
- CONCACAF Gold Cup
- Africa Cup of Nations
- AFC Asian Cup
- UEFA Champions League
- Copa Libertadores

## Error Handling

The scraper handles common issues:
- Rate limit exceeded: Automatic retry with backoff
- Network errors: 3 retry attempts
- Missing data: Graceful fallback with warnings
- API errors: Logged and skipped

## License

MIT License - Feel free to use for your World Cup prediction projects!

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues with the API, check:
- [API-Football Documentation](https://www.api-football.com/documentation-v3)
- [API Status Page](https://status-api-sports.io/)

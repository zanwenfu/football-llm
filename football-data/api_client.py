"""
API Client for API-Football.
Handles all HTTP requests with rate limiting, API key rotation, and error handling.
"""
import time
import threading
import requests
from typing import Optional, Dict, Any, List
from collections import deque
import backoff

from config import (
    API_FOOTBALL_KEYS,
    API_FOOTBALL_BASE_URL,
    REQUESTS_PER_KEY_PER_MINUTE,
)


class RateLimitedKeyManager:
    """
    Manages multiple API keys with per-key rate limiting.
    Rotates through keys to maximize throughput while respecting rate limits.
    Automatically disables suspended or invalid keys.
    """
    
    def __init__(self, api_keys: List[str], requests_per_minute: int = 10):
        if not api_keys:
            raise ValueError("At least one API key is required")
        
        self.api_keys = api_keys
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        
        # Track request timestamps for each key
        # key_index -> deque of timestamps
        self.request_timestamps: Dict[int, deque] = {
            i: deque() for i in range(len(api_keys))
        }
        
        # Track disabled keys (suspended, invalid, etc.)
        self.disabled_keys: set = set()
        
        self.current_key_index = 0
        self.lock = threading.Lock()
        
        print(f"Initialized API client with {len(api_keys)} key(s), "
              f"{requests_per_minute} requests/min per key "
              f"(total: {len(api_keys) * requests_per_minute}/min)")

    def _clean_old_timestamps(self, key_index: int) -> None:
        """Remove timestamps older than the rate limit window."""
        current_time = time.time()
        cutoff = current_time - self.window_seconds
        
        timestamps = self.request_timestamps[key_index]
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

    def _get_available_key(self) -> Optional[int]:
        """
        Find an available key that hasn't exceeded its rate limit and is not disabled.
        Returns key index or None if all keys are exhausted.
        """
        checked = 0
        while checked < len(self.api_keys):
            # Skip disabled keys
            if self.current_key_index in self.disabled_keys:
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                checked += 1
                continue
                
            self._clean_old_timestamps(self.current_key_index)
            
            if len(self.request_timestamps[self.current_key_index]) < self.requests_per_minute:
                return self.current_key_index
            
            # Try next key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            checked += 1
        
        return None

    def disable_key(self, key_index: int, reason: str = ""):
        """Disable a key that is no longer working (suspended, invalid, etc.)."""
        with self.lock:
            self.disabled_keys.add(key_index)
            key_preview = f"{self.api_keys[key_index][:8]}..."
            print(f"  ⛔ Disabled API key #{key_index + 1} ({key_preview}): {reason}")
            
            active_keys = len(self.api_keys) - len(self.disabled_keys)
            if active_keys == 0:
                print("  ⚠️ WARNING: All API keys are disabled!")
            else:
                print(f"  📊 Remaining active keys: {active_keys}")

    def _get_wait_time(self) -> float:
        """Calculate how long to wait until a key becomes available."""
        min_wait = float('inf')
        current_time = time.time()
        
        for key_index in range(len(self.api_keys)):
            # Skip disabled keys
            if key_index in self.disabled_keys:
                continue
                
            self._clean_old_timestamps(key_index)
            timestamps = self.request_timestamps[key_index]
            
            if len(timestamps) < self.requests_per_minute:
                return 0
            
            # Time until oldest request expires
            oldest = timestamps[0]
            wait_time = (oldest + self.window_seconds) - current_time
            min_wait = min(min_wait, wait_time)
        
        return max(0, min_wait) if min_wait != float('inf') else 0

    def get_key_with_rate_limit(self) -> tuple:
        """
        Get an API key that's available for use, waiting if necessary.
        Implements fair rotation across all available keys.
        
        Returns:
            tuple: (api_key, key_index) for the available key
        """
        with self.lock:
            # Check if all keys are disabled
            if len(self.disabled_keys) >= len(self.api_keys):
                raise RuntimeError("All API keys are disabled. Cannot make requests.")
            
            while True:
                key_index = self._get_available_key()
                
                if key_index is not None:
                    # Record this request
                    self.request_timestamps[key_index].append(time.time())
                    # Move to next key for fair distribution
                    self.current_key_index = (key_index + 1) % len(self.api_keys)
                    return (self.api_keys[key_index], key_index)
                
                # All keys exhausted, need to wait
                wait_time = self._get_wait_time()
                if wait_time > 0:
                    print(f"  ⏳ Rate limit reached on all keys, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time + 0.1)  # Small buffer

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all API keys."""
        with self.lock:
            status = {}
            for i, key in enumerate(self.api_keys):
                self._clean_old_timestamps(i)
                remaining = self.requests_per_minute - len(self.request_timestamps[i])
                status[f"key_{i+1}"] = {
                    "key_preview": f"{key[:8]}...{key[-4:]}",
                    "requests_used": len(self.request_timestamps[i]),
                    "requests_remaining": remaining,
                }
            return status


class APIFootballClient:
    """Client for interacting with API-Football with multi-key support."""

    def __init__(self):
        self.base_url = API_FOOTBALL_BASE_URL
        self.key_manager = RateLimitedKeyManager(
            API_FOOTBALL_KEYS, 
            REQUESTS_PER_KEY_PER_MINUTE
        )
        self.session = requests.Session()
        self.total_requests = 0

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.Timeout),
        max_tries=3,
        max_time=60,
    )
    def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a rate-limited request to the API using rotating keys.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url}/{endpoint}"
        
        # Get an available API key (will wait if rate limited)
        api_key, key_index = self.key_manager.get_key_with_rate_limit()
        
        headers = {"x-apisports-key": api_key}
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            self.total_requests += 1
            
            # Check rate limit headers
            remaining = response.headers.get('X-RateLimit-Remaining', 'N/A')
            
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if data.get("errors"):
                errors = data["errors"]
                if errors:
                    error_str = str(errors)
                    
                    # Handle suspended account - disable this key
                    if "suspended" in error_str.lower() or "account is suspended" in error_str.lower():
                        self.key_manager.disable_key(key_index, "Account suspended")
                        # Retry with another key if available
                        if len(self.key_manager.disabled_keys) < len(self.key_manager.api_keys):
                            return self._make_request(endpoint, params)
                        return {"response": []}
                    
                    # Handle rate limit error - wait and retry
                    if "rateLimit" in error_str or "rate limit" in error_str.lower() or "Too many requests" in error_str:
                        print(f"  ⏳ Rate limit hit, waiting 60 seconds...")
                        time.sleep(60)
                        return self._make_request(endpoint, params)  # Retry after waiting
                        
                    print(f"  ⚠️ API Error: {errors}")
                    return {"response": []}
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"  ⚠️ HTTP 429 Too Many Requests - waiting...")
                time.sleep(10)
            elif e.response.status_code == 403:
                # Forbidden - likely suspended or invalid key
                self.key_manager.disable_key(key_index, "HTTP 403 Forbidden")
                if len(self.key_manager.disabled_keys) < len(self.key_manager.api_keys):
                    return self._make_request(endpoint, params)
            print(f"  ❌ HTTP Error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Request Error: {e}")
            raise

    def get_account_status(self, key_index: int = 0) -> Dict[str, Any]:
        """Get current API account status and remaining requests for a specific key."""
        if key_index >= len(API_FOOTBALL_KEYS):
            return {}
        
        headers = {"x-apisports-key": API_FOOTBALL_KEYS[key_index]}
        try:
            response = self.session.get(
                f"{self.base_url}/status", 
                headers=headers, 
                timeout=30
            )
            return response.json()
        except Exception as e:
            print(f"Error checking status: {e}")
            return {}

    def get_all_keys_status(self) -> List[Dict[str, Any]]:
        """Get status for all configured API keys."""
        statuses = []
        for i in range(len(API_FOOTBALL_KEYS)):
            status = self.get_account_status(i)
            if status:
                statuses.append({
                    "key_index": i + 1,
                    "key_preview": f"{API_FOOTBALL_KEYS[i][:8]}...",
                    "status": status.get("response", {})
                })
        return statuses

    def get_key_manager_status(self) -> Dict[str, Any]:
        """Get current rate limit status for all keys."""
        return self.key_manager.get_status()

    def get_countries(self) -> List[Dict[str, Any]]:
        """Get all available countries."""
        response = self._make_request("countries")
        return response.get("response", [])

    def get_leagues(
        self,
        country: Optional[str] = None,
        season: Optional[int] = None,
        league_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get leagues/competitions.
        
        Args:
            country: Filter by country name
            season: Filter by season year
            league_id: Filter by specific league ID
        """
        params = {}
        if country:
            params["country"] = country
        if season:
            params["season"] = season
        if league_id:
            params["id"] = league_id
            
        response = self._make_request("leagues", params)
        return response.get("response", [])

    def get_teams(
        self,
        league_id: int,
        season: int,
    ) -> List[Dict[str, Any]]:
        """
        Get teams for a specific league and season.
        
        Args:
            league_id: League/competition ID
            season: Season year
        """
        params = {
            "league": league_id,
            "season": season,
        }
        response = self._make_request("teams", params)
        return response.get("response", [])

    def get_team_by_id(self, team_id: int) -> Dict[str, Any]:
        """Get team information by ID."""
        params = {"id": team_id}
        response = self._make_request("teams", params)
        teams = response.get("response", [])
        return teams[0] if teams else {}

    def get_team_by_country(self, country: str) -> List[Dict[str, Any]]:
        """Get national team by country name."""
        params = {"country": country}
        response = self._make_request("teams", params)
        return response.get("response", [])

    def get_squad(self, team_id: int) -> List[Dict[str, Any]]:
        """
        Get current squad/players for a team.
        
        Args:
            team_id: Team ID
        """
        params = {"team": team_id}
        response = self._make_request("players/squads", params)
        return response.get("response", [])

    def get_player_info(self, player_id: int) -> Dict[str, Any]:
        """Get detailed player information."""
        params = {"id": player_id}
        response = self._make_request("players", params)
        players = response.get("response", [])
        return players[0] if players else {}

    def get_player_seasons(self, player_id: int) -> List[int]:
        """Get available seasons for a player."""
        params = {"player": player_id}
        response = self._make_request("players/seasons", params)
        return response.get("response", [])

    def get_player_statistics(
        self,
        player_id: int,
        season: int,
        league_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get player statistics for a specific season.
        
        Args:
            player_id: Player ID
            season: Season year
            league_id: Optional league ID filter
        """
        params = {
            "id": player_id,
            "season": season,
        }
        if league_id:
            params["league"] = league_id
            
        response = self._make_request("players", params)
        return response.get("response", [])

    def get_players_by_team_season(
        self,
        team_id: int,
        season: int,
        page: int = 1,
    ) -> Dict[str, Any]:
        """
        Get all players for a team in a specific season with pagination.
        
        Args:
            team_id: Team ID
            season: Season year
            page: Page number for pagination
        """
        params = {
            "team": team_id,
            "season": season,
            "page": page,
        }
        return self._make_request("players", params)

    def get_fixtures(
        self,
        league_id: int,
        season: int,
        team_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get fixtures/matches for a league and season.
        
        Args:
            league_id: League ID
            season: Season year
            team_id: Optional team ID filter
        """
        params = {
            "league": league_id,
            "season": season,
        }
        if team_id:
            params["team"] = team_id
            
        response = self._make_request("fixtures", params)
        return response.get("response", [])

    def get_fixture_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Get statistics for a specific fixture/match."""
        params = {"fixture": fixture_id}
        response = self._make_request("fixtures/statistics", params)
        return response.get("response", [])

    def get_fixture_players(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Get player statistics for a specific fixture/match."""
        params = {"fixture": fixture_id}
        response = self._make_request("fixtures/players", params)
        return response.get("response", [])

    def search_teams(self, name: str) -> List[Dict[str, Any]]:
        """Search for teams by name."""
        params = {"search": name}
        response = self._make_request("teams", params)
        return response.get("response", [])

    def search_players(self, name: str) -> List[Dict[str, Any]]:
        """Search for players by name."""
        params = {"search": name}
        response = self._make_request("players", params)
        return response.get("response", [])


# Singleton instance
api_client = APIFootballClient()

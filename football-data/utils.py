"""
Utility functions for data processing and analysis.
"""
import os
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


def load_all_player_statistics(data_dir: str = "data/statistics") -> pd.DataFrame:
    """
    Load all player statistics from CSV files.
    
    Args:
        data_dir: Directory containing statistics CSV files
        
    Returns:
        Combined DataFrame with all statistics
    """
    all_dfs = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith("_player_statistics.csv"):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            all_dfs.append(df)
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def calculate_player_aggregates(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregate statistics for each player across all seasons.
    
    Args:
        stats_df: DataFrame with player statistics
        
    Returns:
        DataFrame with aggregated statistics per player
    """
    if stats_df.empty:
        return pd.DataFrame()
    
    # Group by player and calculate aggregates
    agg_dict = {
        "appearances": "sum",
        "minutes": "sum",
        "goals": "sum",
        "assists": "sum",
        "shots_total": "sum",
        "shots_on_target": "sum",
        "passes_total": "sum",
        "passes_key": "sum",
        "tackles_total": "sum",
        "duels_won": "sum",
        "dribbles_success": "sum",
        "cards_yellow": "sum",
        "cards_red": "sum",
        "rating": "mean",
        "season": ["min", "max", "count"],
    }
    
    aggregated = stats_df.groupby(["player_id", "player_name", "nationality"]).agg(agg_dict)
    aggregated.columns = ["_".join(col).strip("_") for col in aggregated.columns.values]
    aggregated = aggregated.reset_index()
    
    # Calculate derived metrics
    aggregated["goals_per_90"] = (aggregated["goals_sum"] / aggregated["minutes_sum"]) * 90
    aggregated["assists_per_90"] = (aggregated["assists_sum"] / aggregated["minutes_sum"]) * 90
    aggregated["goal_contributions_per_90"] = (
        (aggregated["goals_sum"] + aggregated["assists_sum"]) / aggregated["minutes_sum"]
    ) * 90
    
    return aggregated


def get_player_career_summary(
    stats_df: pd.DataFrame, 
    player_id: int
) -> Dict:
    """
    Get a comprehensive career summary for a player.
    
    Args:
        stats_df: DataFrame with player statistics
        player_id: Player ID
        
    Returns:
        Dictionary with career summary
    """
    player_stats = stats_df[stats_df["player_id"] == player_id]
    
    if player_stats.empty:
        return {}
    
    # Get basic info from first record
    first_record = player_stats.iloc[0]
    
    return {
        "player_id": player_id,
        "player_name": first_record.get("player_name"),
        "nationality": first_record.get("nationality"),
        "position": player_stats["position"].mode().iloc[0] if not player_stats["position"].mode().empty else None,
        "seasons_played": player_stats["season"].nunique(),
        "teams_played_for": player_stats["team_name"].nunique(),
        "leagues_played_in": player_stats["league_name"].nunique(),
        "total_appearances": player_stats["appearances"].sum(),
        "total_minutes": player_stats["minutes"].sum(),
        "total_goals": player_stats["goals"].sum(),
        "total_assists": player_stats["assists"].sum(),
        "average_rating": player_stats["rating"].mean(),
        "career_yellow_cards": player_stats["cards_yellow"].sum(),
        "career_red_cards": player_stats["cards_red"].sum(),
    }


def get_team_strength_analysis(
    squads_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze team strength based on player statistics.
    
    Args:
        squads_df: DataFrame with squad information
        stats_df: DataFrame with player statistics
        
    Returns:
        DataFrame with team strength analysis
    """
    teams = squads_df["team_name"].unique()
    
    team_analysis = []
    
    for team in teams:
        team_players = squads_df[squads_df["team_name"] == team]["player_id"].tolist()
        team_stats = stats_df[stats_df["player_id"].isin(team_players)]
        
        if team_stats.empty:
            continue
        
        # Get latest season stats
        latest_season = team_stats["season"].max()
        latest_stats = team_stats[team_stats["season"] == latest_season]
        
        team_analysis.append({
            "team_name": team,
            "squad_size": len(team_players),
            "avg_player_age": squads_df[squads_df["team_name"] == team]["age"].mean(),
            "avg_player_rating": latest_stats["rating"].mean(),
            "total_goals_latest": latest_stats["goals"].sum(),
            "total_assists_latest": latest_stats["assists"].sum(),
            "total_appearances_latest": latest_stats["appearances"].sum(),
            "experience_score": team_stats.groupby("player_id")["season"].nunique().mean(),
        })
    
    return pd.DataFrame(team_analysis)


def export_summary_report(
    teams_df: pd.DataFrame,
    squads_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_path: str = "data/summary_report.csv"
):
    """
    Export a summary report combining teams, squads, and statistics.
    
    Args:
        teams_df: DataFrame with team information
        squads_df: DataFrame with squad information
        stats_df: DataFrame with player statistics
        output_path: Path to save the report
    """
    # Calculate team strength
    team_strength = get_team_strength_analysis(squads_df, stats_df)
    
    # Merge with team info
    summary = teams_df.merge(team_strength, on="team_name", how="left")
    
    # Add ranking based on composite score
    if not summary.empty and "avg_player_rating" in summary.columns:
        summary["strength_score"] = (
            summary["avg_player_rating"].fillna(0) * 0.4 +
            summary["experience_score"].fillna(0) * 0.3 +
            (summary["total_goals_latest"].fillna(0) / summary["squad_size"].fillna(1)) * 0.3
        )
        summary = summary.sort_values("strength_score", ascending=False)
        summary["strength_rank"] = range(1, len(summary) + 1)
    
    summary.to_csv(output_path, index=False)
    print(f"Summary report saved to {output_path}")
    
    return summary


def format_player_stats_for_model(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format player statistics for machine learning model input.
    
    Args:
        stats_df: Raw player statistics DataFrame
        
    Returns:
        Formatted DataFrame ready for ML model
    """
    if stats_df.empty:
        return pd.DataFrame()
    
    # Calculate per-90-minute statistics
    stats_df["goals_per_90"] = (stats_df["goals"] / stats_df["minutes"]) * 90
    stats_df["assists_per_90"] = (stats_df["assists"] / stats_df["minutes"]) * 90
    stats_df["shots_per_90"] = (stats_df["shots_total"] / stats_df["minutes"]) * 90
    stats_df["passes_per_90"] = (stats_df["passes_total"] / stats_df["minutes"]) * 90
    stats_df["tackles_per_90"] = (stats_df["tackles_total"] / stats_df["minutes"]) * 90
    stats_df["dribbles_per_90"] = (stats_df["dribbles_success"] / stats_df["minutes"]) * 90
    
    # Handle infinite values (division by zero)
    stats_df = stats_df.replace([float("inf"), float("-inf")], 0)
    stats_df = stats_df.fillna(0)
    
    return stats_df

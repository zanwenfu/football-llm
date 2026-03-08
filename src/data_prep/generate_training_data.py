"""
Step 3: Generate training data in HuggingFace conversational messages format.

Converts match contexts into prompt-completion pairs (system/user/assistant messages)
and splits into train (2010+2014+2018) and eval (2022) JSONL files.

Generates both named-team and anonymized (Team A/Team B) variants.
"""
import json
import os
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
TRAINING_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'training')

SYSTEM_PROMPT = """You are an expert football match analyst specializing in FIFA World Cup predictions. Given detailed statistical profiles of two teams — including their players' recent league performance, historical World Cup records, and head-to-head records — you predict the match outcome with a score and provide reasoning grounded in the data.

# Task
Analyze the provided team profiles and match context, then predict:
1. Match result (home_win, draw, or away_win)
2. Final score
3. Key match events (goal scorers, cards)
4. A reasoning paragraph citing specific statistical evidence

# Guidelines
- Base predictions on statistical evidence, not reputation alone
- Consider both attacking and defensive metrics
- Factor in tournament stage (group vs knockout mentality)
- Account for head-to-head history when available
- Note key player quality and form from recent seasons
- Consider home/away dynamics in the World Cup context"""


def format_player_summary(player: dict) -> str:
    """Format a single player's stats into a readable line."""
    parts = [player.get('name', 'Unknown')]
    pos = player.get('position', '')
    if pos and str(pos) != 'nan' and pos != 'Unknown':
        parts[0] += f" ({pos})"

    stats = []
    if 'appearances' in player and player['appearances'] and player['appearances'] > 0:
        stats.append(f"{int(player['appearances'])} apps")
    if 'goals' in player and player['goals'] and player['goals'] > 0:
        stats.append(f"{int(player['goals'])} goals")
    if 'assists' in player and player['assists'] and player['assists'] > 0:
        stats.append(f"{int(player['assists'])} assists")
    if 'rating' in player and player['rating'] and not (isinstance(player['rating'], float) and str(player['rating']) == 'nan'):
        stats.append(f"rating {player['rating']:.1f}")
    if 'goals_per_90' in player and player['goals_per_90'] and player['goals_per_90'] > 0:
        stats.append(f"{player['goals_per_90']:.2f} goals/90")
    if 'passes_accuracy' in player and player['passes_accuracy'] and not (isinstance(player['passes_accuracy'], float) and str(player['passes_accuracy']) == 'nan'):
        stats.append(f"{player['passes_accuracy']:.0f}% pass acc")

    if stats:
        parts.append(", ".join(stats))
    return " — ".join(parts)


def format_team_profile(profile: dict, team_label: str) -> str:
    """Format a team profile dict into a readable text block."""
    if not profile:
        return f"{team_label}: No detailed statistics available."

    lines = [f"**{team_label}**"]

    # Formation and coach
    formation = profile.get('formation', 'Unknown')
    coach = profile.get('coach', 'Unknown')
    if formation and str(formation) != 'nan' and formation != 'Unknown':
        lines.append(f"Formation: {formation} | Coach: {coach}")
    elif coach and str(coach) != 'nan' and coach != 'Unknown':
        lines.append(f"Coach: {coach}")

    # Team aggregates
    team_stats_lines = []
    stat_labels = {
        'team_total_goals': 'Total Goals (starting XI, prior 3 seasons)',
        'team_total_assists': 'Total Assists',
        'team_avg_rating': 'Avg Player Rating',
        'team_avg_goals_per_90': 'Avg Goals/90',
        'team_avg_assists_per_90': 'Avg Assists/90',
        'team_avg_goal_contributions_per_90': 'Avg Goal Contributions/90',
        'team_avg_passes_accuracy': 'Avg Pass Accuracy',
        'team_avg_tackles_per_90': 'Avg Tackles/90',
        'team_avg_duel_win_pct': 'Avg Duel Win %',
        'team_total_shots_total': 'Total Shots',
        'team_total_shots_on_target': 'Total Shots on Target',
        'team_total_cards_yellow': 'Yellow Cards',
        'team_total_cards_red': 'Red Cards',
        'team_avg_dribble_success_pct': 'Dribble Success %',
    }
    for key, label in stat_labels.items():
        if key in profile and profile[key] is not None:
            val = profile[key]
            # Skip zero values for sum stats (likely missing data)
            if isinstance(val, (int, float)):
                import math
                if math.isnan(val) if isinstance(val, float) else False:
                    continue
                if val == 0 and 'total' in key.lower():
                    continue
            if isinstance(val, float):
                team_stats_lines.append(f"  {label}: {val:.1f}")
            else:
                team_stats_lines.append(f"  {label}: {val}")

    if team_stats_lines:
        lines.append("Team Statistics (aggregated from starting XI's prior 3 league seasons):")
        lines.extend(team_stats_lines)

    # Player summaries
    players = profile.get('player_summaries', [])
    if players:
        lines.append(f"Starting XI ({len(players)} players with data):")
        for p in players:
            lines.append(f"  - {format_player_summary(p)}")

    coverage = profile.get('num_starters_with_data', 0)
    total = profile.get('total_starters', 11)
    if coverage < total:
        lines.append(f"  Note: Data available for {coverage}/{total} starters.")

    return "\n".join(lines)


def format_prior_wc_stats(prior_wc: list, team_label: str) -> str:
    """Format a team's prior WC match history."""
    if not prior_wc:
        return f"No prior World Cup match data available for {team_label}."

    lines = [f"Prior World Cup Performance for {team_label}:"]
    for m in prior_wc[-5:]:  # Last 5 matches
        result_str = m.get('result', 'Unknown')
        score = f"{m.get('goals_scored', '?')}-{m.get('goals_conceded', '?')}"
        line = f"  WC {m['wc_year']} {m.get('stage', '')} vs {m['opponent']}: {score} ({result_str})"
        if m.get('xg') is not None:
            line += f" [xG: {m['xg']:.2f}]"
        if m.get('possession') is not None:
            line += f" [Poss: {m['possession']}%]"
        lines.append(line)

    return "\n".join(lines)


def format_h2h(h2h: dict, home_team: str, away_team: str) -> str:
    """Format head-to-head record."""
    if h2h.get('matches', 0) == 0:
        return f"No prior World Cup meetings between {home_team} and {away_team}."

    return (f"Head-to-head in World Cup ({h2h['matches']} matches): "
            f"{home_team} wins: {h2h['team1_wins']}, "
            f"{away_team} wins: {h2h['team2_wins']}, "
            f"Draws: {h2h['draws']}")


def format_user_message(context: dict, anonymize: bool = False) -> str:
    """Build the user prompt with all team profiles and match context."""
    home_team = "Team A" if anonymize else context['home_team']
    away_team = "Team B" if anonymize else context['away_team']

    home_profile = dict(context.get('home_profile', {}))
    away_profile = dict(context.get('away_profile', {}))

    # Anonymize player summaries if needed
    if anonymize:
        for profile in [home_profile, away_profile]:
            if 'player_summaries' in profile:
                for i, p in enumerate(profile['player_summaries']):
                    p['name'] = f"Player {i+1}"
            if 'team_name' in profile:
                del profile['team_name']

    # Build prompt sections
    sections = []

    # Match context
    sections.append(f"**Match Context**")
    sections.append(f"Tournament: FIFA World Cup {context['world_cup_year']}")
    sections.append(f"Stage: {context['round']}")
    if context.get('venue') and str(context['venue']) != 'nan':
        sections.append(f"Venue: {context['venue']}, {context.get('venue_city', '')}")
    sections.append(f"")

    # Team profiles
    sections.append(format_team_profile(home_profile, f"{home_team} (Home)"))
    sections.append(f"")
    sections.append(format_team_profile(away_profile, f"{away_team} (Away)"))
    sections.append(f"")

    # Prior WC performance
    if anonymize:
        home_prior = [dict(m) for m in context.get('home_prior_wc', [])]
        away_prior = [dict(m) for m in context.get('away_prior_wc', [])]
        for m in home_prior:
            m['opponent'] = 'Opponent'
        for m in away_prior:
            m['opponent'] = 'Opponent'
    else:
        home_prior = context.get('home_prior_wc', [])
        away_prior = context.get('away_prior_wc', [])

    sections.append(format_prior_wc_stats(home_prior, home_team))
    sections.append(format_prior_wc_stats(away_prior, away_team))
    sections.append(f"")

    # H2H
    h2h = context.get('h2h', {})
    if anonymize:
        h2h_text = f"No prior meeting data shown (anonymized)."
    else:
        h2h_text = format_h2h(h2h, home_team, away_team)
    sections.append(h2h_text)
    sections.append(f"")

    sections.append(f"Based on the above profiles and context, predict the match result, score, key events, and provide your reasoning.")

    return "\n".join(sections)


def format_assistant_message(context: dict, anonymize: bool = False) -> str:
    """Build the assistant completion with result, score, events, and reasoning."""
    home_team = "Team A" if anonymize else context['home_team']
    away_team = "Team B" if anonymize else context['away_team']
    home_goals = context['home_goals']
    away_goals = context['away_goals']
    result = context['result']

    lines = []

    # Result and score
    result_map = {
        'home_win': f"{home_team} win",
        'away_win': f"{away_team} win",
        'draw': "Draw"
    }
    lines.append(f"**Prediction: {result_map.get(result, result)}**")
    lines.append(f"Score: {home_team} {home_goals} - {away_goals} {away_team}")
    lines.append(f"")

    # Key events
    events = context.get('events', {})
    goals = events.get('goals', [])
    cards = events.get('cards', [])

    if goals:
        lines.append("**Key Events:**")
        for g in sorted(goals, key=lambda x: int(x.get('time', 0)) if str(x.get('time', 0)).isdigit() else 0):
            team = "Team A" if anonymize and g['team'] == context['home_team'] else \
                   "Team B" if anonymize and g['team'] == context['away_team'] else g['team']
            player = g.get('player', 'Unknown')
            if anonymize:
                player = 'Player'
            time = g.get('time', '?')
            assist = g.get('assist', '')
            goal_type = g.get('type', 'Normal')
            line = f"  ⚽ {time}' - {player} ({team})"
            if goal_type and goal_type != 'Normal':
                line += f" [{goal_type}]"
            if assist and str(assist) != 'nan' and not anonymize:
                line += f" (assist: {assist})"
            lines.append(line)

    if cards:
        for c in sorted(cards, key=lambda x: int(x.get('time', 0)) if str(x.get('time', 0)).isdigit() else 0):
            team = "Team A" if anonymize and c['team'] == context['home_team'] else \
                   "Team B" if anonymize and c['team'] == context['away_team'] else c['team']
            player = c.get('player', 'Unknown')
            if anonymize:
                player = 'Player'
            card_type = c.get('card_type', 'Yellow')
            time = c.get('time', '?')
            lines.append(f"  🟨 {time}' - {player} ({team}) [{card_type}]")

    lines.append(f"")

    # Reasoning
    lines.append("**Reasoning:**")
    reasoning = build_reasoning(context, anonymize)
    lines.append(reasoning)

    # Extra time / penalties info
    if context.get('went_to_extra_time'):
        lines.append(f"\nNote: This match went to extra time.")
    if context.get('went_to_penalties'):
        lines.append(f"Note: This match was decided by penalties.")

    return "\n".join(lines)


def build_reasoning(context: dict, anonymize: bool = False) -> str:
    """
    Build a reasoning paragraph based on the statistical profiles.
    Uses actual data to create a grounded explanation.
    """
    home = "Team A" if anonymize else context['home_team']
    away = "Team B" if anonymize else context['away_team']
    hp = context.get('home_profile', {})
    ap = context.get('away_profile', {})
    result = context['result']
    hg = context['home_goals']
    ag = context['away_goals']

    parts = []

    # Compare attacking output
    h_goals = hp.get('team_avg_goals_per_90', 0) or 0
    a_goals = ap.get('team_avg_goals_per_90', 0) or 0
    h_rating = hp.get('team_avg_rating', 0) or 0
    a_rating = ap.get('team_avg_rating', 0) or 0
    h_pass = hp.get('team_avg_passes_accuracy', 0) or 0
    a_pass = ap.get('team_avg_passes_accuracy', 0) or 0

    if h_goals > 0 or a_goals > 0:
        if h_goals > a_goals * 1.2:
            parts.append(f"{home}'s starting XI showed stronger attacking output in recent seasons "
                         f"(avg {h_goals:.2f} goals/90 vs {away}'s {a_goals:.2f})")
        elif a_goals > h_goals * 1.2:
            parts.append(f"{away}'s starting XI demonstrated superior scoring ability "
                         f"(avg {a_goals:.2f} goals/90 vs {home}'s {h_goals:.2f})")
        else:
            parts.append(f"Both teams showed comparable attacking output "
                         f"({home}: {h_goals:.2f}, {away}: {a_goals:.2f} goals/90)")

    if h_rating > 0 or a_rating > 0:
        if abs(h_rating - a_rating) > 0.2:
            better = home if h_rating > a_rating else away
            parts.append(f"{better}'s players carry a higher average rating "
                         f"({max(h_rating, a_rating):.1f} vs {min(h_rating, a_rating):.1f})")

    if h_pass > 0 or a_pass > 0:
        if abs(h_pass - a_pass) > 5:
            better = home if h_pass > a_pass else away
            parts.append(f"{better} displays better passing accuracy "
                         f"({max(h_pass, a_pass):.0f}% vs {min(h_pass, a_pass):.0f}%)")

    # Defensive comparison
    h_tackles = hp.get('team_avg_tackles_per_90', 0) or 0
    a_tackles = ap.get('team_avg_tackles_per_90', 0) or 0
    h_duels = hp.get('team_avg_duel_win_pct', 0) or 0
    a_duels = ap.get('team_avg_duel_win_pct', 0) or 0

    if h_duels > 0 and a_duels > 0:
        if abs(h_duels - a_duels) > 3:
            better = home if h_duels > a_duels else away
            parts.append(f"{better} is more dominant in physical duels "
                         f"({max(h_duels, a_duels):.0f}% vs {min(h_duels, a_duels):.0f}% win rate)")

    # H2H context
    h2h = context.get('h2h', {})
    if h2h.get('matches', 0) > 0 and not anonymize:
        parts.append(f"Historical WC record shows {h2h['matches']} previous meetings "
                     f"({home} won {h2h['team1_wins']}, {away} won {h2h['team2_wins']}, "
                     f"{h2h['draws']} draws)")

    # Tournament stage
    stage = context.get('round', '')
    if 'knockout' in stage.lower() or 'round of' in stage.lower() or 'quarter' in stage.lower() \
            or 'semi' in stage.lower() or 'final' in stage.lower():
        parts.append("In a knockout match, experience and defensive solidity become critical factors")

    # Result explanation
    if result == 'home_win':
        parts.append(f"The result of {hg}-{ag} reflects {home}'s ability to convert their statistical advantages into goals")
    elif result == 'away_win':
        parts.append(f"The {ag}-{hg} scoreline shows {away} effectively capitalizing despite playing away")
    else:
        parts.append(f"The {hg}-{ag} draw is consistent with the closely matched statistical profiles of both teams")

    return ". ".join(parts) + "." if parts else "Based on the available statistics, the outcome aligns with the relative strengths of both teams."


def create_training_sample(context: dict, anonymize: bool = False) -> dict:
    """Create a single training sample in HF messages format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_message(context, anonymize)},
            {"role": "assistant", "content": format_assistant_message(context, anonymize)},
        ],
        "metadata": {
            "fixture_id": context['fixture_id'],
            "world_cup_year": context['world_cup_year'],
            "home_team": context['home_team'],
            "away_team": context['away_team'],
            "result": context['result'],
            "score": f"{context['home_goals']}-{context['away_goals']}",
            "anonymized": anonymize,
        }
    }


def run():
    """Generate training and eval JSONL files from match contexts."""
    os.makedirs(TRAINING_DIR, exist_ok=True)

    # Load match contexts
    contexts_path = os.path.join(PROCESSED_DIR, 'match_contexts.json')
    with open(contexts_path, 'r') as f:
        contexts = json.load(f)
    logger.info(f"Loaded {len(contexts)} match contexts")

    # Split by year
    train_contexts = [c for c in contexts if c['world_cup_year'] in (2010, 2014, 2018)]
    eval_contexts = [c for c in contexts if c['world_cup_year'] == 2022]
    logger.info(f"Train: {len(train_contexts)} matches (2010+2014+2018)")
    logger.info(f"Eval: {len(eval_contexts)} matches (2022)")

    # Generate samples (both named and anonymized)
    train_samples = []
    eval_samples = []

    for ctx in train_contexts:
        train_samples.append(create_training_sample(ctx, anonymize=False))
        train_samples.append(create_training_sample(ctx, anonymize=True))

    for ctx in eval_contexts:
        eval_samples.append(create_training_sample(ctx, anonymize=False))
        eval_samples.append(create_training_sample(ctx, anonymize=True))

    # Shuffle training data
    random.seed(42)
    random.shuffle(train_samples)

    # Save as JSONL
    train_path = os.path.join(TRAINING_DIR, 'train.jsonl')
    eval_path = os.path.join(TRAINING_DIR, 'eval.jsonl')

    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')

    with open(eval_path, 'w') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + '\n')

    logger.info(f"Saved {len(train_samples)} train samples to {train_path}")
    logger.info(f"Saved {len(eval_samples)} eval samples to {eval_path}")

    # Print a sample
    logger.info("\n--- Sample training example (named) ---")
    named_sample = next(s for s in train_samples if not s['metadata']['anonymized'])
    logger.info(f"User message length: {len(named_sample['messages'][1]['content'])} chars")
    logger.info(f"Assistant message length: {len(named_sample['messages'][2]['content'])} chars")
    logger.info(f"Metadata: {named_sample['metadata']}")

    # Save a human-readable sample
    sample_path = os.path.join(TRAINING_DIR, 'sample_training_example.json')
    with open(sample_path, 'w') as f:
        json.dump(named_sample, f, indent=2)
    logger.info(f"Saved readable sample to {sample_path}")

    return train_samples, eval_samples


if __name__ == '__main__':
    run()

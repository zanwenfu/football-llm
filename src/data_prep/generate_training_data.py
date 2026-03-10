"""
Step 3: Generate COMPACT training data in HuggingFace conversational messages format.

Produces prompts that fit within 768 tokens total:
- System: ~50 tokens (single sentence)
- User: ~200 tokens (team-level aggregates only, no per-player listings)
- Assistant: ~150 tokens (prediction + score + brief reasoning, no events)

Splits: train (2010+2014+2018), eval (2022). Both named + anonymized variants.
"""
import json
import math
import os
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
TRAINING_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'training')

SYSTEM_PROMPT = "You are a football match prediction model. Given team stats, predict the result, score, and brief reasoning."


def _safe(val, default=0):
    """Return val if it's a real number, else default."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return val


def _fmt(val, decimals=1):
    """Format a number, returning '-' if missing/zero."""
    v = _safe(val)
    if v == 0:
        return "-"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(int(v))


def format_team_compact(profile: dict, team_label: str) -> str:
    """Format a team profile into a compact block (~80-120 chars per team)."""
    if not profile:
        return f"{team_label}: No data"

    coach = profile.get('coach', '?')
    if not coach or str(coach) == 'nan':
        coach = '?'
    formation = profile.get('formation', '?')
    if not formation or str(formation) == 'nan':
        formation = '?'

    starters = _safe(profile.get('num_starters_with_data', 11), 11)
    avg_rating = _fmt(profile.get('team_avg_rating'), 1)
    total_goals = _fmt(profile.get('team_total_goals'), 0)
    avg_gp90 = _fmt(profile.get('team_avg_goals_per_90'), 2)
    total_assists = _fmt(profile.get('team_total_assists'), 0)
    yellows = _fmt(profile.get('team_total_cards_yellow'), 0)
    reds = _fmt(profile.get('team_total_cards_red'), 0)
    pass_acc = _fmt(profile.get('team_avg_passes_accuracy'), 0)
    tackles = _fmt(profile.get('team_avg_tackles_per_90'), 2)
    duels = _fmt(profile.get('team_avg_duel_win_pct'), 0)

    # Top scorer from player summaries
    players = profile.get('player_summaries', [])
    top_scorer = 0
    for p in players:
        g = _safe(p.get('goals'))
        if g > top_scorer:
            top_scorer = g
    top_scorer_str = str(int(top_scorer)) if top_scorer > 0 else "-"

    # Position breakdown
    pos = profile.get('position_breakdown', {})
    pos_str = ", ".join(f"{count} {k[:3].upper()}" for k, count in sorted(pos.items(),
                        key=lambda x: ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker', 'Forward'].index(x[0])
                        if x[0] in ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker', 'Forward'] else 99))

    lines = [
        f"{team_label} | Coach: {coach} | Formation: {formation}",
        f"Squad: {starters} starters | Avg Rating: {avg_rating}",
        f"Attack: {total_goals} goals ({avg_gp90}/90) | {total_assists} assists | Top scorer: {top_scorer_str} goals",
        f"Defense: {yellows} yellows, {reds} reds | Tackles/90: {tackles} | Duels: {duels}%",
        f"Passing: {pass_acc}% accuracy",
        f"Positions: {pos_str}" if pos_str else "",
    ]
    return "\n".join(l for l in lines if l)


def format_prior_wc_compact(prior_wc: list) -> str:
    """One-liner WC form summary."""
    if not prior_wc:
        return "No prior WC data"
    results = []
    for m in prior_wc[-3:]:  # Last 3 matches only
        r = m.get('result', '?')[0].upper()  # W/L/D
        score = f"{m.get('goals_scored', '?')}-{m.get('goals_conceded', '?')}"
        results.append(f"{r} {score}")
    return "WC Form: " + ", ".join(results)


def format_h2h_compact(h2h: dict, home: str, away: str) -> str:
    """One-liner H2H."""
    if h2h.get('matches', 0) == 0:
        return "H2H: No prior meetings"
    return (f"H2H ({h2h['matches']} games): "
            f"{home} {h2h['team1_wins']}W, "
            f"{away} {h2h['team2_wins']}W, "
            f"{h2h['draws']}D")


def format_user_message(context: dict, anonymize: bool = False) -> str:
    """Compact user message: ~200 tokens."""
    home = "Team A (Home)" if anonymize else f"{context['home_team']} (Home)"
    away = "Team B (Away)" if anonymize else f"{context['away_team']} (Away)"
    home_name = "Team A" if anonymize else context['home_team']
    away_name = "Team B" if anonymize else context['away_team']

    home_profile = context.get('home_profile', {})
    away_profile = context.get('away_profile', {})

    # Header line
    venue = context.get('venue', '')
    if not venue or str(venue) == 'nan':
        venue = 'Unknown'
    header = f"World Cup {context['world_cup_year']} | {context['round']} | {venue}"

    # Team blocks
    home_block = format_team_compact(home_profile, home)
    away_block = format_team_compact(away_profile, away)

    # Prior WC form
    if anonymize:
        home_prior = context.get('home_prior_wc', [])
        away_prior = context.get('away_prior_wc', [])
    else:
        home_prior = context.get('home_prior_wc', [])
        away_prior = context.get('away_prior_wc', [])
    home_wc = format_prior_wc_compact(home_prior)
    away_wc = format_prior_wc_compact(away_prior)

    # H2H
    h2h = context.get('h2h', {})
    if anonymize:
        h2h_line = "H2H: Hidden (anonymized)"
    else:
        h2h_line = format_h2h_compact(h2h, context['home_team'], context['away_team'])

    sections = [
        header,
        "",
        home_block,
        home_wc,
        "",
        away_block,
        away_wc,
        "",
        h2h_line,
        "",
        "Predict result, score, and reasoning.",
    ]
    return "\n".join(sections)


def format_assistant_message(context: dict, anonymize: bool = False) -> str:
    """Compact assistant: exactly 3 lines — prediction, score, reasoning."""
    home = "Team A" if anonymize else context['home_team']
    away = "Team B" if anonymize else context['away_team']
    hg = context['home_goals']
    ag = context['away_goals']
    result = context['result']

    reasoning = build_compact_reasoning(context, anonymize)

    lines = [
        f"Prediction: {result}",
        f"Score: {hg}-{ag}",
        f"Reasoning: {reasoning}",
    ]
    return "\n".join(lines)


def build_compact_reasoning(context: dict, anonymize: bool = False) -> str:
    """Short reasoning paragraph citing 2-3 key stat differentials."""
    home = "Team A" if anonymize else context['home_team']
    away = "Team B" if anonymize else context['away_team']
    hp = context.get('home_profile', {})
    ap = context.get('away_profile', {})
    result = context['result']

    parts = []

    # Pick the 2-3 most meaningful differentials
    h_goals = _safe(hp.get('team_total_goals'))
    a_goals = _safe(ap.get('team_total_goals'))
    h_gp90 = _safe(hp.get('team_avg_goals_per_90'))
    a_gp90 = _safe(ap.get('team_avg_goals_per_90'))
    h_rating = _safe(hp.get('team_avg_rating'))
    a_rating = _safe(ap.get('team_avg_rating'))
    h_pass = _safe(hp.get('team_avg_passes_accuracy'))
    a_pass = _safe(ap.get('team_avg_passes_accuracy'))

    # Top scorer comparison
    h_top = max((_safe(p.get('goals')) for p in hp.get('player_summaries', [])), default=0)
    a_top = max((_safe(p.get('goals')) for p in ap.get('player_summaries', [])), default=0)

    if h_goals > 0 and a_goals > 0:
        if h_goals > a_goals * 1.15:
            parts.append(f"{home}'s squad has higher goal output ({int(h_goals)} vs {int(a_goals)})")
        elif a_goals > h_goals * 1.15:
            parts.append(f"{away}'s squad has higher goal output ({int(a_goals)} vs {int(h_goals)})")

    if h_rating > 0 and a_rating > 0 and abs(h_rating - a_rating) > 0.15:
        better = home if h_rating > a_rating else away
        parts.append(f"{better} has a higher average rating ({max(h_rating, a_rating):.1f} vs {min(h_rating, a_rating):.1f})")

    if h_top > 0 and a_top > 0 and abs(h_top - a_top) > 5:
        better = home if h_top > a_top else away
        parts.append(f"{better}'s top scorer is more prolific ({int(max(h_top, a_top))} vs {int(min(h_top, a_top))} goals)")

    if h_pass > 0 and a_pass > 0 and abs(h_pass - a_pass) > 3:
        better = home if h_pass > a_pass else away
        parts.append(f"{better} has better passing ({max(h_pass, a_pass):.0f}% vs {min(h_pass, a_pass):.0f}%)")

    # Keep max 3 differentials
    parts = parts[:3]

    if not parts:
        if result == 'draw':
            parts.append("Both teams have closely matched statistical profiles suggesting an even contest")
        else:
            winner = home if result == 'home_win' else away
            parts.append(f"{winner}'s overall squad quality and form give them the edge")

    return ". ".join(parts) + "."


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

    # Token count verification
    logger.info("\n--- Token count verification ---")
    for name, path in [('Train', train_path), ('Eval', eval_path)]:
        max_chars = 0
        total_chars = 0
        min_chars = float('inf')
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            s = json.loads(line)
            chars = sum(len(m['content']) for m in s['messages'])
            max_chars = max(max_chars, chars)
            min_chars = min(min_chars, chars)
            total_chars += chars
        avg_chars = total_chars / len(lines)
        logger.info(f"{name}: {len(lines)} samples | "
                    f"chars min={min_chars} avg={avg_chars:.0f} max={max_chars} | "
                    f"est tokens min~{min_chars//4} avg~{avg_chars//4:.0f} max~{max_chars//4}")
        if max_chars // 4 > 625:
            logger.warning(f"  ⚠️ {name} max chars {max_chars} (~{max_chars//4} tokens) may exceed 768 token limit!")
        else:
            logger.info(f"  ✓ {name} safely within 768 token budget")

    # Print a sample
    logger.info("\n--- Sample training example (named) ---")
    named_sample = next(s for s in train_samples if not s['metadata']['anonymized'])
    for msg in named_sample['messages']:
        logger.info(f"  [{msg['role']}] ({len(msg['content'])} chars):")
        logger.info(f"    {msg['content'][:200]}...")

    # Save a human-readable sample
    sample_path = os.path.join(TRAINING_DIR, 'sample_training_example.json')
    with open(sample_path, 'w') as f:
        json.dump(named_sample, f, indent=2)
    logger.info(f"Saved readable sample to {sample_path}")

    return train_samples, eval_samples


if __name__ == '__main__':
    run()

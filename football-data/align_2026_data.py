#!/usr/bin/env python3
"""
align_2026_data.py

Aligns data/statistics/ (2026 WC team player stats) to exactly match the
column schema of data/wc_player_career_stats_clean/ (history WC career stats).

Operations applied (zero assumptions, zero invented data):
  1. GUARD: verify raw row count hasn't changed
  2. Drop `dribbles_past`           — confirmed 100% null
  3. Add `height_cm`                — parsed from existing `height` API string (strip " cm")
  4. Add `weight_kg`                — parsed from existing `weight` API string (strip " kg")
  5. Add `wc_years = '2026'`        — factual label
  6. Add `wc_teams`                 — copied from `national_team_name` (API data, rename only)
  7. Add `wc_team_ids`              — copied from `national_team_id`   (API data, rename only)
  8. Drop `national_team_name` and `national_team_id`  — redundant after step 6-7 (Option B)
  9. Reorder columns to exactly match history schema
 10. Save master to data/statistics_clean/all_player_statistics_clean.csv
 11. Split per-team and save to data/statistics_clean/{team}_player_statistics_clean.csv

Output directory: data/statistics_clean/  (originals in data/statistics/ untouched)

Known honest differences vs history (not errors, just factual):
  - age:  float64 in 2026 (2 actual nulls from API) vs int64 in history (0 nulls)
  - wc_years: single value '2026' in every row vs comma-separated multi-year strings
  - wc_teams: single team per row (2026 roster) vs may list multiple for multi-WC players
"""

import os
import sys
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = '/Users/zanwenfu/IdeaProject/football-data'
SRC  = os.path.join(ROOT, 'data', 'statistics', 'all_player_statistics.csv')
HIST = os.path.join(ROOT, 'data', 'wc_player_career_stats_clean', 'all_wc_player_career_stats_clean.csv')
DST_DIR = os.path.join(ROOT, 'data', 'statistics_clean')

EXPECTED_ROWS = 55813
EXPECTED_COLS_SRC = 57
EXPECTED_COLS_CLEAN = 59  # must match history

# ── Exact target column order (taken from history file) ───────────────────────
HISTORY_COL_ORDER = [
    'player_id', 'player_name', 'firstname', 'lastname', 'nationality',
    'birth_date', 'birth_place', 'birth_country', 'age',
    'height', 'height_cm', 'weight', 'weight_kg',
    'season', 'team_id', 'team_name', 'league_id', 'league_name', 'league_country',
    'position', 'captain', 'injured',
    'appearances', 'lineups', 'minutes', 'rating',
    'substitutes_in', 'substitutes_out', 'substitutes_bench',
    'shots_total', 'shots_on_target',
    'goals', 'assists', 'penalty_scored', 'penalty_missed',
    'passes_total', 'passes_key', 'passes_accuracy',
    'tackles_total', 'tackles_blocks', 'tackles_interceptions',
    'duels_total', 'duels_won',
    'dribbles_attempts', 'dribbles_success',
    'fouls_drawn', 'fouls_committed',
    'cards_yellow', 'cards_yellowred', 'cards_red',
    'penalty_won', 'penalty_committed',
    'goals_conceded', 'saves', 'penalty_saved',
    'wc_years', 'wc_teams', 'wc_team_ids',
    'photo',
]
assert len(HISTORY_COL_ORDER) == EXPECTED_COLS_CLEAN, \
    f"HISTORY_COL_ORDER has {len(HISTORY_COL_ORDER)} cols, expected {EXPECTED_COLS_CLEAN}"


# ── Parsers (identical logic as used for history data) ────────────────────────
def parse_cm(s):
    """Parse height string '184 cm' or '184' → float, NaN on null/error."""
    if pd.isna(s):
        return float('nan')
    try:
        return float(str(s).strip().replace(' cm', '').replace('cm', ''))
    except ValueError:
        return float('nan')


def parse_kg(s):
    """Parse weight string '78 kg' or '78' → float, NaN on null/error."""
    if pd.isna(s):
        return float('nan')
    try:
        return float(str(s).strip().replace(' kg', '').replace('kg', ''))
    except ValueError:
        return float('nan')


# ─────────────────────────────────────────────────────────────────────────────
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all alignment ops to a dataframe.
    Input:  57-col 2026 schema
    Output: 59-col history schema (column-identical)
    """
    df = df.copy()

    # GUARD: source schema must contain the expected columns
    required_src_cols = {'dribbles_past', 'height', 'weight',
                         'national_team_id', 'national_team_name'}
    missing = required_src_cols - set(df.columns)
    if missing:
        raise ValueError(f"Source data missing expected columns: {missing}")

    # ── Op 2: Drop dribbles_past (100% null) ──────────────────────────────────
    assert df['dribbles_past'].isnull().all(), \
        f"ABORT: dribbles_past has {df['dribbles_past'].notnull().sum()} non-null values — cannot drop safely"
    df = df.drop(columns=['dribbles_past'])

    # ── Op 3: Add height_cm ───────────────────────────────────────────────────
    df['height_cm'] = df['height'].apply(parse_cm)
    # Verify: parse errors = new NaNs beyond existing NaNs
    parse_errors_h = df['height_cm'].isnull().sum() - df['height'].isnull().sum()
    if parse_errors_h != 0:
        raise ValueError(f"height parser introduced {parse_errors_h} unexpected NaNs")

    # ── Op 4: Add weight_kg ───────────────────────────────────────────────────
    df['weight_kg'] = df['weight'].apply(parse_kg)
    parse_errors_w = df['weight_kg'].isnull().sum() - df['weight'].isnull().sum()
    if parse_errors_w != 0:
        raise ValueError(f"weight parser introduced {parse_errors_w} unexpected NaNs")

    # ── Op 5: Add wc_years ────────────────────────────────────────────────────
    df['wc_years'] = '2026'

    # ── Op 6-7: Add wc_teams and wc_team_ids from national_team_* ────────────
    df['wc_teams']    = df['national_team_name']
    df['wc_team_ids'] = df['national_team_id']

    # ── Op 8: Drop national_team_name, national_team_id (Option B) ───────────
    df = df.drop(columns=['national_team_name', 'national_team_id'])

    # ── Op 9: Reorder columns to exactly match history schema ─────────────────
    # Verify the set matches before reordering
    missing_from_df = set(HISTORY_COL_ORDER) - set(df.columns)
    extra_in_df     = set(df.columns) - set(HISTORY_COL_ORDER)
    if missing_from_df:
        raise ValueError(f"After ops, columns still missing: {missing_from_df}")
    if extra_in_df:
        raise ValueError(f"After ops, unexpected extra columns: {extra_in_df}")

    df = df[HISTORY_COL_ORDER]

    assert list(df.columns) == HISTORY_COL_ORDER, "Column order mismatch after reorder"
    assert len(df.columns) == EXPECTED_COLS_CLEAN, \
        f"Expected {EXPECTED_COLS_CLEAN} cols, got {len(df.columns)}"

    return df


# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DST_DIR, exist_ok=True)

    print("=" * 80)
    print("STEP 1 — Loading source files")
    print("=" * 80)
    df_src  = pd.read_csv(SRC)
    df_hist = pd.read_csv(HIST)
    print(f"  2026 source loaded:  {len(df_src):,} rows × {len(df_src.columns)} cols")
    print(f"  History loaded:      {len(df_hist):,} rows × {len(df_hist.columns)} cols")

    # ── Guard: row count must not have changed ────────────────────────────────
    if len(df_src) != EXPECTED_ROWS:
        print(f"  ABORT: expected {EXPECTED_ROWS} rows, found {len(df_src)}")
        sys.exit(1)
    if len(df_src.columns) != EXPECTED_COLS_SRC:
        print(f"  ABORT: expected {EXPECTED_COLS_SRC} cols, found {len(df_src.columns)}")
        sys.exit(1)
    print(f"  ✓ Row and column count guards passed")

    # ── Verify history column order matches our constant ──────────────────────
    if list(df_hist.columns) != HISTORY_COL_ORDER:
        print(f"  ABORT: history column order does not match HISTORY_COL_ORDER constant")
        print(f"  History cols:   {list(df_hist.columns)}")
        print(f"  Expected order: {HISTORY_COL_ORDER}")
        sys.exit(1)
    print(f"  ✓ History column order verified against constant")

    print("\n" + "=" * 80)
    print("STEP 2 — Transforming master file")
    print("=" * 80)
    df_clean = transform(df_src)
    print(f"  ✓ Transform complete: {len(df_clean):,} rows × {len(df_clean.columns)} cols")
    print(f"  ✓ Row count preserved: {len(df_clean)} == {EXPECTED_ROWS}")
    print(f"  ✓ Column count:        {len(df_clean.columns)} == {EXPECTED_COLS_CLEAN}")
    print(f"  ✓ Columns match history: {list(df_clean.columns) == list(df_hist.columns)}")

    print("\n" + "=" * 80)
    print("STEP 3 — Saving clean master")
    print("=" * 80)
    master_out = os.path.join(DST_DIR, 'all_player_statistics_clean.csv')
    df_clean.to_csv(master_out, index=False)
    print(f"  ✓ Saved: {master_out}")

    print("\n" + "=" * 80)
    print("STEP 4 — Splitting per-team files from clean master")
    print("=" * 80)
    teams = df_clean['wc_teams'].unique()
    print(f"  Teams to write: {len(teams)}")
    total_rows_written = 0
    team_file_map = {}

    for team in sorted(teams):
        team_df = df_clean[df_clean['wc_teams'] == team].copy()
        # Filename: same convention as source files (lowercase, spaces→underscores)
        team_slug = team.lower().replace(' ', '_').replace("'", '').replace('-', '_')
        fname = f"{team_slug}_player_statistics_clean.csv"
        fpath = os.path.join(DST_DIR, fname)
        team_df.to_csv(fpath, index=False)
        total_rows_written += len(team_df)
        team_file_map[team] = (fname, len(team_df))

    print(f"  ✓ {len(teams)} team files written")
    print(f"  ✓ Total rows across team files: {total_rows_written:,}")
    if total_rows_written != EXPECTED_ROWS:
        print(f"  ⚠️  ROW COUNT MISMATCH: expected {EXPECTED_ROWS}, got {total_rows_written}")
        sys.exit(1)
    else:
        print(f"  ✓ Row count matches master ({EXPECTED_ROWS:,})")

    print("\n" + "=" * 80)
    print("STEP 5 — Final verification: 2026 clean vs history")
    print("=" * 80)
    run_final_check(df_clean, df_hist, team_file_map)


def run_final_check(df_clean: pd.DataFrame, df_hist: pd.DataFrame, team_file_map: dict):
    """
    Comprehensive cross-check of the clean 2026 data against history.
    Every assertion is intentional — failure means something went wrong.
    """
    all_ok = True

    def check(label, condition, detail=""):
        nonlocal all_ok
        status = "✓" if condition else "✗"
        msg = f"  {status} {label}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        if not condition:
            all_ok = False

    # ── Schema alignment ──────────────────────────────────────────────────────
    print("\n[Schema]")
    check("Column count matches history",
          len(df_clean.columns) == len(df_hist.columns),
          f"2026={len(df_clean.columns)}, hist={len(df_hist.columns)}")
    check("Column names match history exactly",
          list(df_clean.columns) == list(df_hist.columns))
    check("Column ORDER matches history exactly",
          list(df_clean.columns) == HISTORY_COL_ORDER)

    # ── Row integrity ─────────────────────────────────────────────────────────
    print("\n[Row Integrity]")
    check("Row count preserved",
          len(df_clean) == EXPECTED_ROWS,
          f"{len(df_clean):,} rows")
    # Note: (player_id, season, team_id) is NOT unique by design — the API returns
    # one row per player per season per competition (league, cup, European etc.).
    # A player at FC Basel in 2015 may have 3 rows: Champions League, Europa League,
    # domestic league.  This is identical behaviour in both history and 2026 data.
    # The only valid uniqueness check is: zero FULLY IDENTICAL rows.
    full_dupes = df_clean.duplicated().sum()
    check("No fully identical rows (true duplicates)",
          full_dupes == 0,
          f"{full_dupes} identical rows")

    # ── Ops verification ──────────────────────────────────────────────────────
    print("\n[Operation Verification]")
    check("dribbles_past not in 2026 clean",
          'dribbles_past' not in df_clean.columns)
    check("national_team_id not in 2026 clean (Option B)",
          'national_team_id' not in df_clean.columns)
    check("national_team_name not in 2026 clean (Option B)",
          'national_team_name' not in df_clean.columns)
    check("wc_years column present and all = '2026'",
          'wc_years' in df_clean.columns and (df_clean['wc_years'] == '2026').all())
    check("wc_teams has no nulls",
          df_clean['wc_teams'].isnull().sum() == 0)
    check("wc_team_ids has no nulls",
          df_clean['wc_team_ids'].isnull().sum() == 0)
    check("height_cm parse errors = 0",
          df_clean['height_cm'].isnull().sum() == df_clean['height'].isnull().sum(),
          f"height nulls: raw={df_clean['height'].isnull().sum()}, parsed={df_clean['height_cm'].isnull().sum()}")
    check("weight_kg parse errors = 0",
          df_clean['weight_kg'].isnull().sum() == df_clean['weight'].isnull().sum(),
          f"weight nulls: raw={df_clean['weight'].isnull().sum()}, parsed={df_clean['weight_kg'].isnull().sum()}")
    check("height_cm range sane [140, 220] cm",
          df_clean['height_cm'].dropna().between(140, 220).all(),
          f"min={df_clean['height_cm'].min():.0f}, max={df_clean['height_cm'].max():.0f}")
    check("weight_kg range sane [40, 130] kg",
          df_clean['weight_kg'].dropna().between(40, 130).all(),
          f"min={df_clean['weight_kg'].min():.0f}, max={df_clean['weight_kg'].max():.0f}")

    # ── Shared columns: stat data unchanged ──────────────────────────────────
    print("\n[Stat Data Integrity — shared columns unchanged]")
    shared_stats = [
        'appearances', 'lineups', 'minutes', 'goals', 'assists',
        'cards_yellow', 'cards_red', 'passes_total', 'shots_total',
    ]
    df_src = pd.read_csv(SRC)
    for col in shared_stats:
        orig_sum = df_src[col].sum()
        clean_sum = df_clean[col].sum()
        check(f"{col} sum unchanged after transform",
              orig_sum == clean_sum,
              f"orig={orig_sum:.0f}, clean={clean_sum:.0f}")

    # ── Team coverage ─────────────────────────────────────────────────────────
    print("\n[Team Coverage]")
    check("42 teams in clean 2026 data",
          df_clean['wc_teams'].nunique() == 42,
          f"found {df_clean['wc_teams'].nunique()}")
    check(f"{len(team_file_map)} per-team files written",
          len(team_file_map) == 42,
          f"found {len(team_file_map)}")

    hist_teams = set(df_hist['wc_teams'].str.split(',').explode().str.strip().unique())
    clean_teams = set(df_clean['wc_teams'].unique())
    overlap = hist_teams & clean_teams
    new_2026 = clean_teams - hist_teams
    check("34 teams overlap with history",
          len(overlap) == 34,
          f"found {len(overlap)}")
    check("8 new teams in 2026 not in history",
          len(new_2026) == 8,
          f"found {len(new_2026)}: {sorted(new_2026)}")

    # ── dtype alignment ───────────────────────────────────────────────────────
    print("\n[DType Notes (documented differences)]")
    age_2026_dtype = str(df_clean['age'].dtype)
    age_hist_dtype = str(df_hist['age'].dtype)
    age_nulls = df_clean['age'].isnull().sum()
    print(f"  ℹ️  age: 2026={age_2026_dtype} ({age_nulls} API nulls), history={age_hist_dtype} (0 nulls) — expected difference")

    # All other numeric columns should be float64 (nulls present → float)
    numeric_cols = [c for c in HISTORY_COL_ORDER
                    if df_clean[c].dtype in ('float64', 'int64')
                    and c not in ('player_id', 'season', 'age', 'wc_team_ids')]
    type_mismatch = [(c, str(df_clean[c].dtype), str(df_hist[c].dtype))
                     for c in numeric_cols if str(df_clean[c].dtype) != str(df_hist[c].dtype)]
    check("All numeric stat columns have same dtype as history",
          len(type_mismatch) == 0,
          f"mismatches: {type_mismatch}" if type_mismatch else "")

    print("\n" + "=" * 80)
    if all_ok:
        print("ALL CHECKS PASSED ✓")
    else:
        print("SOME CHECKS FAILED ✗ — review output above")
    print("=" * 80)


if __name__ == '__main__':
    main()

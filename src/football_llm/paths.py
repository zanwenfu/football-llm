"""Canonical filesystem paths for the repo.

Importing from this module means no more `os.path.dirname(__file__)` chains
scattered across scripts. All paths are absolute and computed once at import.
"""

from __future__ import annotations

from pathlib import Path

# src/football_llm/paths.py → repo root is 3 levels up.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
TRAINING_DIR: Path = DATA_DIR / "training"

RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = REPO_ROOT / "figures"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"

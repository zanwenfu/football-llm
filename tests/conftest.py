"""Shared fixtures for the Football-LLM test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    from football_llm.paths import REPO_ROOT

    return REPO_ROOT


@pytest.fixture(scope="session")
def results_dir(repo_root: Path) -> Path:
    return repo_root / "results"


@pytest.fixture(scope="session")
def pregame_df(results_dir: Path):
    """Load the committed pregame predictions once per session."""
    from football_llm.eval.loader import load_predictions

    return load_predictions("pregame", results_dir=results_dir)


@pytest.fixture(scope="session")
def halftime_df(results_dir: Path):
    from football_llm.eval.loader import load_predictions

    return load_predictions("halftime", results_dir=results_dir)

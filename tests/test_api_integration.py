"""FastAPI integration tests — routes, validation, auth.

The /predict endpoint is mocked via monkeypatching the OpenAI client so these
tests run without a live vLLM server.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from football_llm.serving import api as api_module

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs):
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content: str = "Prediction: home_win\nScore: 3-1\nReasoning: solid form."):
        self.chat = type("C", (), {"completions": _FakeChatCompletions(content)})()
        self.models = type("M", (), {"list": lambda self=None: type("R", (), {"data": []})()})()


@pytest.fixture
def client(monkeypatch):
    """FastAPI test client with a mocked vLLM backend."""
    fake = _FakeClient()
    monkeypatch.setattr(api_module, "get_client", lambda: fake)
    monkeypatch.setattr(api_module, "API_TOKEN", None)  # auth disabled by default
    return TestClient(api_module.app)


@pytest.fixture
def authed_client(monkeypatch):
    """Test client with auth enabled."""
    fake = _FakeClient()
    monkeypatch.setattr(api_module, "get_client", lambda: fake)
    monkeypatch.setattr(api_module, "API_TOKEN", "s3cr3t")
    return TestClient(api_module.app)


# ---------------------------------------------------------------------------
# Request body helpers
# ---------------------------------------------------------------------------


def _sample_body(regime: str = "pregame", **kwargs) -> dict:
    body = {
        "home_team": {
            "name": "Argentina",
            "goals": 450,
            "goals_per_90": 0.35,
            "assists": 180,
            "avg_rating": 7.2,
            "top_scorer_goals": 200,
            "yellows": 120,
            "reds": 2,
            "tackles_per_90": 0.6,
            "duels_pct": 55,
            "pass_accuracy": 72,
            "formation": "4-3-3",
            "coach": "Scaloni",
        },
        "away_team": {
            "name": "France",
            "goals": 420,
            "goals_per_90": 0.38,
            "assists": 170,
            "avg_rating": 7.3,
            "top_scorer_goals": 190,
            "yellows": 100,
            "reds": 1,
            "tackles_per_90": 0.55,
            "duels_pct": 54,
            "pass_accuracy": 74,
            "formation": "4-2-3-1",
            "coach": "Deschamps",
        },
        "match": {"tournament": "WC 2022", "stage": "Final", "venue": "Lusail"},
        "regime": regime,
    }
    body.update(kwargs)
    return body


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_version_and_auth_state(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "vllm_connected" in data
        assert data["version"] == api_module.app.version
        assert data["auth_enabled"] is False


class TestPredictPregame:
    def test_happy_path(self, client):
        r = client.post("/predict", json=_sample_body("pregame"))
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["prediction"] == "home_win"
        assert data["score"] == "3-1"
        assert data["pred_home"] == 3
        assert data["pred_away"] == 1
        assert 0.0 <= data["over_2_5_probability"] <= 1.0
        assert data["regime"] == "pregame"

    def test_invalid_rating_rejected(self, client):
        body = _sample_body("pregame")
        body["home_team"]["avg_rating"] = 15  # violates 0-10 bound
        r = client.post("/predict", json=body)
        assert r.status_code == 422


class TestPredictHalftime:
    def test_requires_halftime_score(self, client):
        """Schema validation: halftime regime needs halftime_score."""
        r = client.post("/predict", json=_sample_body("halftime"))
        assert r.status_code == 422

    def test_happy_path(self, client):
        body = _sample_body("halftime", halftime_score={"home": 1, "away": 0})
        r = client.post("/predict", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["regime"] == "halftime"


class TestPredictHalftimeEvents:
    def test_requires_events(self, client):
        """halftime_events regime requires non-empty first_half_events."""
        body = _sample_body("halftime_events", halftime_score={"home": 1, "away": 0})
        r = client.post("/predict", json=body)
        assert r.status_code == 422

    def test_happy_path(self, client):
        body = _sample_body(
            "halftime_events",
            halftime_score={"home": 2, "away": 2},
            first_half_events=["10' Argentina goal", "30' France yellow card"],
        )
        r = client.post("/predict", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["regime"] == "halftime_events"


class TestAuth:
    def test_missing_header_returns_401(self, authed_client):
        r = authed_client.post("/predict", json=_sample_body("pregame"))
        assert r.status_code == 401

    def test_wrong_token_returns_401(self, authed_client):
        r = authed_client.post(
            "/predict",
            json=_sample_body("pregame"),
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 401

    def test_right_token_passes(self, authed_client):
        r = authed_client.post(
            "/predict",
            json=_sample_body("pregame"),
            headers={"Authorization": "Bearer s3cr3t"},
        )
        assert r.status_code == 200

"""Integration tests for the HTTP server using Starlette test client."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from starlette.testclient import TestClient

from umbra.config import UmbraConfig
from umbra.server import UmbraServer


@pytest.fixture
def config():
    return UmbraConfig(
        api_key="test_key_123",
        policy="enforce",
        episode_size=3,
    )


@pytest.fixture
def mock_session_check():
    """Mock SessionManager.check to avoid real API calls."""
    with patch("umbra.sessions.SessionManager.check", new_callable=AsyncMock) as m:
        yield m


@pytest.fixture
def mock_session_create():
    """Mock the fleet session creation."""
    with patch("umbra.sessions.SessionManager._create_fleet_session", new_callable=AsyncMock) as m:
        m.return_value = "test-session-id"
        yield m


@pytest.fixture
def app(config):
    gate = UmbraServer(config)
    return gate.build_app()


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["policy"] == "enforce"


class TestCheck:
    def test_missing_body(self, client):
        resp = client.post("/check", content=b"not json", headers={"content-type": "application/json"})
        assert resp.status_code == 400

    def test_missing_action(self, client):
        resp = client.post("/check", json={})
        assert resp.status_code == 400
        assert "action" in resp.json()["error"]

    def test_invalid_action_type(self, client):
        resp = client.post("/check", json={"action": 12345})
        assert resp.status_code == 400

    def test_buffered_response(self, client, mock_session_check):
        """When check returns None (still buffering), response says buffered."""
        mock_session_check.return_value = None
        resp = client.post("/check", json={"action": "file_read", "agent": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["buffered"] is True
        assert data["decision"] == "allow"

    def test_round_response(self, client, mock_session_check):
        """When check returns a round result, response has full policy decision."""
        mock_session_check.return_value = {
            "snapshot": {
                "nodes": [{
                    "ci_out": 45000,
                    "ci_ema_out": 40000,
                    "al_out": 3,
                    "ghost_suspect": False,
                    "ghost_confirmed": False,
                    "warn": True,
                    "fault": False,
                }]
            },
            "round": 5,
            "credits_remaining": 950,
        }
        resp = client.post("/check", json={"action": "terminal_exec", "agent": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "gate"  # AL3 in enforce mode
        assert data["al"] == 3
        assert data["agent"] == "test"

    def test_custom_score(self, client, mock_session_check):
        """Can send raw Q0.16 score instead of action type."""
        mock_session_check.return_value = None
        resp = client.post("/check", json={"score": 50000, "agent": "test"})
        assert resp.status_code == 200


class TestReport:
    def test_report_accepted(self, client, mock_session_check):
        mock_session_check.return_value = None
        resp = client.post("/report", json={"action": "file_read", "agent": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True


class TestStatus:
    def test_status_empty(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agents"] == []
        assert data["policy"] == "enforce"

    def test_agent_status_404(self, client):
        resp = client.get("/status/nonexistent")
        assert resp.status_code == 404


class TestDeleteSession:
    def test_delete_nonexistent(self, client):
        resp = client.delete("/sessions/nonexistent")
        assert resp.status_code == 404


class TestDecisions:
    def test_decisions_empty(self, client):
        resp = client.get("/decisions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["decisions"] == []

    def test_decisions_populated(self, client, mock_session_check):
        """After a round result, /decisions should have an entry."""
        mock_session_check.return_value = {
            "snapshot": {
                "nodes": [{
                    "ci_out": 30000,
                    "ci_ema_out": 28000,
                    "al_out": 2,
                    "ghost_suspect": False,
                    "ghost_confirmed": False,
                }]
            },
            "round": 1,
            "credits_remaining": 900,
        }
        client.post("/check", json={"action": "file_write", "agent": "obs-test"})
        resp = client.get("/decisions")
        data = resp.json()
        assert len(data["decisions"]) == 1
        assert data["decisions"][0]["agent"] == "obs-test"
        assert data["decisions"][0]["decision"] == "warn"

    def test_decisions_agent_filter(self, client, mock_session_check):
        """Agent query param should filter decisions."""
        mock_session_check.return_value = {
            "snapshot": {"nodes": [{"ci_out": 0, "ci_ema_out": 0, "al_out": 0}]},
            "round": 1, "credits_remaining": 900,
        }
        client.post("/check", json={"action": "file_read", "agent": "a1"})
        client.post("/check", json={"action": "file_read", "agent": "a2"})
        resp = client.get("/decisions?agent=a1")
        data = resp.json()
        assert all(d["agent"] == "a1" for d in data["decisions"])


class TestEpisodes:
    def test_episodes_empty(self, client):
        resp = client.get("/episodes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["episodes"] == []

    def test_episodes_populated(self, client, mock_session_check):
        """After a round result, /episodes should have an entry with CI data."""
        mock_session_check.return_value = {
            "snapshot": {
                "nodes": [{
                    "ci_out": 45000,
                    "ci_ema_out": 40000,
                    "al_out": 3,
                    "ghost_suspect": True,
                    "ghost_confirmed": False,
                }]
            },
            "round": 2,
            "credits_remaining": 800,
        }
        client.post("/check", json={"action": "terminal_exec", "agent": "ep-test"})
        resp = client.get("/episodes")
        data = resp.json()
        assert len(data["episodes"]) == 1
        ep = data["episodes"][0]
        assert ep["agent"] == "ep-test"
        assert ep["al"] == 3
        assert ep["ghost_suspect"] is True
        assert "ci" in ep


# ── Security ──


class TestSecurity:
    def test_invalid_content_length(self, client):
        """Content-Length with non-numeric value should return 400, not crash."""
        resp = client.post(
            "/check",
            content=b'{"action": "test"}',
            headers={"content-type": "application/json", "content-length": "abc"},
        )
        assert resp.status_code == 400

    def test_oversized_body(self, client):
        """Body exceeding MAX_BODY_SIZE should be rejected."""
        resp = client.post(
            "/check",
            content=b'{"action": "' + b"x" * 70000 + b'"}',
            headers={"content-type": "application/json", "content-length": "70016"},
        )
        assert resp.status_code == 413

    def test_control_chars_in_action(self, client):
        """Action with control characters should be rejected."""
        resp = client.post("/check", json={"action": "file_read\r\n", "agent": "test"})
        assert resp.status_code == 400

    def test_non_dict_body(self, client):
        """Array body should be rejected."""
        resp = client.post(
            "/check",
            content=b'[1, 2, 3]',
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_score_out_of_range(self, client):
        """Score > 65535 should be rejected."""
        resp = client.post("/check", json={"score": 99999})
        assert resp.status_code == 400

    def test_negative_score(self, client):
        """Negative score should be rejected."""
        resp = client.post("/check", json={"score": -1})
        assert resp.status_code == 400

    def test_health_uses_version(self, client):
        """Health endpoint should use __version__, not hardcoded string."""
        from umbra import __version__
        resp = client.get("/health")
        assert resp.json()["version"] == __version__

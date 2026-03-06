"""Tests for multi-agent coordination: influence gating, cascade, consensus."""

from __future__ import annotations

import time

import pytest
from unittest.mock import AsyncMock, patch

from starlette.testclient import TestClient

from umbra.config import UmbraConfig, MultiAgentConfig
from umbra.multi import (
    CausalGraph,
    InfluenceGate,
    compute_cascade_penalties,
    compute_consensus_al,
    CASCADE_CI_THRESHOLD,
)
from umbra.scorer import Q16_MAX
from umbra.server import UmbraServer


# ── CausalGraph ──


class TestCausalGraph:
    def test_record_and_get_upstream(self):
        g = CausalGraph()
        g.record("agent-A", "agent-B")
        upstream = g.get_upstream("agent-B")
        assert ("agent-A", 1) in upstream

    def test_self_reference_ignored(self):
        g = CausalGraph()
        g.record("agent-A", "agent-A")
        assert g.get_upstream("agent-A") == []

    def test_multi_hop(self):
        g = CausalGraph()
        g.record("agent-A", "agent-B")
        g.record("agent-B", "agent-C")
        upstream = g.get_upstream("agent-C")
        agents = {name for name, _ in upstream}
        assert "agent-B" in agents
        assert "agent-A" in agents

    def test_hop_distances(self):
        g = CausalGraph()
        g.record("agent-A", "agent-B")
        g.record("agent-B", "agent-C")
        upstream = dict(g.get_upstream("agent-C"))
        assert upstream["agent-B"] == 1
        assert upstream["agent-A"] == 2

    def test_max_hops_respected(self):
        g = CausalGraph()
        g.record("a1", "a2")
        g.record("a2", "a3")
        g.record("a3", "a4")
        g.record("a4", "a5")
        g.record("a5", "a6")
        # Max 4 hops: a6 can see a5(1), a4(2), a3(3), a2(4) but NOT a1(5)
        upstream = dict(g.get_upstream("a6", max_hops=4))
        assert "a5" in upstream
        assert "a2" in upstream
        assert "a1" not in upstream

    def test_expired_edges_pruned(self):
        g = CausalGraph(edge_ttl=1)
        g.record("agent-A", "agent-B")
        # Manually expire the edge
        for e in g._edges.get("agent-B", []):
            e.timestamp = time.time() - 2
        upstream = g.get_upstream("agent-B")
        assert upstream == []

    def test_clear_agent(self):
        g = CausalGraph()
        g.record("agent-A", "agent-B")
        g.record("agent-B", "agent-C")
        g.clear_agent("agent-B")
        # agent-B's incoming edges gone
        assert g.get_upstream("agent-B") == []
        # agent-B as a source for agent-C also gone
        upstream_c = g.get_upstream("agent-C")
        agents = {name for name, _ in upstream_c}
        assert "agent-B" not in agents

    def test_no_cycles(self):
        g = CausalGraph()
        g.record("a", "b")
        g.record("b", "c")
        g.record("c", "a")  # cycle
        # Should not infinite loop
        upstream = g.get_upstream("a")
        agents = {name for name, _ in upstream}
        assert "c" in agents
        assert "b" in agents

    def test_empty_graph(self):
        g = CausalGraph()
        assert g.get_upstream("nonexistent") == []


# ── InfluenceGate ──


class TestInfluenceGate:
    def test_monitor_mode_always_allows(self):
        allowed, _ = InfluenceGate.check(
            triggering_al=4, target_action_risk=0.90, policy_enabled=False
        )
        assert allowed

    def test_high_risk_blocked_by_low_authority(self):
        allowed, reason = InfluenceGate.check(
            triggering_al=3, target_action_risk=0.70, policy_enabled=True
        )
        assert not allowed
        assert "AL3" in reason

    def test_high_risk_allowed_by_high_authority(self):
        allowed, _ = InfluenceGate.check(
            triggering_al=0, target_action_risk=0.90, policy_enabled=True
        )
        assert allowed

    def test_medium_risk_blocked_by_al3(self):
        allowed, _ = InfluenceGate.check(
            triggering_al=3, target_action_risk=0.40, policy_enabled=True
        )
        assert not allowed

    def test_medium_risk_allowed_by_al2(self):
        allowed, _ = InfluenceGate.check(
            triggering_al=2, target_action_risk=0.40, policy_enabled=True
        )
        assert allowed

    def test_low_risk_always_allowed(self):
        allowed, _ = InfluenceGate.check(
            triggering_al=4, target_action_risk=0.20, policy_enabled=True
        )
        assert allowed


# ── Cascade Penalties ──


class TestCascadePenalties:
    def test_no_penalty_below_threshold(self):
        penalties = compute_cascade_penalties(
            ci_raw=CASCADE_CI_THRESHOLD - 1,
            upstream=[("agent-A", 1)],
        )
        assert penalties == []

    def test_penalty_above_threshold(self):
        ci = int(0.8 * Q16_MAX)
        penalties = compute_cascade_penalties(
            ci_raw=ci,
            upstream=[("agent-A", 1)],
        )
        assert len(penalties) == 1
        name, score = penalties[0]
        assert name == "agent-A"
        # 50% attenuation at 1 hop
        assert score == int(ci * 0.5)

    def test_attenuation_by_distance(self):
        ci = int(0.8 * Q16_MAX)
        penalties = compute_cascade_penalties(
            ci_raw=ci,
            upstream=[("a1", 1), ("a2", 2), ("a3", 3)],
        )
        scores = {name: score for name, score in penalties}
        assert scores["a1"] > scores["a2"] > scores["a3"]
        assert scores["a1"] == int(ci * 0.5)
        assert scores["a2"] == int(ci * 0.25)
        assert scores["a3"] == int(ci * 0.125)

    def test_empty_upstream(self):
        penalties = compute_cascade_penalties(
            ci_raw=Q16_MAX,
            upstream=[],
        )
        assert penalties == []


# ── Consensus ──


class TestConsensus:
    def test_empty_group(self):
        al, explanation = compute_consensus_al({})
        assert al == 4
        assert "no agents" in explanation

    def test_single_agent(self):
        al, _ = compute_consensus_al({"agent-A": 1})
        assert al == 1

    def test_weakest_link(self):
        al, explanation = compute_consensus_al({
            "agent-A": 0,
            "agent-B": 1,
            "agent-C": 3,
        })
        assert al == 3
        assert "agent-C" in explanation

    def test_all_same(self):
        al, explanation = compute_consensus_al({
            "a": 2, "b": 2, "c": 2,
        })
        assert al == 2
        assert "all agents" in explanation

    def test_multiple_weakest(self):
        al, explanation = compute_consensus_al({
            "a": 1, "b": 4, "c": 4,
        })
        assert al == 4
        assert "b" in explanation
        assert "c" in explanation


# ── Server Integration ──


@pytest.fixture
def multi_config():
    return UmbraConfig(
        api_key="test_key_123",
        policy="enforce",
        episode_size=3,
        multi_agent=MultiAgentConfig(enabled=True),
    )


@pytest.fixture
def multi_app(multi_config):
    gate = UmbraServer(multi_config)
    return gate.build_app()


@pytest.fixture
def multi_client(multi_app):
    return TestClient(multi_app, raise_server_exceptions=False)


@pytest.fixture
def mock_session_check():
    with patch("umbra.sessions.SessionManager.check", new_callable=AsyncMock) as m:
        yield m


@pytest.fixture
def mock_session_status():
    with patch("umbra.sessions.SessionManager.get_status", new_callable=AsyncMock) as m:
        yield m


class TestServerInfluenceGating:
    def test_triggered_by_blocked(self, multi_client, mock_session_check, mock_session_status):
        """An agent at AL3 should not be able to trigger a high-risk action."""
        mock_session_status.return_value = {
            "last_episode": {"al_out": 3},
        }
        mock_session_check.return_value = None

        resp = multi_client.post("/check", json={
            "agent": "target-agent",
            "action": "credential_access",  # risk 0.90
            "triggered_by": "weak-agent",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "block"
        assert data["influence_gated"] is True

    def test_triggered_by_allowed(self, multi_client, mock_session_check, mock_session_status):
        """An agent at AL0 should be able to trigger any action."""
        mock_session_status.return_value = {
            "last_episode": {"al_out": 0},
        }
        mock_session_check.return_value = None

        resp = multi_client.post("/check", json={
            "agent": "target-agent",
            "action": "credential_access",
            "triggered_by": "strong-agent",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("influence_gated") is not True
        assert data["decision"] == "allow"  # buffered

    def test_triggered_by_low_risk_always_allowed(
        self, multi_client, mock_session_check, mock_session_status
    ):
        """Low-risk actions are allowed regardless of triggering agent's AL."""
        mock_session_status.return_value = {
            "last_episode": {"al_out": 4},
        }
        mock_session_check.return_value = None

        resp = multi_client.post("/check", json={
            "agent": "target",
            "action": "file_read",  # risk 0.05
            "triggered_by": "worst-agent",
        })
        data = resp.json()
        assert data.get("influence_gated") is not True


class TestServerConsensus:
    def test_consensus_endpoint(self, multi_client, mock_session_status):
        """Consensus endpoint returns the weakest AL."""
        async def status_side_effect(agent):
            als = {"a1": 0, "a2": 1, "a3": 3}
            al = als.get(agent, 4)
            return {"last_episode": {"al_out": al}}

        mock_session_status.side_effect = status_side_effect

        resp = multi_client.post("/consensus", json={
            "agents": ["a1", "a2", "a3"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["effective_al"] == 3
        assert "a3" in data["explanation"]

    def test_consensus_unknown_agents(self, multi_client, mock_session_status):
        """Unknown agents default to AL4."""
        mock_session_status.return_value = None

        resp = multi_client.post("/consensus", json={
            "agents": ["unknown-1", "unknown-2"],
        })
        data = resp.json()
        assert data["effective_al"] == 4

    def test_consensus_empty_list(self, multi_client):
        resp = multi_client.post("/consensus", json={"agents": []})
        assert resp.status_code == 400

    def test_consensus_missing_field(self, multi_client):
        resp = multi_client.post("/consensus", json={})
        assert resp.status_code == 400


class TestServerCausal:
    def test_causal_endpoint_empty(self, multi_client):
        resp = multi_client.get("/causal/test-agent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "test-agent"
        assert data["upstream"] == []

    def test_causal_after_triggered_by(self, multi_client, mock_session_check, mock_session_status):
        """After a triggered_by check, the causal graph should contain the edge."""
        mock_session_status.return_value = {
            "last_episode": {"al_out": 0},
        }
        mock_session_check.return_value = None

        # Agent-B triggers an action in Agent-C
        multi_client.post("/check", json={
            "agent": "agent-C",
            "action": "file_read",
            "triggered_by": "agent-B",
        })

        resp = multi_client.get("/causal/agent-C")
        data = resp.json()
        assert len(data["upstream"]) == 1
        assert data["upstream"][0]["agent"] == "agent-B"
        assert data["upstream"][0]["hops"] == 1


class TestServerTriggeredByValidation:
    def test_triggered_by_control_chars_rejected(self, multi_client):
        resp = multi_client.post("/check", json={
            "agent": "test",
            "action": "file_read",
            "triggered_by": "evil\r\nagent",
        })
        assert resp.status_code == 400

    def test_triggered_by_too_long_rejected(self, multi_client):
        resp = multi_client.post("/check", json={
            "agent": "test",
            "action": "file_read",
            "triggered_by": "a" * 200,
        })
        assert resp.status_code == 400


class TestMultiAgentDisabled:
    """When multi_agent.enabled is False, triggered_by has no effect."""

    @pytest.fixture
    def disabled_config(self):
        return UmbraConfig(
            api_key="test_key_123",
            policy="enforce",
            episode_size=3,
            multi_agent=MultiAgentConfig(enabled=False),
        )

    @pytest.fixture
    def disabled_client(self, disabled_config):
        gate = UmbraServer(disabled_config)
        app = gate.build_app()
        return TestClient(app, raise_server_exceptions=False)

    def test_triggered_by_ignored_when_disabled(self, disabled_client, mock_session_check):
        mock_session_check.return_value = None
        resp = disabled_client.post("/check", json={
            "agent": "target",
            "action": "credential_access",
            "triggered_by": "weak-agent",
        })
        data = resp.json()
        # Should not be influence gated when disabled
        assert data.get("influence_gated") is not True
        assert data["decision"] == "allow"

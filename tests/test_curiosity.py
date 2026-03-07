"""Tests for CI-C1: Structural Curiosity Engine."""

from __future__ import annotations

import math
import time
from collections import deque

import pytest
from unittest.mock import AsyncMock, patch

from starlette.testclient import TestClient

from umbra.config import UmbraConfig, CuriosityConfig
from umbra.curiosity import (
    BehavioralFingerprint,
    CuriosityEngine,
    Discovery,
    EpisodeRecord,
    build_fingerprint,
    cosine_similarity,
    explain_match,
    structural_similarity,
)
from umbra.server import UmbraServer


# ── Cosine Similarity ──


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = (0.5, 0.1, 0.3, 0.2, 0.8, 0.1, 0.0, 0.0)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        b = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        other = (0.5, 0.1, 0.3, 0.2, 0.8, 0.1, 0.0, 0.0)
        assert cosine_similarity(zero, other) == 0.0
        assert cosine_similarity(other, zero) == 0.0

    def test_similar_vectors_high_score(self):
        a = (0.5, 0.1, 0.3, 0.2, 0.8, 0.1, 0.0, 0.0)
        b = (0.5, 0.12, 0.28, 0.22, 0.78, 0.1, 0.0, 0.0)
        sim = cosine_similarity(a, b)
        assert sim > 0.99

    def test_opposite_vectors(self):
        a = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        b = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)


# ── Structural Similarity ──


class TestStructuralSimilarity:
    def test_identical_fingerprints_score_1(self):
        fp = BehavioralFingerprint(
            agent="a", window_start=0, window_end=100, window_index=0,
            mean_ci=0.5, std_ci=0.1, trend=0.0, volatility=0.0,
            max_ci=0.6, min_ci=0.4, al_changes=0.0, ghost_rate=0.0,
        )
        assert structural_similarity(fp, fp) == pytest.approx(1.0)

    def test_same_shape_different_magnitude_penalized(self):
        """The bug: flat-low and flat-high agents matched at 95% with cosine.
        Structural similarity must penalize the magnitude difference."""
        fp_low = BehavioralFingerprint(
            agent="a", window_start=0, window_end=100, window_index=0,
            mean_ci=0.17, std_ci=0.01, trend=0.0, volatility=0.0,
            max_ci=0.18, min_ci=0.15, al_changes=0.0, ghost_rate=0.0,
        )
        fp_high = BehavioralFingerprint(
            agent="b", window_start=0, window_end=100, window_index=0,
            mean_ci=0.35, std_ci=0.003, trend=0.0, volatility=0.0,
            max_ci=0.35, min_ci=0.34, al_changes=0.0, ghost_rate=0.0,
        )
        # Cosine would give ~0.95+; structural should be meaningfully lower
        cosine_score = cosine_similarity(fp_low.vector(), fp_high.vector())
        structural_score = structural_similarity(fp_low, fp_high)
        assert structural_score < cosine_score
        assert structural_score < 0.90  # Must not false-match at 90%

    def test_same_shape_same_magnitude_stays_high(self):
        """Two agents at similar CI levels with similar shapes should still match."""
        fp_a = BehavioralFingerprint(
            agent="a", window_start=0, window_end=100, window_index=0,
            mean_ci=0.17, std_ci=0.01, trend=0.006, volatility=0.0,
            max_ci=0.18, min_ci=0.15, al_changes=0.0, ghost_rate=0.0,
        )
        fp_b = BehavioralFingerprint(
            agent="b", window_start=0, window_end=100, window_index=0,
            mean_ci=0.18, std_ci=0.012, trend=0.005, volatility=0.0,
            max_ci=0.19, min_ci=0.16, al_changes=0.0, ghost_rate=0.0,
        )
        score = structural_similarity(fp_a, fp_b)
        assert score > 0.90

    def test_magnitude_differs_on_only_mean_ci(self):
        """Even if only mean_ci differs, it should hurt the score."""
        fp_a = BehavioralFingerprint(
            agent="a", window_start=0, window_end=100, window_index=0,
            mean_ci=0.1, std_ci=0.0, trend=0.0, volatility=0.0,
            max_ci=0.1, min_ci=0.1, al_changes=0.0, ghost_rate=0.0,
        )
        fp_b = BehavioralFingerprint(
            agent="b", window_start=0, window_end=100, window_index=0,
            mean_ci=0.9, std_ci=0.0, trend=0.0, volatility=0.0,
            max_ci=0.9, min_ci=0.9, al_changes=0.0, ghost_rate=0.0,
        )
        score = structural_similarity(fp_a, fp_b)
        # Huge magnitude gap should heavily penalize
        assert score < 0.70

    def test_cycle_rejects_different_magnitude_agents(self):
        """End-to-end: two agents with same shape but different CI levels
        should NOT produce a cross_agent discovery at threshold 0.9."""
        engine = CuriosityEngine(
            window_size=5, similarity_threshold=0.9, cooldown_cycles=0,
        )
        # Agent a: low CI (stable)
        for i, ci in enumerate([0.17, 0.18, 0.16, 0.17, 0.18]):
            engine._episodes.setdefault("a", deque(maxlen=500))
            engine._episodes["a"].append(EpisodeRecord(
                agent="a", ci=ci, al=1, ghost_suspect=False,
                timestamp=1000.0 + i * 10.0,
            ))
        # Agent b: high CI (risky) but same flat shape
        for i, ci in enumerate([0.35, 0.36, 0.34, 0.35, 0.36]):
            engine._episodes.setdefault("b", deque(maxlen=500))
            engine._episodes["b"].append(EpisodeRecord(
                agent="b", ci=ci, al=1, ghost_suspect=False,
                timestamp=1000.0 + i * 10.0,
            ))
        discoveries = engine._run_cycle()
        cross = [d for d in discoveries if d.match_type == "cross_agent"]
        # Should NOT match at 0.9 threshold anymore
        assert len(cross) == 0


# ── Build Fingerprint ──


def _make_episodes(
    agent: str,
    ci_values: list[float],
    al_values: list[int] | None = None,
    ghost_values: list[bool] | None = None,
    start_time: float = 1000.0,
) -> list[EpisodeRecord]:
    """Helper to create a list of EpisodeRecord from CI values."""
    n = len(ci_values)
    if al_values is None:
        al_values = [1] * n
    if ghost_values is None:
        ghost_values = [False] * n
    return [
        EpisodeRecord(
            agent=agent,
            ci=ci_values[i],
            al=al_values[i],
            ghost_suspect=ghost_values[i],
            timestamp=start_time + i * 10.0,
        )
        for i in range(n)
    ]


class TestBuildFingerprint:
    def test_too_few_episodes_returns_none(self):
        eps = _make_episodes("a", [0.5, 0.6])
        assert build_fingerprint("a", eps) is None

    def test_basic_fingerprint(self):
        eps = _make_episodes("a", [0.2, 0.4, 0.6, 0.8, 1.0])
        fp = build_fingerprint("a", eps)
        assert fp is not None
        assert fp.agent == "a"
        assert fp.mean_ci == pytest.approx(0.6)
        assert fp.max_ci == pytest.approx(1.0)
        assert fp.min_ci == pytest.approx(0.2)
        # Positive trend (CI increasing = degrading)
        assert fp.trend > 0

    def test_flat_signal_zero_std(self):
        eps = _make_episodes("a", [0.5, 0.5, 0.5, 0.5])
        fp = build_fingerprint("a", eps)
        assert fp is not None
        assert fp.std_ci == pytest.approx(0.0)
        assert fp.volatility == pytest.approx(0.0)
        assert fp.trend == pytest.approx(0.0)

    def test_al_changes_tracked(self):
        eps = _make_episodes("a", [0.5] * 5, al_values=[1, 1, 2, 2, 3])
        fp = build_fingerprint("a", eps)
        assert fp is not None
        # 2 transitions: 1->2 and 2->3
        assert fp.al_changes == pytest.approx(2 / 5)

    def test_ghost_rate(self):
        eps = _make_episodes(
            "a", [0.5] * 4,
            ghost_values=[True, False, True, False],
        )
        fp = build_fingerprint("a", eps)
        assert fp is not None
        assert fp.ghost_rate == pytest.approx(0.5)

    def test_volatility_zigzag(self):
        # Zigzag: up, down, up, down -> 3 direction changes / (5-2) = 1.0
        eps = _make_episodes("a", [0.2, 0.8, 0.2, 0.8, 0.2])
        fp = build_fingerprint("a", eps)
        assert fp is not None
        assert fp.volatility == pytest.approx(1.0)

    def test_volatility_monotone(self):
        # Monotonically increasing: 0 direction changes
        eps = _make_episodes("a", [0.1, 0.3, 0.5, 0.7, 0.9])
        fp = build_fingerprint("a", eps)
        assert fp is not None
        assert fp.volatility == pytest.approx(0.0)

    def test_vector_has_8_dimensions(self):
        eps = _make_episodes("a", [0.2, 0.4, 0.6])
        fp = build_fingerprint("a", eps)
        assert fp is not None
        vec = fp.vector()
        assert len(vec) == 8

    def test_window_timestamps(self):
        eps = _make_episodes("a", [0.2, 0.4, 0.6], start_time=100.0)
        fp = build_fingerprint("a", eps)
        assert fp is not None
        assert fp.window_start == 100.0
        assert fp.window_end == 120.0


# ── Discovery Serialization ──


class TestDiscovery:
    def test_to_dict_structure(self):
        fp_a = BehavioralFingerprint(
            agent="a", window_start=100, window_end=200, window_index=0,
        )
        fp_b = BehavioralFingerprint(
            agent="a", window_start=0, window_end=100, window_index=2,
        )
        disc = Discovery(
            match_type="self_temporal",
            agent_a="a", agent_b="a",
            similarity=0.85,
            fingerprint_a=fp_a, fingerprint_b=fp_b,
            explanation="test", cycle=3,
        )
        d = disc.to_dict()
        assert d["match_type"] == "self_temporal"
        assert d["similarity"] == 0.85
        assert d["cycle"] == 3
        assert d["window_a"]["index"] == 0
        assert d["window_b"]["index"] == 2


# ── Explain Match ──


class TestExplainMatch:
    def test_self_temporal_explanation(self):
        fp_a = BehavioralFingerprint(
            agent="bot", window_start=0, window_end=500, window_index=0,
            mean_ci=0.4, trend=0.1,
        )
        fp_b = BehavioralFingerprint(
            agent="bot", window_start=0, window_end=100, window_index=3,
            mean_ci=0.38, trend=0.12,
        )
        text = explain_match("self_temporal", "bot", "bot", fp_a, fp_b, 0.92)
        assert "bot" in text
        assert "92%" in text
        assert "resembles" in text

    def test_cross_agent_explanation(self):
        fp_a = BehavioralFingerprint(
            agent="a", window_start=0, window_end=100, window_index=0,
            mean_ci=0.5, trend=-0.1,
        )
        fp_b = BehavioralFingerprint(
            agent="b", window_start=0, window_end=100, window_index=0,
            mean_ci=0.48, trend=-0.08,
        )
        text = explain_match("cross_agent", "a", "b", fp_a, fp_b, 0.95)
        assert "a" in text
        assert "b" in text
        assert "Correlated" in text

    def test_cross_temporal_explanation(self):
        fp_a = BehavioralFingerprint(
            agent="a", window_start=0, window_end=500, window_index=0,
            mean_ci=0.6, trend=0.2,
        )
        fp_b = BehavioralFingerprint(
            agent="b", window_start=0, window_end=100, window_index=2,
            mean_ci=0.58, trend=0.18,
        )
        text = explain_match("cross_temporal", "a", "b", fp_a, fp_b, 0.88)
        assert "a" in text
        assert "b" in text
        assert "88%" in text


# ── CuriosityEngine: Recording ──


class TestCuriosityRecord:
    def test_record_creates_episode(self):
        engine = CuriosityEngine()
        engine.record("agent-a", ci=0.5, al=1, ghost_suspect=False)
        assert "agent-a" in engine._episodes
        assert len(engine._episodes["agent-a"]) == 1

    def test_record_multiple_agents(self):
        engine = CuriosityEngine()
        engine.record("a", ci=0.3, al=0, ghost_suspect=False)
        engine.record("b", ci=0.7, al=3, ghost_suspect=True)
        assert len(engine._episodes) == 2

    def test_record_respects_max_agents(self):
        engine = CuriosityEngine()
        # Fill to max
        for i in range(200):
            engine.record(f"agent-{i}", ci=0.5, al=1, ghost_suspect=False)
        assert len(engine._episodes) == 200
        # One more should be ignored
        engine.record("overflow", ci=0.5, al=1, ghost_suspect=False)
        assert "overflow" not in engine._episodes

    def test_record_respects_max_episodes_per_agent(self):
        engine = CuriosityEngine()
        for i in range(600):
            engine.record("a", ci=float(i % 100) / 100, al=1, ghost_suspect=False)
        assert len(engine._episodes["a"]) == 500


# ── CuriosityEngine: Cycle ──


class TestCuriosityCycle:
    def _populate_agent(
        self, engine: CuriosityEngine, agent: str,
        ci_values: list[float], al: int = 1, ghost: bool = False,
    ):
        """Inject episodes directly into the engine."""
        for i, ci in enumerate(ci_values):
            engine._episodes.setdefault(agent, __import__("collections").deque(maxlen=500))
            engine._episodes[agent].append(EpisodeRecord(
                agent=agent, ci=ci, al=al, ghost_suspect=ghost,
                timestamp=1000.0 + i * 10.0,
            ))

    def test_no_discoveries_with_insufficient_data(self):
        engine = CuriosityEngine(window_size=5)
        self._populate_agent(engine, "a", [0.5, 0.5, 0.5])  # only 3 episodes
        discoveries = engine._run_cycle()
        assert discoveries == []

    def test_self_temporal_discovery(self):
        engine = CuriosityEngine(
            window_size=5, history_depth=5, similarity_threshold=0.8,
            cooldown_cycles=0,
        )
        # Create a pattern, then different behavior, then same pattern again.
        # Pattern: [0.2, 0.4, 0.6, 0.8, 1.0] (rising)
        # Different: [0.5, 0.5, 0.5, 0.5, 0.5] (flat) x2
        # Same again: [0.2, 0.4, 0.6, 0.8, 1.0] (rising)
        ci_stream = (
            [0.2, 0.4, 0.6, 0.8, 1.0]  # window index 3 (oldest)
            + [0.5, 0.5, 0.5, 0.5, 0.5]  # window index 2
            + [0.5, 0.5, 0.5, 0.5, 0.5]  # window index 1 (skipped)
            + [0.2, 0.4, 0.6, 0.8, 1.0]  # window index 0 (current)
        )
        self._populate_agent(engine, "bot", ci_stream)
        discoveries = engine._run_cycle()
        self_disc = [d for d in discoveries if d.match_type == "self_temporal"]
        assert len(self_disc) >= 1
        assert self_disc[0].agent_a == "bot"
        assert self_disc[0].similarity > 0.8

    def test_cross_agent_discovery(self):
        engine = CuriosityEngine(
            window_size=5, similarity_threshold=0.9, cooldown_cycles=0,
        )
        # Two agents with nearly identical behavior
        pattern = [0.1, 0.3, 0.5, 0.7, 0.9]
        self._populate_agent(engine, "agent-a", pattern)
        self._populate_agent(engine, "agent-b", pattern)
        discoveries = engine._run_cycle()
        cross_disc = [d for d in discoveries if d.match_type == "cross_agent"]
        assert len(cross_disc) == 1
        assert cross_disc[0].similarity > 0.99

    def test_cross_temporal_discovery(self):
        engine = CuriosityEngine(
            window_size=5, history_depth=3, similarity_threshold=0.8,
            cooldown_cycles=0,
        )
        # Agent B had a pattern in the past
        b_stream = (
            [0.2, 0.4, 0.6, 0.8, 1.0]  # B's past window
            + [0.5, 0.5, 0.5, 0.5, 0.5]  # B's current (different)
        )
        self._populate_agent(engine, "agent-b", b_stream)
        # Agent A currently matches B's past pattern
        self._populate_agent(engine, "agent-a", [0.2, 0.4, 0.6, 0.8, 1.0])
        discoveries = engine._run_cycle()
        temp_disc = [d for d in discoveries if d.match_type == "cross_temporal"]
        assert len(temp_disc) >= 1
        # A's current should match B's past
        match = [d for d in temp_disc if d.agent_a == "agent-a" and d.agent_b == "agent-b"]
        assert len(match) >= 1

    def test_no_discovery_below_threshold(self):
        engine = CuriosityEngine(
            window_size=5, similarity_threshold=0.99, cooldown_cycles=0,
        )
        # Two very different patterns
        self._populate_agent(engine, "a", [0.1, 0.1, 0.1, 0.1, 0.1])
        self._populate_agent(engine, "b", [0.1, 0.9, 0.1, 0.9, 0.1])
        discoveries = engine._run_cycle()
        cross_disc = [d for d in discoveries if d.match_type == "cross_agent"]
        assert len(cross_disc) == 0

    def test_cycle_count_increments(self):
        engine = CuriosityEngine(window_size=3)
        self._populate_agent(engine, "a", [0.5, 0.5, 0.5])
        assert engine.cycle_count == 0
        engine._run_cycle()
        assert engine.cycle_count == 1
        engine._run_cycle()
        assert engine.cycle_count == 2


# ── Cooldown ──


class TestCooldown:
    def _populate_agent(self, engine, agent, ci_values):
        for i, ci in enumerate(ci_values):
            engine._episodes.setdefault(agent, __import__("collections").deque(maxlen=500))
            engine._episodes[agent].append(EpisodeRecord(
                agent=agent, ci=ci, al=1, ghost_suspect=False,
                timestamp=1000.0 + i * 10.0,
            ))

    def test_cooldown_suppresses_repeat_discovery(self):
        engine = CuriosityEngine(
            window_size=5, similarity_threshold=0.9, cooldown_cycles=3,
        )
        pattern = [0.1, 0.3, 0.5, 0.7, 0.9]
        self._populate_agent(engine, "a", pattern)
        self._populate_agent(engine, "b", pattern)

        # First cycle: should discover
        d1 = engine._run_cycle()
        cross = [d for d in d1 if d.match_type == "cross_agent"]
        assert len(cross) == 1

        # Second cycle: should be suppressed (cooldown=3)
        d2 = engine._run_cycle()
        cross2 = [d for d in d2 if d.match_type == "cross_agent"]
        assert len(cross2) == 0

    def test_cooldown_expires(self):
        engine = CuriosityEngine(
            window_size=5, similarity_threshold=0.9, cooldown_cycles=2,
        )
        pattern = [0.1, 0.3, 0.5, 0.7, 0.9]
        self._populate_agent(engine, "a", pattern)
        self._populate_agent(engine, "b", pattern)

        # Cycle 1: discover
        engine._run_cycle()
        # Cycle 2: cooldown
        engine._run_cycle()
        # Cycle 3: cooldown (cycle_count=3, last=1, diff=2 == cooldown_cycles)
        d3 = engine._run_cycle()
        cross3 = [d for d in d3 if d.match_type == "cross_agent"]
        assert len(cross3) == 1


# ── Clear Agent ──


class TestClearAgent:
    def test_clear_removes_episodes_and_fingerprints(self):
        engine = CuriosityEngine(window_size=3)
        engine.record("a", 0.5, 1, False)
        engine.record("a", 0.5, 1, False)
        engine.record("a", 0.5, 1, False)
        engine._run_cycle()
        assert "a" in engine._episodes
        engine.clear_agent("a")
        assert "a" not in engine._episodes
        assert "a" not in engine._fingerprints


# ── Get Discoveries ──


class TestGetDiscoveries:
    def _populate_agent(self, engine, agent, ci_values):
        for i, ci in enumerate(ci_values):
            engine._episodes.setdefault(agent, __import__("collections").deque(maxlen=500))
            engine._episodes[agent].append(EpisodeRecord(
                agent=agent, ci=ci, al=1, ghost_suspect=False,
                timestamp=1000.0 + i * 10.0,
            ))

    def test_get_discoveries_empty(self):
        engine = CuriosityEngine()
        assert engine.get_discoveries() == []

    def test_get_discoveries_filtered_by_agent(self):
        engine = CuriosityEngine(
            window_size=5, similarity_threshold=0.9, cooldown_cycles=0,
        )
        pattern = [0.1, 0.3, 0.5, 0.7, 0.9]
        self._populate_agent(engine, "a", pattern)
        self._populate_agent(engine, "b", pattern)
        self._populate_agent(engine, "c", [0.9, 0.9, 0.9, 0.9, 0.9])
        engine._run_cycle()

        # Filter by agent a
        disc_a = engine.get_discoveries(agent="a")
        for d in disc_a:
            assert d["agent_a"] == "a" or d["agent_b"] == "a"

    def test_get_discoveries_limit(self):
        engine = CuriosityEngine(
            window_size=5, similarity_threshold=0.5, cooldown_cycles=0,
        )
        # Create many agents with similar patterns to generate discoveries
        for i in range(10):
            self._populate_agent(engine, f"agent-{i}", [0.1, 0.3, 0.5, 0.7, 0.9])
        engine._run_cycle()
        limited = engine.get_discoveries(limit=3)
        assert len(limited) <= 3


# ── Server Integration ──


def _make_config_with_curiosity(**overrides) -> UmbraConfig:
    """Create a config with curiosity enabled."""
    cfg = UmbraConfig(
        api_key="test_key_123",
        policy="enforce",
        episode_size=3,
        curiosity=CuriosityConfig(enabled=True, **overrides),
    )
    return cfg


@pytest.fixture
def curiosity_config():
    return _make_config_with_curiosity()


@pytest.fixture
def curiosity_app(curiosity_config):
    gate = UmbraServer(curiosity_config)
    return gate.build_app()


@pytest.fixture
def curiosity_client(curiosity_app):
    return TestClient(curiosity_app, raise_server_exceptions=False)


class TestServerDiscoveriesEndpoint:
    def test_discoveries_enabled(self, curiosity_client):
        resp = curiosity_client.get("/discoveries")
        assert resp.status_code == 200
        data = resp.json()
        assert "discoveries" in data
        assert "cycle_count" in data

    def test_discoveries_disabled(self):
        cfg = UmbraConfig(api_key="test_key_123", policy="enforce")
        app = UmbraServer(cfg).build_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/discoveries")
        assert resp.status_code == 404

    def test_discoveries_with_agent_filter(self, curiosity_client):
        resp = curiosity_client.get("/discoveries?agent=bot-1")
        assert resp.status_code == 200

    def test_discoveries_with_limit(self, curiosity_client):
        resp = curiosity_client.get("/discoveries?limit=5")
        assert resp.status_code == 200

    def test_discoveries_invalid_limit_falls_back(self, curiosity_client):
        resp = curiosity_client.get("/discoveries?limit=abc")
        assert resp.status_code == 200


class TestServerCuriosityRecording:
    """Verify that the server records episodes to the curiosity engine."""

    def test_check_records_to_curiosity(self, curiosity_config):
        """After a round fires, the episode should be recorded in the curiosity engine."""
        gate = UmbraServer(curiosity_config)
        app = gate.build_app()
        client = TestClient(app, raise_server_exceptions=False)

        round_result = {
            "snapshot": {
                "nodes": [{
                    "ci_out": 19660,
                    "ci_ema_out": 18000,
                    "al_out": 2,
                    "ghost_suspect": False,
                    "ghost_confirmed": False,
                }]
            },
            "round": 1,
            "credits_remaining": 100,
        }

        with patch("umbra.sessions.SessionManager.check", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = round_result
            with patch("umbra.sessions.SessionManager._create_fleet_session", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = "test-session-id"
                client.post("/check", json={"agent": "bot-1", "action": "file_read"})

        # Check that curiosity engine recorded the episode
        assert gate.curiosity is not None
        assert "bot-1" in gate.curiosity._episodes
        assert len(gate.curiosity._episodes["bot-1"]) == 1


class TestServerCuriosityDisabled:
    def test_no_curiosity_engine_when_disabled(self):
        cfg = UmbraConfig(api_key="test_key_123", policy="enforce")
        gate = UmbraServer(cfg)
        assert gate.curiosity is None

    def test_curiosity_engine_created_when_enabled(self):
        cfg = _make_config_with_curiosity()
        gate = UmbraServer(cfg)
        assert gate.curiosity is not None


# ── Config Parsing ──


class TestCuriosityConfig:
    def test_default_config(self):
        cfg = CuriosityConfig()
        assert cfg.enabled is False
        assert cfg.cycle_interval == 60
        assert cfg.window_size == 10

    def test_custom_config(self):
        cfg = CuriosityConfig(
            enabled=True,
            cycle_interval=30,
            window_size=20,
            similarity_threshold=0.8,
        )
        assert cfg.enabled is True
        assert cfg.cycle_interval == 30
        assert cfg.window_size == 20
        assert cfg.similarity_threshold == 0.8

    def test_config_in_umbra_config(self):
        cfg = UmbraConfig(
            api_key="test",
            curiosity=CuriosityConfig(enabled=True, cycle_interval=30),
        )
        assert cfg.curiosity.enabled is True
        assert cfg.curiosity.cycle_interval == 30

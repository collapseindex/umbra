"""Tests for Umbra core modules."""

from __future__ import annotations

import pytest

from umbra.scorer import Q16_MAX, RiskConfig, to_q16
from umbra.policy import PolicyDecision, PolicyGate, PolicyResult
from umbra.config import (
    UmbraConfig, load_config, DEFAULT_RISK_MAP,
    mask_key, sanitize_agent_name, MAX_SESSIONS,
)


# ── Scorer ──


class TestToQ16:
    def test_float_zero(self):
        assert to_q16(0.0) == 0

    def test_float_one(self):
        assert to_q16(1.0) == Q16_MAX

    def test_float_half(self):
        assert to_q16(0.5) == 32767

    def test_int_passthrough(self):
        assert to_q16(42000) == 42000

    def test_float_out_of_range(self):
        with pytest.raises(ValueError):
            to_q16(1.5)

    def test_int_out_of_range(self):
        with pytest.raises(ValueError):
            to_q16(70000)


class TestRiskConfig:
    def test_known_action(self):
        rc = RiskConfig(risk_map={"file_read": 0.05, "unknown": 0.50})
        score = rc.score("file_read")
        # 0.05 +/- 0.02 jitter -> roughly 0.03-0.07 -> ~1966-4588
        assert 0 <= score <= Q16_MAX
        assert score < Q16_MAX // 4  # should be well below 25%

    def test_unknown_action(self):
        rc = RiskConfig(risk_map={"unknown": 0.50})
        score = rc.score("totally_made_up")
        # 0.50 +/- 0.02 -> roughly 31457-34078
        assert Q16_MAX // 4 < score < (Q16_MAX * 3) // 4

    def test_escalation_boost(self):
        rc = RiskConfig(risk_map={"file_read": 0.05, "unknown": 0.50})
        normal = rc.score("file_read", is_escalation=False)
        boosted = rc.score("file_read", is_escalation=True)
        # Escalation adds 0.25, so boosted should be much higher (on average)
        # Can't test exact values due to jitter, but boosted center is 0.30 vs 0.05
        # Run multiple times to avoid flaky test
        total_n = 0
        total_b = 0
        for _ in range(100):
            total_n += rc.score("file_read", is_escalation=False)
            total_b += rc.score("file_read", is_escalation=True)
        assert total_b > total_n  # on average, boosted is higher

    def test_from_dict(self):
        rc = RiskConfig.from_dict({"file_read": 0.90, "custom_action": 0.75})
        assert rc.risk_map["file_read"] == 0.90
        assert rc.risk_map["custom_action"] == 0.75

    def test_dotted_namespace_exact_match(self):
        rc = RiskConfig(risk_map={"cloud.deploy": 0.80, "unknown": 0.50})
        score = rc.score("cloud.deploy")
        # 0.80 +/- 0.02 jitter -> should be high
        assert score > Q16_MAX // 2

    def test_dotted_namespace_domain_fallback(self):
        rc = RiskConfig(risk_map={"cloud": 0.70, "unknown": 0.50})
        score = rc.score("cloud.deploy")
        # Falls back to "cloud" domain -> 0.70 +/- 0.02
        assert score > Q16_MAX // 3

    def test_dotted_namespace_unknown_fallback(self):
        rc = RiskConfig(risk_map={"unknown": 0.50})
        score = rc.score("browser.click")
        # No match for "browser.click" or "browser", falls to unknown 0.50
        assert Q16_MAX // 4 < score < (Q16_MAX * 3) // 4


# ── Policy ──


class TestPolicyGate:
    def _episode(self, al: int = 0, ci: int = 0, ghost_suspect: bool = False, ghost_confirmed: bool = False):
        return {
            "al_out": al,
            "ci_out": ci,
            "ci_ema_out": ci,
            "ghost_suspect": ghost_suspect,
            "ghost_confirmed": ghost_confirmed,
        }

    def test_monitor_mode_always_allows(self):
        gate = PolicyGate(enabled=False)
        result = gate.check("test-agent", self._episode(al=4))
        assert result.decision == PolicyDecision.ALLOW

    def test_enforce_al0_allows(self):
        gate = PolicyGate(enabled=True)
        result = gate.check("test-agent", self._episode(al=0))
        assert result.decision == PolicyDecision.ALLOW

    def test_enforce_al1_allows(self):
        gate = PolicyGate(enabled=True)
        result = gate.check("test-agent", self._episode(al=1))
        assert result.decision == PolicyDecision.ALLOW

    def test_enforce_al2_warns(self):
        gate = PolicyGate(enabled=True)
        result = gate.check("test-agent", self._episode(al=2))
        assert result.decision == PolicyDecision.WARN

    def test_enforce_al3_gates(self):
        gate = PolicyGate(enabled=True)
        result = gate.check("test-agent", self._episode(al=3))
        assert result.decision == PolicyDecision.GATE

    def test_enforce_al4_blocks(self):
        gate = PolicyGate(enabled=True)
        result = gate.check("test-agent", self._episode(al=4))
        assert result.decision == PolicyDecision.BLOCK

    def test_ghost_confirmed_blocks(self):
        gate = PolicyGate(enabled=True)
        result = gate.check("test-agent", self._episode(al=0, ghost_confirmed=True))
        assert result.decision == PolicyDecision.BLOCK

    def test_to_dict(self):
        gate = PolicyGate(enabled=True)
        result = gate.check("test-agent", self._episode(al=3, ci=32768))
        d = result.to_dict()
        assert d["decision"] == "gate"
        assert d["al"] == 3
        assert d["agent"] == "test-agent"
        assert "ci" in d
        assert "ghost_suspect" in d

    def test_history_bounded(self):
        gate = PolicyGate(enabled=False)
        for i in range(1100):
            gate.check(f"agent-{i}", self._episode())
        assert len(gate.history) <= 1000


# ── Config ──


class TestConfig:
    def test_defaults(self):
        cfg = UmbraConfig()
        assert cfg.port == 8400
        assert cfg.policy == "monitor"
        assert cfg.episode_size == 3
        assert not cfg.enforce

    def test_enforce_property(self):
        cfg = UmbraConfig(policy="enforce")
        assert cfg.enforce is True

    def test_validate_no_key(self):
        cfg = UmbraConfig(api_key="")
        errors = cfg.validate()
        assert any("api_key" in e for e in errors)

    def test_validate_bad_port(self):
        cfg = UmbraConfig(api_key="test", port=99999)
        errors = cfg.validate()
        assert any("port" in e for e in errors)

    def test_validate_bad_episode_size(self):
        cfg = UmbraConfig(api_key="test", episode_size=1)
        errors = cfg.validate()
        assert any("episode_size" in e for e in errors)

    def test_validate_ok(self):
        cfg = UmbraConfig(api_key="test_key")
        errors = cfg.validate()
        assert errors == []

    def test_default_risk_map_has_basics(self):
        assert "file_read" in DEFAULT_RISK_MAP
        assert "terminal_exec" in DEFAULT_RISK_MAP
        assert "credential_access" in DEFAULT_RISK_MAP
        assert "unknown" in DEFAULT_RISK_MAP


# ── Security ──


class TestMaskKey:
    def test_short_key(self):
        assert mask_key("abc") == "***"

    def test_empty_key(self):
        assert mask_key("") == "***"

    def test_none_key(self):
        assert mask_key("") == "***"

    def test_normal_key(self):
        result = mask_key("sk-abc123456789")
        assert result == "***6789"
        assert "sk-abc" not in result

    def test_exactly_8_chars(self):
        result = mask_key("12345678")
        assert result == "***5678"


class TestSanitizeAgentName:
    def test_normal_name(self):
        assert sanitize_agent_name("my-agent") == "my-agent"

    def test_dots_underscores(self):
        assert sanitize_agent_name("agent.v2_test") == "agent.v2_test"

    def test_empty_returns_default(self):
        assert sanitize_agent_name("") == "default"

    def test_none_returns_default(self):
        assert sanitize_agent_name(None) == "default"

    def test_strips_control_chars(self):
        result = sanitize_agent_name("agent\r\n\t-name")
        assert "\r" not in result
        assert "\n" not in result
        assert "\t" not in result

    def test_truncates_long_name(self):
        long_name = "a" * 200
        result = sanitize_agent_name(long_name)
        assert len(result) <= 100

    def test_replaces_special_chars(self):
        result = sanitize_agent_name("agent/../../etc/passwd")
        assert "/" not in result
        assert " " not in result

    def test_rejects_header_injection(self):
        result = sanitize_agent_name("evil\r\nBcc: victim@evil.com")
        assert "\r" not in result
        assert "\n" not in result


class TestConfigRepr:
    def test_repr_masks_key(self):
        cfg = UmbraConfig(api_key="sk-secret-key-value-123")
        r = repr(cfg)
        assert "sk-secret" not in r
        assert "***" in r

    def test_repr_contains_port(self):
        cfg = UmbraConfig(port=9000)
        assert "9000" in repr(cfg)

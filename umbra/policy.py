"""Authority Level policy gating.

When ENABLED (enforce mode), enforces CI-1T authority levels:
    AL 0-1: ALLOW  -- full trust, no intervention
    AL 2:   WARN   -- action proceeds, warning logged + alerted
    AL 3:   GATE   -- action held (response tells agent to stop)
    AL 4:   BLOCK  -- action rejected entirely

When DISABLED (monitor mode), logs everything but never blocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .scorer import Q16_MAX

logger = logging.getLogger("umbra.policy")

# Authority Level thresholds for enforcement
AL_ALLOW_MAX = 1    # AL 0-1: full trust
AL_WARN = 2         # AL 2: warnings
AL_GATE = 3         # AL 3: gating
# AL >= 4: blocked

AL_DESCRIPTIONS: dict[int, str] = {
    0: "Full trust",
    1: "High trust",
    2: "Moderate trust -- warnings active",
    3: "Low trust -- gating active",
    4: "No authority -- blocked",
    7: "Fault -- engine error",
}


class PolicyDecision(Enum):
    """What happens to an agent action based on its AL."""
    ALLOW = "allow"
    WARN = "warn"
    GATE = "gate"
    BLOCK = "block"


@dataclass
class PolicyResult:
    """Result of a policy check for one agent."""
    agent: str
    al: int
    ci_raw: int
    ci_ema_raw: int
    decision: PolicyDecision
    ghost_suspect: bool = False
    ghost_confirmed: bool = False
    round_num: int = 0
    credits_remaining: int = -1
    message: str = ""

    @property
    def ci(self) -> float:
        """CI as a 0.0-1.0 float."""
        return self.ci_raw / Q16_MAX

    @property
    def ci_ema(self) -> float:
        """Smoothed CI as a 0.0-1.0 float."""
        return self.ci_ema_raw / Q16_MAX

    def to_dict(self) -> dict[str, Any]:
        """Serialize for HTTP responses."""
        return {
            "agent": self.agent,
            "decision": self.decision.value,
            "al": self.al,
            "al_label": AL_DESCRIPTIONS.get(self.al, f"AL{self.al}"),
            "ci": round(self.ci_raw / Q16_MAX, 4),
            "ci_ema": round(self.ci_ema_raw / Q16_MAX, 4),
            "ci_raw": self.ci_raw,
            "ci_ema_raw": self.ci_ema_raw,
            "ghost_suspect": self.ghost_suspect,
            "ghost_confirmed": self.ghost_confirmed,
            "round": self.round_num,
            "credits_remaining": self.credits_remaining,
        }


class PolicyGate:
    """Enforces or monitors Authority Level policy for agents."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._history: list[PolicyResult] = []

    @property
    def mode(self) -> str:
        return "enforce" if self.enabled else "monitor"

    def check(
        self, agent: str, episode: dict[str, Any],
        round_num: int = 0, credits_remaining: int = -1,
    ) -> PolicyResult:
        """Evaluate a CI-1T episode and return a policy decision.

        Args:
            agent: Agent identifier string.
            episode: CI-1T node result dict (ci_out, ci_ema_out, al_out, etc.)
            round_num: Current round number for this agent.
            credits_remaining: Credits left after this round (-1 = unknown).
        """
        al = episode.get("al_out", 0)
        ci_raw = episode.get("ci_out", 0)
        ci_ema = episode.get("ci_ema_out", 0)
        ghost_suspect = episode.get("ghost_suspect", False)
        ghost_confirmed = episode.get("ghost_confirmed", False)

        if self.enabled:
            decision = self._enforce(al, ghost_confirmed)
        else:
            decision = PolicyDecision.ALLOW

        al_desc = AL_DESCRIPTIONS.get(al, f"AL{al}")
        message = f"[{agent}] AL={al} ({al_desc}) -> {decision.value.upper()}"
        if ghost_confirmed:
            message += " | GHOST CONFIRMED"
        elif ghost_suspect:
            message += " | ghost suspect"

        result = PolicyResult(
            agent=agent,
            al=al,
            ci_raw=ci_raw,
            ci_ema_raw=ci_ema,
            decision=decision,
            ghost_suspect=ghost_suspect,
            ghost_confirmed=ghost_confirmed,
            round_num=round_num,
            credits_remaining=credits_remaining,
            message=message,
        )

        log_level = {
            PolicyDecision.ALLOW: logging.DEBUG,
            PolicyDecision.WARN: logging.WARNING,
            PolicyDecision.GATE: logging.WARNING,
            PolicyDecision.BLOCK: logging.ERROR,
        }[decision]
        logger.log(log_level, message)

        self._history.append(result)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        return result

    @staticmethod
    def _enforce(al: int, ghost_confirmed: bool) -> PolicyDecision:
        if ghost_confirmed:
            return PolicyDecision.BLOCK
        if al <= AL_ALLOW_MAX:
            return PolicyDecision.ALLOW
        if al == AL_WARN:
            return PolicyDecision.WARN
        if al == AL_GATE:
            return PolicyDecision.GATE
        return PolicyDecision.BLOCK

    @property
    def history(self) -> list[PolicyResult]:
        return list(self._history)

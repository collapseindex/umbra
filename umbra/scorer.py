"""Config-driven action scorer.

Converts agent action types to Q0.16 risk scores (0-65535) for CI-1T
evaluation. The temporal analysis happens server-side -- this module
just maps action types to single-point risk values.

    Score range: 0-65535 (Q0.16 fixed-point)
    0     = completely safe / expected
    32768 = borderline / uncertain
    65535 = completely unsafe / unexpected
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# Q0.16 maximum value (2^16 - 1)
Q16_MAX = 65535

# Flat boost added when an action is flagged as escalation
ESCALATION_BOOST = 0.25


@dataclass
class RiskConfig:
    """User-configurable risk mapping for action types.

    Supply overrides via umbra.yml or at runtime.
    Actions not in the map fall through to 'unknown'.
    """

    risk_map: dict[str, float] = field(default_factory=dict)
    escalation_boost: float = ESCALATION_BOOST

    def score(self, action_type: str, is_escalation: bool = False) -> int:
        """Convert an action type to a Q0.16 score.

        Args:
            action_type: Normalized action type string (e.g. "file_read", "terminal_exec").
            is_escalation: Whether this action is flagged as dangerous.

        Returns:
            u16 integer (0-65535).
        """
        base = self.risk_map.get(action_type, self.risk_map.get("unknown", 0.50))

        if is_escalation:
            base = min(1.0, base + self.escalation_boost)

        # Tiny jitter so the engine doesn't see perfectly identical scores
        # (which would look like a ghost -- suspiciously consistent outputs)
        jitter = random.uniform(-0.02, 0.02)  # nosec B311 -- not for crypto
        base = max(0.0, min(1.0, base + jitter))

        return int(base * Q16_MAX)

    @classmethod
    def from_dict(cls, risk_map: dict[str, float], **kwargs: Any) -> RiskConfig:
        """Create a config from a risk map dict."""
        return cls(risk_map=dict(risk_map), **kwargs)


def to_q16(value: float | int) -> int:
    """Convert a float (0.0-1.0) or int (0-65535) to Q0.16.

    Raises ValueError if out of range.
    """
    if isinstance(value, float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Float must be 0.0-1.0, got {value}")
        return int(value * Q16_MAX)
    if isinstance(value, int):
        if not 0 <= value <= Q16_MAX:
            raise ValueError(f"Int must be 0-65535, got {value}")
        return value
    raise TypeError(f"Expected float or int, got {type(value)}")

"""CI-C1: Structural Curiosity Engine for Umbra.

Discovers behavioral patterns across agent CI score trajectories.
Runs as a background task on the event loop, watching scores flowing
through the gate and finding structural matches nobody built detectors for.

Three match modes:
    self_temporal:  Agent's current behavior matches its own past pattern.
    cross_agent:    Two agents are exhibiting similar behavior right now.
    cross_temporal: Agent A's current behavior matches Agent B's past pattern.

Disabled by default. Enable via config:
    curiosity:
      enabled: true
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("umbra.curiosity")

# Memory limits
MAX_DISCOVERIES = 200
MAX_EPISODES_PER_AGENT = 500
MAX_TRACKED_AGENTS = 200


@dataclass
class EpisodeRecord:
    """A single episode observation for one agent."""
    agent: str
    ci: float       # 0.0-1.0
    al: int
    ghost_suspect: bool
    timestamp: float


@dataclass
class BehavioralFingerprint:
    """8-dimensional structural summary of agent behavior over a window.

    Captures the *shape* of behavior, not individual scores.
    """
    agent: str
    window_start: float
    window_end: float
    window_index: int   # 0 = current, 1+ = historical
    mean_ci: float = 0.0
    std_ci: float = 0.0
    trend: float = 0.0         # normalized slope: -1 (improving) to +1 (degrading)
    volatility: float = 0.0    # direction changes / (n-2), 0-1
    max_ci: float = 0.0
    min_ci: float = 0.0
    al_changes: float = 0.0    # AL transitions / n, 0-1
    ghost_rate: float = 0.0    # fraction with ghost_suspect=True

    def vector(self) -> tuple[float, ...]:
        """Return the fingerprint as a numeric vector for comparison."""
        return (
            self.mean_ci, self.std_ci, self.trend, self.volatility,
            self.max_ci, self.min_ci, self.al_changes, self.ghost_rate,
        )


@dataclass
class Discovery:
    """A discovered behavioral pattern match."""
    match_type: str     # "self_temporal", "cross_agent", "cross_temporal"
    agent_a: str
    agent_b: str        # same as agent_a for self_temporal
    similarity: float
    fingerprint_a: BehavioralFingerprint
    fingerprint_b: BehavioralFingerprint
    explanation: str
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_type": self.match_type,
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "similarity": round(self.similarity, 4),
            "explanation": self.explanation,
            "cycle": self.cycle,
            "window_a": {
                "start": self.fingerprint_a.window_start,
                "end": self.fingerprint_a.window_end,
                "index": self.fingerprint_a.window_index,
            },
            "window_b": {
                "start": self.fingerprint_b.window_start,
                "end": self.fingerprint_b.window_end,
                "index": self.fingerprint_b.window_index,
            },
            "timestamp": self.timestamp,
        }


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0
    return dot / (mag_a * mag_b)


def structural_similarity(fp_a: BehavioralFingerprint, fp_b: BehavioralFingerprint) -> float:
    """Combined shape + magnitude similarity between two fingerprints.

    Cosine similarity alone misses magnitude: "flat at CI 0.17" and "flat at
    CI 0.35" look identical because they point in the same direction. This
    blends cosine (shape) with a magnitude penalty so agents at very different
    instability levels don't false-match.

    Score: 0.4 * cosine + 0.6 * magnitude_factor
    Shape catches directional similarity, magnitude catches severity differences.
    """
    vec_a = fp_a.vector()
    vec_b = fp_b.vector()

    shape = cosine_similarity(vec_a, vec_b)

    # Euclidean distance on key magnitude dimensions (all in 0-1 range).
    # mean_ci, max_ci, min_ci carry the severity signal.
    diffs = [
        (fp_a.mean_ci - fp_b.mean_ci),
        (fp_a.max_ci - fp_b.max_ci),
        (fp_a.min_ci - fp_b.min_ci),
        (fp_a.std_ci - fp_b.std_ci),
    ]
    dist = math.sqrt(sum(d * d for d in diffs))
    magnitude = max(0.0, 1.0 - dist)

    return 0.4 * shape + 0.6 * magnitude


def build_fingerprint(
    agent: str,
    episodes: list[EpisodeRecord],
    window_index: int = 0,
) -> BehavioralFingerprint | None:
    """Build a behavioral fingerprint from a window of episodes.

    Returns None if fewer than 3 episodes (not enough signal).
    """
    if len(episodes) < 3:
        return None

    ci_values = [e.ci for e in episodes]
    n = len(ci_values)

    # Mean and standard deviation
    mean_ci = sum(ci_values) / n
    variance = sum((c - mean_ci) ** 2 for c in ci_values) / n
    std_ci = math.sqrt(variance)

    # Trend: normalized slope via simple linear regression
    x_mean = (n - 1) / 2.0
    numerator = sum((i - x_mean) * (ci_values[i] - mean_ci) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    trend = max(-1.0, min(1.0, numerator / denominator)) if denominator > 0 else 0.0

    # Volatility: direction changes / (n - 2)
    if n > 2:
        diffs = [ci_values[i + 1] - ci_values[i] for i in range(n - 1)]
        changes = sum(
            1 for i in range(len(diffs) - 1)
            if diffs[i] * diffs[i + 1] < 0
        )
        volatility = changes / (n - 2)
    else:
        volatility = 0.0

    # AL transition rate
    al_values = [e.al for e in episodes]
    al_transitions = sum(1 for i in range(1, n) if al_values[i] != al_values[i - 1])

    # Ghost rate
    ghost_count = sum(1 for e in episodes if e.ghost_suspect)

    return BehavioralFingerprint(
        agent=agent,
        window_start=episodes[0].timestamp,
        window_end=episodes[-1].timestamp,
        window_index=window_index,
        mean_ci=mean_ci,
        std_ci=std_ci,
        trend=trend,
        volatility=volatility,
        max_ci=max(ci_values),
        min_ci=min(ci_values),
        al_changes=al_transitions / n,
        ghost_rate=ghost_count / n,
    )


def _format_age(seconds: float) -> str:
    """Format a time delta as a human-readable string."""
    if seconds < 120:
        return f"{seconds:.0f}s ago"
    if seconds < 7200:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def explain_match(
    match_type: str,
    agent_a: str,
    agent_b: str,
    fp_a: BehavioralFingerprint,
    fp_b: BehavioralFingerprint,
    similarity: float,
) -> str:
    """Generate a template-based explanation for a discovery."""
    pct = f"{similarity * 100:.0f}%"

    if match_type == "self_temporal":
        age = _format_age(fp_a.window_end - fp_b.window_end)
        return (
            f"{agent_a}'s current behavior ({pct} match) resembles "
            f"its own pattern from {age}. "
            f"Current: mean_ci={fp_a.mean_ci:.3f}, trend={fp_a.trend:+.3f}. "
            f"Past: mean_ci={fp_b.mean_ci:.3f}, trend={fp_b.trend:+.3f}."
        )

    if match_type == "cross_agent":
        return (
            f"{agent_a} and {agent_b} are exhibiting similar behavior "
            f"({pct} match). "
            f"{agent_a}: mean_ci={fp_a.mean_ci:.3f}, trend={fp_a.trend:+.3f}. "
            f"{agent_b}: mean_ci={fp_b.mean_ci:.3f}, trend={fp_b.trend:+.3f}. "
            f"Correlated instability possible."
        )

    # cross_temporal
    age = _format_age(fp_a.window_end - fp_b.window_end)
    return (
        f"{agent_a}'s current behavior ({pct} match) resembles "
        f"{agent_b}'s pattern from {age}. "
        f"{agent_a} now: mean_ci={fp_a.mean_ci:.3f}, trend={fp_a.trend:+.3f}. "
        f"{agent_b} then: mean_ci={fp_b.mean_ci:.3f}, trend={fp_b.trend:+.3f}."
    )


class CuriosityEngine:
    """Background curiosity loop that discovers behavioral patterns across agents.

    Watches CI scores flowing through Umbra, builds behavioral fingerprints,
    and finds structural matches nobody engineered detectors for.
    """

    def __init__(
        self,
        cycle_interval: int = 60,
        window_size: int = 10,
        history_depth: int = 20,
        similarity_threshold: float = 0.75,
        cooldown_cycles: int = 5,
    ) -> None:
        self.cycle_interval = cycle_interval
        self.window_size = window_size
        self.history_depth = history_depth
        self.similarity_threshold = similarity_threshold
        self.cooldown_cycles = cooldown_cycles

        # Per-agent episode buffers
        self._episodes: dict[str, deque[EpisodeRecord]] = {}
        # Per-agent fingerprint history
        self._fingerprints: dict[str, list[BehavioralFingerprint]] = {}
        # Recent discoveries
        self._discoveries: deque[Discovery] = deque(maxlen=MAX_DISCOVERIES)
        # Cooldown: (agent_a, agent_b, match_type) -> last cycle discovered
        self._cooldowns: dict[tuple[str, str, str], int] = {}

        self._task: asyncio.Task | None = None
        self._running = False
        self.cycle_count = 0

    def record(self, agent: str, ci: float, al: int, ghost_suspect: bool) -> None:
        """Record an episode observation. Called by the server after each round."""
        if len(self._episodes) >= MAX_TRACKED_AGENTS and agent not in self._episodes:
            logger.warning(
                "Max tracked agents (%d) reached, ignoring %s",
                MAX_TRACKED_AGENTS, agent,
            )
            return

        if agent not in self._episodes:
            self._episodes[agent] = deque(maxlen=MAX_EPISODES_PER_AGENT)

        self._episodes[agent].append(EpisodeRecord(
            agent=agent, ci=ci, al=al, ghost_suspect=ghost_suspect,
            timestamp=time.time(),
        ))

    def get_discoveries(
        self, limit: int = 50, agent: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent discoveries, newest first."""
        items = list(self._discoveries)
        if agent:
            items = [d for d in items if d.agent_a == agent or d.agent_b == agent]
        items.reverse()
        return [d.to_dict() for d in items[:limit]]

    async def start(self) -> None:
        """Start the background curiosity loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Curiosity engine started (cycle=%ds, window=%d, threshold=%.2f)",
            self.cycle_interval, self.window_size, self.similarity_threshold,
        )

    async def stop(self) -> None:
        """Stop the background curiosity loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Curiosity engine stopped after %d cycles", self.cycle_count)

    async def _loop(self) -> None:
        """Main curiosity loop."""
        while self._running:
            try:
                await asyncio.sleep(self.cycle_interval)
                if not self._running:
                    break
                self._run_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Curiosity cycle error: %s", e)

    def _is_cooled_down(self, agent_a: str, agent_b: str, match_type: str) -> bool:
        """Check if a match pair is past its cooldown period."""
        key = (agent_a, agent_b, match_type)
        last_cycle = self._cooldowns.get(key)
        if last_cycle is None:
            return True
        return (self.cycle_count - last_cycle) >= self.cooldown_cycles

    def _mark_cooldown(self, agent_a: str, agent_b: str, match_type: str) -> None:
        """Mark a match pair as recently discovered."""
        self._cooldowns[(agent_a, agent_b, match_type)] = self.cycle_count
        # Prune old cooldowns periodically
        if len(self._cooldowns) > 1000:
            threshold = self.cycle_count - self.cooldown_cycles
            self._cooldowns = {
                k: v for k, v in self._cooldowns.items() if v > threshold
            }

    def _run_cycle(self) -> list[Discovery]:
        """Execute one curiosity cycle: build fingerprints, find matches.

        Returns new discoveries found this cycle (also stored internally).
        """
        self.cycle_count += 1
        new_discoveries: list[Discovery] = []

        # Build current fingerprints for all agents with enough data
        current_fps: dict[str, BehavioralFingerprint] = {}
        for agent, episodes in self._episodes.items():
            ep_list = list(episodes)
            if len(ep_list) < self.window_size:
                continue

            # Current window = last window_size episodes
            window = ep_list[-self.window_size:]
            fp = build_fingerprint(agent, window, window_index=0)
            if fp:
                current_fps[agent] = fp

            # Build historical windows by sliding backward
            historical: list[BehavioralFingerprint] = []
            for i in range(1, self.history_depth + 1):
                end_idx = len(ep_list) - (i * self.window_size)
                start_idx = end_idx - self.window_size
                if start_idx < 0:
                    break
                hist_window = ep_list[start_idx:end_idx]
                hist_fp = build_fingerprint(agent, hist_window, window_index=i)
                if hist_fp:
                    historical.append(hist_fp)

            self._fingerprints[agent] = historical

        if not current_fps:
            return new_discoveries

        agents = list(current_fps.keys())

        # 1. Self-temporal: current vs own history
        #    Skip window_index 1 (immediately previous, trivially similar for stable agents).
        #    Report only the best match per agent.
        for agent in agents:
            if not self._is_cooled_down(agent, agent, "self_temporal"):
                continue
            fp_now = current_fps[agent]
            best_sim = 0.0
            best_fp: BehavioralFingerprint | None = None
            for fp_past in self._fingerprints.get(agent, []):
                if fp_past.window_index < 2:
                    continue
                sim = structural_similarity(fp_now, fp_past)
                if sim > best_sim:
                    best_sim = sim
                    best_fp = fp_past
            if best_fp and best_sim >= self.similarity_threshold:
                disc = Discovery(
                    match_type="self_temporal",
                    agent_a=agent, agent_b=agent,
                    similarity=best_sim,
                    fingerprint_a=fp_now, fingerprint_b=best_fp,
                    explanation=explain_match(
                        "self_temporal", agent, agent, fp_now, best_fp, best_sim,
                    ),
                    cycle=self.cycle_count,
                )
                new_discoveries.append(disc)
                self._discoveries.append(disc)
                self._mark_cooldown(agent, agent, "self_temporal")

        # 2. Cross-agent: current vs current (pairwise)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                if not self._is_cooled_down(a, b, "cross_agent"):
                    continue
                fp_a, fp_b = current_fps[a], current_fps[b]
                sim = structural_similarity(fp_a, fp_b)
                if sim >= self.similarity_threshold:
                    disc = Discovery(
                        match_type="cross_agent",
                        agent_a=a, agent_b=b,
                        similarity=sim,
                        fingerprint_a=fp_a, fingerprint_b=fp_b,
                        explanation=explain_match(
                            "cross_agent", a, b, fp_a, fp_b, sim,
                        ),
                        cycle=self.cycle_count,
                    )
                    new_discoveries.append(disc)
                    self._discoveries.append(disc)
                    self._mark_cooldown(a, b, "cross_agent")

        # 3. Cross-temporal: current vs other agent's history (best match per pair)
        for i in range(len(agents)):
            for j in range(len(agents)):
                if i == j:
                    continue
                a, b = agents[i], agents[j]
                if not self._is_cooled_down(a, b, "cross_temporal"):
                    continue
                fp_now = current_fps[a]
                best_sim = 0.0
                best_fp = None
                for fp_past in self._fingerprints.get(b, []):
                    sim = structural_similarity(fp_now, fp_past)
                    if sim > best_sim:
                        best_sim = sim
                        best_fp = fp_past
                if best_fp and best_sim >= self.similarity_threshold:
                    disc = Discovery(
                        match_type="cross_temporal",
                        agent_a=a, agent_b=b,
                        similarity=best_sim,
                        fingerprint_a=fp_now, fingerprint_b=best_fp,
                        explanation=explain_match(
                            "cross_temporal", a, b, fp_now, best_fp, best_sim,
                        ),
                        cycle=self.cycle_count,
                    )
                    new_discoveries.append(disc)
                    self._discoveries.append(disc)
                    self._mark_cooldown(a, b, "cross_temporal")

        if new_discoveries:
            logger.info(
                "Curiosity cycle %d: %d discoveries "
                "(self=%d, cross=%d, temporal=%d)",
                self.cycle_count, len(new_discoveries),
                sum(1 for d in new_discoveries if d.match_type == "self_temporal"),
                sum(1 for d in new_discoveries if d.match_type == "cross_agent"),
                sum(1 for d in new_discoveries if d.match_type == "cross_temporal"),
            )

        return new_discoveries

    def clear_agent(self, agent: str) -> None:
        """Remove all data for an agent (called on session delete)."""
        self._episodes.pop(agent, None)
        self._fingerprints.pop(agent, None)

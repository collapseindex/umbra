"""Multi-agent coordination: influence gating, cascade propagation, consensus.

Adapted from CI-M1 multi-agent governance for Umbra's HTTP proxy model.
Three features:

1. Influence Gating:
   Agent A at AL3 cannot trigger actions that require AL1 through Agent B.
   Authority does not transfer upward.

2. Cascade CI Propagation:
   When Agent C goes unstable and was triggered by Agent A -> B -> C,
   a penalty score is injected back into Agent A's and B's buffers.
   Attenuated by hop distance: 50% per hop, max 4 hops.

3. Consensus Authority:
   When multiple agents contribute to a decision, the effective authority
   is the minimum AL of all contributors. One unstable agent drags down
   the group.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .scorer import Q16_MAX

logger = logging.getLogger("umbra.multi")

# Cascade propagation attenuation per hop (0.5 = 50% per hop)
CASCADE_DECAY = 0.5

# Maximum hops for cascade propagation
CASCADE_MAX_HOPS = 4

# Maximum causal edges tracked per agent (prevent memory bloat)
MAX_CAUSAL_EDGES = 200

# How long causal edges stay alive (seconds). 10 minutes default.
CAUSAL_EDGE_TTL = 600

# CI threshold above which cascade propagation fires.
# Only propagate when instability is meaningful (> 0.3 on 0-1 scale).
CASCADE_CI_THRESHOLD = int(0.3 * Q16_MAX)


@dataclass
class CausalEdge:
    """A record that agent_from triggered an action in agent_to."""
    agent_from: str
    agent_to: str
    timestamp: float = field(default_factory=time.time)


class CausalGraph:
    """Lightweight in-memory causal graph tracking which agents triggered which.

    Agents declare causality via the `triggered_by` field on /check.
    Edges have a TTL and are pruned on access.
    """

    def __init__(self, edge_ttl: float = CAUSAL_EDGE_TTL) -> None:
        self.edge_ttl = edge_ttl
        # agent_to -> list of CausalEdge (who triggered actions in this agent)
        self._edges: dict[str, deque[CausalEdge]] = {}

    def record(self, agent_from: str, agent_to: str) -> None:
        """Record that agent_from triggered an action in agent_to."""
        if agent_from == agent_to:
            return
        if agent_to not in self._edges:
            self._edges[agent_to] = deque(maxlen=MAX_CAUSAL_EDGES)
        self._edges[agent_to].append(
            CausalEdge(agent_from=agent_from, agent_to=agent_to)
        )

    def get_upstream(self, agent: str, max_hops: int = CASCADE_MAX_HOPS) -> list[tuple[str, int]]:
        """Get all upstream agents that (directly or indirectly) triggered this agent.

        Returns list of (agent_name, hop_distance) pairs. Hop 1 = direct trigger.
        Traverses the causal graph breadth-first, pruning expired edges.
        """
        now = time.time()
        visited: set[str] = {agent}
        result: list[tuple[str, int]] = []
        frontier: list[tuple[str, int]] = [(agent, 0)]

        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_hops:
                continue

            edges = self._edges.get(current)
            if not edges:
                continue

            # Prune expired edges in-place
            while edges and (now - edges[0].timestamp) > self.edge_ttl:
                edges.popleft()

            for edge in edges:
                if edge.agent_from not in visited:
                    visited.add(edge.agent_from)
                    hop = depth + 1
                    result.append((edge.agent_from, hop))
                    frontier.append((edge.agent_from, hop))

        return result

    def clear_agent(self, agent: str) -> None:
        """Remove all causal edges involving an agent (called on session delete)."""
        self._edges.pop(agent, None)
        for target in list(self._edges):
            edges = self._edges[target]
            # Remove edges from this agent
            self._edges[target] = deque(
                (e for e in edges if e.agent_from != agent),
                maxlen=MAX_CAUSAL_EDGES,
            )


class InfluenceGate:
    """Checks whether a triggering agent has sufficient authority to initiate
    an action in another agent.

    Rule: An agent can only trigger actions in other agents up to its own
    authority level. Authority doesn't transfer upward.
    """

    @staticmethod
    def check(
        triggering_al: int,
        target_action_risk: float,
        policy_enabled: bool,
    ) -> tuple[bool, str]:
        """Check if the triggering agent's AL permits this action.

        Args:
            triggering_al: Current AL of the agent that initiated the action.
            target_action_risk: Risk score (0.0-1.0) of the action being attempted.
            policy_enabled: Whether enforce mode is on.

        Returns:
            (allowed, reason) tuple.
        """
        if not policy_enabled:
            return True, "monitor mode"

        # Map risk thresholds to required AL.
        # High-risk actions (>0.6) need AL 0-1 (high trust).
        # Medium-risk (0.3-0.6) need AL 0-2.
        # Low-risk (<0.3) allowed at any AL.
        if target_action_risk > 0.6 and triggering_al > 1:
            return False, (
                f"Triggering agent at AL{triggering_al} cannot initiate "
                f"high-risk action (risk={target_action_risk:.2f}, requires AL0-1)"
            )
        if target_action_risk > 0.3 and triggering_al > 2:
            return False, (
                f"Triggering agent at AL{triggering_al} cannot initiate "
                f"medium-risk action (risk={target_action_risk:.2f}, requires AL0-2)"
            )
        return True, "allowed"


def compute_cascade_penalties(
    ci_raw: int,
    upstream: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Compute cascade penalty scores to inject into upstream agents' buffers.

    When an agent goes unstable (CI above threshold), a fraction of that
    instability propagates backward through the causal graph.

    Args:
        ci_raw: The unstable agent's CI score (Q0.16).
        upstream: List of (agent_name, hop_distance) from CausalGraph.get_upstream().

    Returns:
        List of (agent_name, penalty_q16) to inject.
    """
    if ci_raw < CASCADE_CI_THRESHOLD:
        return []

    penalties: list[tuple[str, int]] = []
    for agent, hops in upstream:
        # Attenuate by hop distance: 50% per hop
        attenuation = CASCADE_DECAY ** hops
        penalty = int(ci_raw * attenuation)
        penalty = min(penalty, Q16_MAX)
        if penalty > 0:
            penalties.append((agent, penalty))

    return penalties


def compute_consensus_al(agent_als: dict[str, int]) -> tuple[int, str]:
    """Compute the effective authority level for a group of agents.

    The weakest link determines the group's authority.

    Args:
        agent_als: Dict of {agent_name: current_al}.

    Returns:
        (effective_al, explanation) tuple.
    """
    if not agent_als:
        return 4, "no agents in group"

    effective = max(agent_als.values())  # Higher AL = less authority
    weakest = [name for name, al in agent_als.items() if al == effective]

    if len(weakest) == len(agent_als):
        explanation = f"all agents at AL{effective}"
    else:
        explanation = (
            f"AL{effective} (limited by {', '.join(weakest)})"
        )

    return effective, explanation

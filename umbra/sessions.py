"""Per-agent fleet session lifecycle manager.

Each agent that hits /check or /report gets its own CI-1T fleet session.
Sessions are created lazily on the first request and cleaned up on
DELETE /sessions/{agent} or server shutdown.

Score buffering happens here: actions are scored and buffered until
episode_size scores accumulate, then pushed as one fleet round.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import MAX_SESSIONS
from .scorer import Q16_MAX, RiskConfig

logger = logging.getLogger("umbra.sessions")

DEFAULT_TIMEOUT = 30.0


@dataclass
class AgentSession:
    """Runtime state for a single monitored agent."""
    agent: str
    session_id: str
    score_buffer: list[int] = field(default_factory=list)
    round_count: int = 0
    last_episode: dict[str, Any] = field(default_factory=dict)
    credits_remaining: int = -1
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class SessionManager:
    """Manages CI-1T fleet sessions for multiple agents.

    One fleet session per agent. Scores are buffered until episode_size
    scores accumulate, then pushed as a round to CI-1T.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        episode_size: int = 3,
        risk_config: RiskConfig | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.episode_size = episode_size
        self.risk_config = risk_config or RiskConfig()
        self.timeout = timeout
        self._sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a persistent httpx client (connection pooling)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    # -- Public API --

    async def check(
        self,
        agent: str,
        action: str,
        is_escalation: bool = False,
        score: int | None = None,
    ) -> dict[str, Any] | None:
        """Score an action and buffer it. Returns a result dict when a round fires.

        Args:
            agent: Agent identifier (creates session if new).
            action: Action type string (e.g. "file_read", "terminal_exec").
            is_escalation: Whether this action is flagged as dangerous.
            score: Optional raw Q0.16 score (0-65535). If provided, skips risk_config scoring.

        Returns:
            None if score was buffered (episode not full yet).
            CI-1T round result dict when episode_size scores accumulated and round was pushed.
        """
        session = await self._get_or_create(agent)
        session.last_active = time.time()

        # Score the action
        if score is not None:
            q16 = max(0, min(Q16_MAX, score))
        else:
            q16 = self.risk_config.score(action, is_escalation)

        session.score_buffer.append(q16)

        # Push when buffer reaches episode_size
        if len(session.score_buffer) >= self.episode_size:
            return await self._push_round(session)

        return None

    async def get_status(self, agent: str) -> dict[str, Any] | None:
        """Get current status for an agent. Returns None if no session exists."""
        session = self._sessions.get(agent)
        if not session:
            return None
        return {
            "agent": session.agent,
            "session_id": session.session_id,
            "round_count": session.round_count,
            "buffered_scores": len(session.score_buffer),
            "last_episode": session.last_episode,
            "credits_remaining": session.credits_remaining,
            "created_at": session.created_at,
            "last_active": session.last_active,
        }

    async def get_all_status(self) -> list[dict[str, Any]]:
        """Get status for all active agent sessions."""
        results = []
        for agent in list(self._sessions):
            status = await self.get_status(agent)
            if status:
                results.append(status)
        return results

    async def flush(self, agent: str) -> dict[str, Any] | None:
        """Force-push any buffered scores for an agent, even if < episode_size.

        Returns None if no scores buffered.
        """
        session = self._sessions.get(agent)
        if not session or not session.score_buffer:
            return None
        return await self._push_round(session)

    async def delete_session(self, agent: str) -> bool:
        """Delete a fleet session for an agent. Returns True if it existed."""
        session = self._sessions.pop(agent, None)
        if not session:
            return False

        try:
            client = await self._get_client()
            await client.delete(
                f"{self.api_url}/fleet/sessions/{session.session_id}",
            )
            logger.info("Deleted session for %s (id=%s)", agent, session.session_id)
        except httpx.HTTPStatusError as e:
            logger.warning("Failed to delete session for %s: HTTP %s", agent, e.response.status_code)
        except Exception as e:
            logger.warning("Failed to delete session for %s: %s", agent, e)

        return True

    async def shutdown(self) -> None:
        """Clean up all sessions and close the HTTP client. Call on server shutdown."""
        agents = list(self._sessions.keys())
        for agent in agents:
            await self.delete_session(agent)
        # Close persistent client
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        logger.info("All sessions cleaned up")

    # -- Internal --

    async def _get_or_create(self, agent: str) -> AgentSession:
        """Get existing session or create a new fleet session."""
        if agent in self._sessions:
            return self._sessions[agent]

        async with self._lock:
            # Double-check after acquiring lock
            if agent in self._sessions:
                return self._sessions[agent]

            # Enforce session limit to prevent unbounded memory growth
            if len(self._sessions) >= MAX_SESSIONS:
                raise ValueError(
                    f"Maximum session limit ({MAX_SESSIONS}) reached. "
                    f"Delete inactive sessions before creating new ones."
                )

            session_id = await self._create_fleet_session(agent)
            session = AgentSession(agent=agent, session_id=session_id)
            self._sessions[agent] = session
            logger.info("Created session for %s (id=%s)", agent, session_id)
            return session

    async def _create_fleet_session(self, agent: str) -> str:
        """Create a fleet session on the CI-1T API."""
        body: dict[str, Any] = {
            "node_count": 1,
            "node_names": [agent],
        }

        client = await self._get_client()
        resp = await client.post(
            f"{self.api_url}/fleet/sessions",
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["session_id"]

    async def _push_round(self, session: AgentSession) -> dict[str, Any]:
        """Push buffered scores as a fleet round."""
        scores = session.score_buffer[:self.episode_size]
        session.score_buffer = session.score_buffer[self.episode_size:]

        client = await self._get_client()
        resp = await client.post(
            f"{self.api_url}/fleet/sessions/{session.session_id}/round",
            json={"scores": [scores]},
        )
        resp.raise_for_status()
        data = resp.json()

        session.round_count += 1

        # Parse the response
        snapshot = data.get("snapshot", data)
        nodes = snapshot.get("nodes", [])
        if nodes:
            session.last_episode = nodes[0]

        # Track credits
        credits_remaining = data.get("credits_remaining", -1)
        if credits_remaining >= 0:
            session.credits_remaining = credits_remaining

        return data

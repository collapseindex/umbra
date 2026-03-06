"""Starlette HTTP server -- the core of Umbra.

Endpoints:
    POST /check     Score an action and return a policy decision (synchronous).
    POST /report    Score an action without blocking (fire-and-forget).
    GET  /status    Current CI/AL/ghost for all monitored agents.
    GET  /status/{agent}  Status for a single agent.
    GET  /health    Liveness check.
    DELETE /sessions/{agent}  Reset an agent's monitoring session.
    POST /consensus     Effective authority level for a group of agents.
    GET  /causal/{agent}  Upstream causal chain for an agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

import httpx

from . import __version__
from .alerts import AlertDispatcher
from .config import UmbraConfig, sanitize_agent_name
from .multi import (
    CausalGraph, InfluenceGate, compute_cascade_penalties, compute_consensus_al,
)
from .policy import PolicyGate, PolicyResult
from .scorer import Q16_MAX
from .sessions import SessionManager

logger = logging.getLogger("umbra.server")

# Maximum request body size (64KB -- actions are small)
MAX_BODY_SIZE = 65536

# Rate limiting: max requests per window
RATE_LIMIT_MAX = 200
RATE_LIMIT_WINDOW = 60  # seconds


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _validate_check_body(body: dict[str, Any]) -> str | None:
    """Validate a /check or /report request body. Returns error message or None."""
    if "action" not in body and "score" not in body:
        return "Request must include 'action' (string) or 'score' (int 0-65535)"

    if "action" in body:
        action = body["action"]
        if not isinstance(action, str) or len(action) > 200:
            return "'action' must be a string up to 200 chars"
        # Reject control characters in action
        if any(ord(c) < 32 for c in action):
            return "'action' must not contain control characters"

    if "score" in body:
        score = body["score"]
        if not isinstance(score, (int, float)) or score < 0 or score > Q16_MAX:
            return f"'score' must be an integer 0-{Q16_MAX}"

    agent = body.get("agent", "")
    if agent and (not isinstance(agent, str) or len(agent) > 100):
        return "'agent' must be a string up to 100 chars"

    # Validate triggered_by (optional, string)
    triggered_by = body.get("triggered_by")
    if triggered_by is not None:
        if not isinstance(triggered_by, str) or len(triggered_by) > 100:
            return "'triggered_by' must be a string up to 100 chars"
        if any(ord(c) < 32 for c in triggered_by):
            return "'triggered_by' must not contain control characters"

    return None


class UmbraServer:
    """HTTP server wrapping the session manager, policy gate, and alert dispatcher."""

    def __init__(self, config: UmbraConfig) -> None:
        self.config = config
        self.policy = PolicyGate(enabled=config.enforce)

        self.sessions = SessionManager(
            api_key=config.api_key,
            api_url=config.api_url,
            episode_size=config.episode_size,
            risk_config=None,  # set below
        )

        # Initialize risk config from loaded config
        from .scorer import RiskConfig
        self.sessions.risk_config = RiskConfig(risk_map=dict(config.risk_map))

        self.alerts = AlertDispatcher(config.alerts)
        self.multi_enabled = config.multi_agent.enabled
        self.causal_graph = CausalGraph(edge_ttl=config.multi_agent.causal_edge_ttl)
        self._multi_cfg = config.multi_agent
        self._start_time = time.time()
        self._request_count = 0
        self._credit_warning_sent = False
        self._rate_limiter: deque[float] = deque()
        self._decision_log: deque[dict[str, Any]] = deque(maxlen=500)
        self._episode_log: deque[dict[str, Any]] = deque(maxlen=500)

    def build_app(self) -> Starlette:
        """Build the Starlette ASGI application."""
        routes = [
            Route("/check", self._handle_check, methods=["POST"]),
            Route("/report", self._handle_report, methods=["POST"]),
            Route("/status", self._handle_status, methods=["GET"]),
            Route("/status/{agent:path}", self._handle_agent_status, methods=["GET"]),
            Route("/health", self._handle_health, methods=["GET"]),
            Route("/sessions/{agent:path}", self._handle_delete_session, methods=["DELETE"]),
            Route("/decisions", self._handle_decisions, methods=["GET"]),
            Route("/episodes", self._handle_episodes, methods=["GET"]),
            Route("/consensus", self._handle_consensus, methods=["POST"]),
            Route("/causal/{agent:path}", self._handle_causal, methods=["GET"]),
        ]

        app = Starlette(
            routes=routes,
            on_startup=[self._on_startup],
            on_shutdown=[self._on_shutdown],
        )
        return app

    # -- Lifecycle --

    async def _on_startup(self) -> None:
        logger.info(
            "umbra started (policy=%s, episode_size=%d, api=%s)",
            self.config.policy, self.config.episode_size, self.config.api_url,
        )

    async def _on_shutdown(self) -> None:
        logger.info("Shutting down, cleaning up sessions...")
        await self.sessions.shutdown()

    # -- POST /check --

    async def _handle_check(self, request: Request) -> Response:
        """Score an action synchronously and return the policy decision.

        This is the main endpoint agents call before executing an action.
        If the response is GATE or BLOCK, the agent should abort.
        """
        body = await self._parse_body(request)
        if isinstance(body, Response):
            return body

        err = _validate_check_body(body)
        if err:
            return _error(400, err)

        self._request_count += 1
        agent = sanitize_agent_name(body.get("agent", "default"))
        action = body.get("action", "unknown")
        is_escalation = bool(body.get("escalation", False))
        raw_score = body.get("score")
        triggered_by = body.get("triggered_by")
        if triggered_by:
            triggered_by = sanitize_agent_name(triggered_by)

        # Multi-agent: influence gating
        if triggered_by and self.multi_enabled and self._multi_cfg.influence_gating:
            trigger_status = await self.sessions.get_status(triggered_by)
            if trigger_status:
                trigger_ep = trigger_status.get("last_episode", {})
                trigger_al = trigger_ep.get("al_out", 4)
                risk = self.sessions.risk_config.risk_map.get(action, 0.50)
                allowed, reason = InfluenceGate.check(
                    triggering_al=trigger_al,
                    target_action_risk=risk,
                    policy_enabled=self.config.enforce,
                )
                if not allowed:
                    return JSONResponse({
                        "decision": "block",
                        "agent": agent,
                        "triggered_by": triggered_by,
                        "influence_gated": True,
                        "reason": reason,
                    })

        # Multi-agent: record causal edge
        if triggered_by and self.multi_enabled:
            self.causal_graph.record(agent_from=triggered_by, agent_to=agent)

        # Score + buffer + maybe push round
        try:
            round_result = await self.sessions.check(
                agent=agent,
                action=action,
                is_escalation=is_escalation,
                score=int(raw_score) if raw_score is not None else None,
            )
        except httpx.HTTPStatusError as e:
            logger.error("CI-1T API returned %s for %s", e.response.status_code, agent)
            return _error(502, "CI-1T API returned an error")
        except httpx.TimeoutException:
            logger.error("CI-1T API timeout for %s", agent)
            return _error(504, "CI-1T API timeout")
        except Exception as e:
            logger.error("Session check failed for %s: %s", agent, e)
            return _error(502, "CI-1T API error")

        # If no round was pushed yet (still buffering), return ALLOW
        if round_result is None:
            return JSONResponse({
                "decision": "allow",
                "buffered": True,
                "agent": agent,
                "message": "Score buffered, waiting for full episode",
            })

        # We got a round result -- apply policy
        result = self._apply_policy(agent, round_result)

        # Multi-agent: cascade propagation
        if self.multi_enabled and self._multi_cfg.cascade:
            asyncio.create_task(self._cascade_propagate(agent, result.ci_raw))

        # Log decision + episode
        result_dict = result.to_dict()
        self._decision_log.append({
            "agent": agent,
            "decision": result.decision.value,
            "al": result.al,
            "ghost_confirmed": result.ghost_confirmed,
            "round": result.round_num,
            "timestamp": time.time(),
        })
        self._episode_log.append({
            "agent": agent,
            "ci": result_dict["ci"],
            "ci_ema": result_dict["ci_ema"],
            "al": result.al,
            "ghost_suspect": result.ghost_suspect,
            "ghost_confirmed": result.ghost_confirmed,
            "round": result.round_num,
            "decision": result.decision.value,
            "timestamp": time.time(),
        })

        # Fire alerts in background (don't slow down the response)
        asyncio.create_task(self._maybe_alert(result))

        return JSONResponse(result_dict)

    # -- POST /report --

    async def _handle_report(self, request: Request) -> Response:
        """Score an action without blocking. Fire-and-forget for agents
        that don't want to wait for a policy decision.
        """
        body = await self._parse_body(request)
        if isinstance(body, Response):
            return body

        err = _validate_check_body(body)
        if err:
            return _error(400, err)

        self._request_count += 1
        agent = sanitize_agent_name(body.get("agent", "default"))
        action = body.get("action", "unknown")
        is_escalation = bool(body.get("escalation", False))
        raw_score = body.get("score")

        # Score in background
        async def _bg_check() -> None:
            try:
                round_result = await self.sessions.check(
                    agent=agent,
                    action=action,
                    is_escalation=is_escalation,
                    score=int(raw_score) if raw_score is not None else None,
                )
                if round_result is not None:
                    result = self._apply_policy(agent, round_result)
                    await self._maybe_alert(result)
            except Exception as e:
                logger.error("Background check failed for %s: %s", agent, e)

        asyncio.create_task(_bg_check())

        return JSONResponse({"accepted": True, "agent": agent})

    # -- GET /status --

    async def _handle_status(self, request: Request) -> Response:
        """Return status for all active agent sessions."""
        statuses = await self.sessions.get_all_status()

        # Enrich with policy info
        for s in statuses:
            ep = s.get("last_episode", {})
            if ep:
                al = ep.get("al_out", 0)
                s["al"] = al
                s["ci"] = round(ep.get("ci_out", 0) / Q16_MAX, 4)
                s["ghost_suspect"] = ep.get("ghost_suspect", False)
                s["ghost_confirmed"] = ep.get("ghost_confirmed", False)

        return JSONResponse({
            "policy": self.config.policy,
            "agents": statuses,
            "uptime_seconds": int(time.time() - self._start_time),
            "total_requests": self._request_count,
        })

    # -- GET /status/{agent} --

    async def _handle_agent_status(self, request: Request) -> Response:
        agent = request.path_params["agent"]
        status = await self.sessions.get_status(agent)
        if not status:
            return _error(404, f"No active session for agent '{agent}'")
        return JSONResponse(status)

    # -- GET /health --

    async def _handle_health(self, request: Request) -> Response:
        return JSONResponse({
            "status": "ok",
            "version": __version__,
            "policy": self.config.policy,
            "uptime_seconds": int(time.time() - self._start_time),
        })

    # -- DELETE /sessions/{agent} --

    async def _handle_delete_session(self, request: Request) -> Response:
        agent = request.path_params["agent"]
        deleted = await self.sessions.delete_session(agent)
        if not deleted:
            return _error(404, f"No active session for agent '{agent}'")
        self.causal_graph.clear_agent(agent)
        return JSONResponse({"deleted": True, "agent": agent})

    # -- GET /decisions --

    async def _handle_decisions(self, request: Request) -> Response:
        """Return recent policy decisions, optionally filtered by agent."""
        agent_filter = request.query_params.get("agent")
        items = list(self._decision_log)
        if agent_filter:
            items = [d for d in items if d["agent"] == agent_filter]
        # Return newest first
        items.reverse()
        return JSONResponse({"decisions": items[:100]})

    # -- GET /episodes --

    async def _handle_episodes(self, request: Request) -> Response:
        """Return recent episode results, optionally filtered by agent."""
        agent_filter = request.query_params.get("agent")
        items = list(self._episode_log)
        if agent_filter:
            items = [d for d in items if d["agent"] == agent_filter]
        items.reverse()
        return JSONResponse({"episodes": items[:100]})

    # -- POST /consensus --

    async def _handle_consensus(self, request: Request) -> Response:
        """Compute effective authority level for a group of agents.

        The weakest link determines the group's authority.
        """
        body = await self._parse_body(request)
        if isinstance(body, Response):
            return body

        agents = body.get("agents")
        if not agents or not isinstance(agents, list):
            return _error(400, "'agents' must be a non-empty list of agent names")
        if len(agents) > 100:
            return _error(400, "'agents' list too large (max 100)")

        agent_als: dict[str, int] = {}
        for name in agents:
            if not isinstance(name, str):
                continue
            name = sanitize_agent_name(name)
            status = await self.sessions.get_status(name)
            if status:
                ep = status.get("last_episode", {})
                agent_als[name] = ep.get("al_out", 4)
            else:
                # Unknown agent treated as AL4 (no trust)
                agent_als[name] = 4

        effective_al, explanation = compute_consensus_al(agent_als)

        return JSONResponse({
            "effective_al": effective_al,
            "explanation": explanation,
            "agents": {name: {"al": al} for name, al in agent_als.items()},
        })

    # -- GET /causal/{agent} --

    async def _handle_causal(self, request: Request) -> Response:
        """Return the upstream causal chain for an agent."""
        agent = request.path_params["agent"]
        upstream = self.causal_graph.get_upstream(agent)
        return JSONResponse({
            "agent": agent,
            "upstream": [
                {"agent": name, "hops": hops}
                for name, hops in upstream
            ],
        })

    # -- Internal helpers --

    async def _parse_body(self, request: Request) -> dict[str, Any] | Response:
        """Parse JSON body with size limit and rate limiting."""
        # Rule #13: Rate limiting
        if not self._check_rate_limit():
            return _error(429, "Rate limit exceeded. Try again later.")

        # Rule #6: Validate content-length before reading
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
            except (ValueError, OverflowError):
                return _error(400, "Invalid Content-Length header")
            if length > MAX_BODY_SIZE:
                return _error(413, "Request body too large")

        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _error(400, "Invalid JSON body")
        except Exception:
            return _error(400, "Could not parse request body")

        if not isinstance(body, dict):
            return _error(400, "Request body must be a JSON object")

        return body

    def _check_rate_limit(self) -> bool:
        """Sliding window rate limiter. Returns True if request is allowed."""
        now = time.time()
        # Remove entries outside window
        while self._rate_limiter and self._rate_limiter[0] < now - RATE_LIMIT_WINDOW:
            self._rate_limiter.popleft()
        if len(self._rate_limiter) >= RATE_LIMIT_MAX:
            return False
        self._rate_limiter.append(now)
        return True

    def _apply_policy(self, agent: str, round_result: dict[str, Any]) -> PolicyResult:
        """Extract the node episode from a round result and apply policy."""
        snapshot = round_result.get("snapshot", round_result)
        nodes = snapshot.get("nodes", [])
        episode = nodes[0] if nodes else {}
        round_num = round_result.get("round", 0)
        credits_remaining = round_result.get("credits_remaining", -1)

        result = self.policy.check(
            agent=agent,
            episode=episode,
            round_num=round_num,
            credits_remaining=credits_remaining,
        )

        return result

    async def _maybe_alert(self, result: PolicyResult) -> None:
        """Send alerts and credit warnings if needed."""
        await self.alerts.maybe_alert(result)

        # Credit warning check
        if (
            result.credits_remaining >= 0
            and result.credits_remaining <= self.config.credits.low_warning
            and not self._credit_warning_sent
        ):
            self._credit_warning_sent = True
            logger.warning(
                "Low credits: %d remaining (threshold: %d)",
                result.credits_remaining, self.config.credits.low_warning,
            )
            await self.alerts.send_credit_warning(result.credits_remaining, result.agent)

    async def _cascade_propagate(self, agent: str, ci_raw: int) -> None:
        """Propagate instability backward through the causal graph.

        When an agent goes unstable, a penalty score is injected into
        upstream agents' score buffers, attenuated by hop distance.
        """
        upstream = self.causal_graph.get_upstream(
            agent, max_hops=self._multi_cfg.cascade_max_hops
        )
        if not upstream:
            return

        penalties = compute_cascade_penalties(ci_raw, upstream)
        for target_agent, penalty_q16 in penalties:
            try:
                # Inject penalty score directly into the target's buffer
                await self.sessions.check(
                    agent=target_agent,
                    action="_cascade_penalty",
                    score=penalty_q16,
                )
                logger.info(
                    "Cascade penalty: %s -> %s (score=%d)",
                    agent, target_agent, penalty_q16,
                )
            except Exception as e:
                logger.warning(
                    "Failed to propagate cascade to %s: %s", target_agent, e
                )

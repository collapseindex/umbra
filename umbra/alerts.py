"""Multi-channel alert dispatcher (Slack, email, SMS).

All channels are opt-in. Nothing fires unless explicitly configured
in umbra.yml. Alerts are rate-limited to avoid spam.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import time
from email.message import EmailMessage
from typing import Any

import httpx

from .config import AlertsConfig
from .policy import PolicyDecision, PolicyResult
from .scorer import Q16_MAX

logger = logging.getLogger("umbra.alerts")

# Minimum seconds between alerts for the same agent
ALERT_COOLDOWN = 60

# AL level ordering for threshold comparison
AL_RANK: dict[str, int] = {
    "AL1": 1,
    "AL2": 2,
    "AL3": 3,
    "AL4": 4,
    "ghost": 10,
}


def _should_alert(result: PolicyResult, min_level: str) -> bool:
    """Check if this result meets the minimum alert threshold."""
    threshold = AL_RANK.get(min_level, 3)

    # Ghost always alerts if min_level is ghost or lower
    if result.ghost_confirmed and threshold <= AL_RANK["ghost"]:
        return True

    return result.al >= threshold


class AlertDispatcher:
    """Sends alerts to configured channels when policy thresholds are crossed."""

    def __init__(self, config: AlertsConfig) -> None:
        self.config = config
        self._last_alert: dict[str, float] = {}  # agent -> timestamp

    def _cooldown_ok(self, agent: str) -> bool:
        """Check if enough time has passed since the last alert for this agent."""
        last = self._last_alert.get(agent, 0.0)
        if time.time() - last < ALERT_COOLDOWN:
            return False
        self._last_alert[agent] = time.time()
        return True

    async def maybe_alert(self, result: PolicyResult) -> None:
        """Send alerts if the result meets the threshold and cooldown has passed."""
        if not _should_alert(result, self.config.min_level):
            return

        if not self._cooldown_ok(result.agent):
            logger.debug("Alert cooldown active for %s, skipping", result.agent)
            return

        tasks: list[Any] = []

        if self.config.slack.enabled:
            tasks.append(self._send_slack(result))

        if self.config.email.enabled:
            tasks.append(self._send_email(result))

        if self.config.sms.enabled:
            tasks.append(self._send_sms(result))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_credit_warning(self, credits_remaining: int, agent: str) -> None:
        """Send a low credit warning to all configured channels."""
        msg = (
            f"CI-1T credits low: {credits_remaining} remaining\n"
            f"Agent: {agent}\n"
            f"Top up at https://collapseindex.org/dashboard"
        )

        tasks: list[Any] = []

        if self.config.slack.enabled:
            tasks.append(self._post_slack(":warning: CI-1T Low Credits", msg))

        if self.config.email.enabled:
            tasks.append(self._post_email("CI-1T Low Credits Warning", msg))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -- Slack --

    async def _send_slack(self, result: PolicyResult) -> None:
        """Send a Slack notification for a policy event."""
        emoji = {
            PolicyDecision.WARN: ":warning:",
            PolicyDecision.GATE: ":octagonal_sign:",
            PolicyDecision.BLOCK: ":no_entry:",
        }.get(result.decision, ":information_source:")

        ci_pct = round(result.ci_raw / Q16_MAX * 100, 1)
        ghost_text = ""
        if result.ghost_confirmed:
            ghost_text = " :ghost: *GHOST CONFIRMED*"
        elif result.ghost_suspect:
            ghost_text = " :ghost: ghost suspect"

        text = (
            f"{emoji} *CI-1T Alert -- {result.decision.value.upper()}*\n"
            f"Agent: `{result.agent}`\n"
            f"AL: {result.al} | CI: {ci_pct}% | Round: {result.round_num}"
            f"{ghost_text}"
        )

        if result.credits_remaining >= 0:
            text += f"\nCredits remaining: {result.credits_remaining}"

        await self._post_slack(None, text)

    async def _post_slack(self, title: str | None, text: str) -> None:
        """Post a message to Slack via webhook."""
        payload: dict[str, Any] = {"text": text}
        if self.config.slack.channel:
            payload["channel"] = self.config.slack.channel

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.config.slack.webhook_url, json=payload)
                resp.raise_for_status()
            logger.info("Slack alert sent")
        except Exception as e:
            logger.error("Slack alert failed: %s", e)

    # -- Email --

    async def _send_email(self, result: PolicyResult) -> None:
        """Send an email notification for a policy event."""
        ci_pct = round(result.ci_raw / Q16_MAX * 100, 1)
        # Sanitize agent name in subject to prevent email header injection
        safe_agent = result.agent.replace("\r", "").replace("\n", "")
        subject = f"CI-1T Alert: {result.decision.value.upper()} -- {safe_agent}"
        body = (
            f"CI-1T Policy Alert\n"
            f"{'=' * 40}\n\n"
            f"Agent:    {result.agent}\n"
            f"Decision: {result.decision.value.upper()}\n"
            f"AL:       {result.al}\n"
            f"CI:       {ci_pct}%\n"
            f"Round:    {result.round_num}\n"
            f"Ghost:    {'CONFIRMED' if result.ghost_confirmed else 'suspect' if result.ghost_suspect else 'clean'}\n"
        )

        if result.credits_remaining >= 0:
            body += f"Credits:  {result.credits_remaining}\n"

        body += "\nDashboard: https://collapseindex.org/dashboard"

        await self._post_email(subject, body)

    async def _post_email(self, subject: str, body: str) -> None:
        """Send an email via SMTP (runs in thread to avoid blocking)."""
        ec = self.config.email

        def _send() -> None:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = ec.from_addr or ec.smtp_user
            msg["To"] = ", ".join(ec.to_addrs)
            msg.set_content(body)

            with smtplib.SMTP(ec.smtp_host, ec.smtp_port) as server:
                server.starttls()
                if ec.smtp_user and ec.smtp_pass:
                    server.login(ec.smtp_user, ec.smtp_pass)
                server.send_message(msg)

        try:
            await asyncio.get_event_loop().run_in_executor(None, _send)
            logger.info("Email alert sent to %s", ec.to_addrs)
        except Exception as e:
            logger.error("Email alert failed: %s", e)

    # -- SMS (Twilio) --

    async def _send_sms(self, result: PolicyResult) -> None:
        """Send SMS via Twilio API."""
        ci_pct = round(result.ci_raw / Q16_MAX * 100, 1)
        body = (
            f"CI-1T {result.decision.value.upper()}: "
            f"{result.agent} AL={result.al} CI={ci_pct}%"
        )
        if result.ghost_confirmed:
            body += " GHOST"

        sc = self.config.sms
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sc.account_sid}/Messages.json"

        for to_number in sc.to_numbers:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        url,
                        auth=(sc.account_sid, sc.auth_token),
                        data={
                            "From": sc.from_number,
                            "To": to_number,
                            "Body": body,
                        },
                    )
                    resp.raise_for_status()
                logger.info("SMS sent to %s", to_number)
            except Exception as e:
                logger.error("SMS to %s failed: %s", to_number, e)

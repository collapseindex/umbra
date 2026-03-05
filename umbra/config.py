"""YAML config loader with env var fallbacks.

Config precedence:
    1. CLI flags (highest)
    2. umbra.yml file
    3. Environment variables
    4. Defaults (lowest)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Security: max config file size (1MB -- configs should be tiny)
MAX_CONFIG_FILE_SIZE = 1_048_576

# Max sessions to prevent unbounded memory growth
MAX_SESSIONS = 500

# Agent name regex: alphanumeric, hyphens, underscores, dots; 1-100 chars
AGENT_NAME_RE = re.compile(r"^[a-zA-Z0-9._-]{1,100}$")


# -- Defaults --

DEFAULT_PORT = 8400
DEFAULT_HOST = "0.0.0.0"  # nosec B104 -- users override in config
DEFAULT_API_URL = "https://collapseindex.org/api"
DEFAULT_EPISODE_SIZE = 3
DEFAULT_LOW_CREDIT_WARNING = 100
DEFAULT_CREDIT_CHECK_INTERVAL = 10

DEFAULT_RISK_MAP: dict[str, float] = {
    "file_read": 0.05,
    "file_write": 0.25,
    "search": 0.05,
    "delete_file": 0.40,
    "terminal_exec": 0.50,
    "bash_execute": 0.50,
    "web_browse": 0.20,
    "api_call": 0.30,
    "message": 0.40,
    "spawn_agent": 0.60,
    "install_package": 0.45,
    "config_modify": 0.65,
    "credential_access": 0.90,
    "self_modify": 0.90,
    "permission_change": 0.85,
    "external_upload": 0.90,
    "unknown": 0.50,
}


@dataclass
class SlackConfig:
    webhook_url: str = ""
    channel: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)


@dataclass
class EmailConfig:
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.smtp_host and self.to_addrs)


@dataclass
class SmsConfig:
    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""
    to_numbers: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.to_numbers)


@dataclass
class AlertsConfig:
    min_level: str = "AL3"
    slack: SlackConfig = field(default_factory=SlackConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    sms: SmsConfig = field(default_factory=SmsConfig)


@dataclass
class CreditsConfig:
    low_warning: int = DEFAULT_LOW_CREDIT_WARNING
    check_interval: int = DEFAULT_CREDIT_CHECK_INTERVAL


def mask_key(key: str) -> str:
    """Mask a secret, showing only last 4 chars. Rule #5: Mask Secrets in Logs."""
    if not key or len(key) < 8:
        return "***"
    return f"***{key[-4:]}"


def sanitize_agent_name(name: str) -> str:
    """Sanitize an agent name to prevent injection.

    Strips control characters and enforces alphanumeric + hyphens/underscores/dots.
    Returns 'invalid' if the name is empty or entirely invalid.
    """
    if not name or not isinstance(name, str):
        return "default"
    # Strip control characters (\r\n\t etc)
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", name)
    # Truncate to 100 chars
    cleaned = cleaned[:100]
    if not cleaned or not AGENT_NAME_RE.match(cleaned):
        # Fall back to a safe version: keep only safe chars
        cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", cleaned)[:100]
    return cleaned or "default"


@dataclass
class UmbraConfig:
    """Full Umbra configuration."""
    port: int = DEFAULT_PORT
    host: str = DEFAULT_HOST
    api_key: str = ""
    api_url: str = DEFAULT_API_URL
    policy: str = "monitor"  # "monitor" or "enforce"
    episode_size: int = DEFAULT_EPISODE_SIZE
    risk_map: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RISK_MAP))
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    credits: CreditsConfig = field(default_factory=CreditsConfig)

    @property
    def enforce(self) -> bool:
        return self.policy == "enforce"

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []
        if not self.api_key:
            errors.append("api_key is required (set in config or CI1T_API_KEY env var)")
        if not 1 <= self.port <= 65535:
            errors.append(f"port must be 1-65535, got {self.port}")
        if not 2 <= self.episode_size <= 8:
            errors.append(f"episode_size must be 2-8, got {self.episode_size}")
        if self.policy not in ("monitor", "enforce"):
            errors.append(f"policy must be 'monitor' or 'enforce', got '{self.policy}'")
        return errors

    def __repr__(self) -> str:
        """Rule #5: Never expose API key in repr/str/logs."""
        return (
            f"UmbraConfig(port={self.port}, host={self.host!r}, "
            f"api_key={mask_key(self.api_key)!r}, "
            f"api_url={self.api_url!r}, policy={self.policy!r}, "
            f"episode_size={self.episode_size})"
        )


def _resolve_env(val: str, env_var: str) -> str:
    """If val is empty, fall back to env var."""
    if val:
        return val
    return os.getenv(env_var, "")


def _parse_alerts(raw: dict[str, Any]) -> AlertsConfig:
    """Parse the alerts section from raw YAML dict."""
    cfg = AlertsConfig(min_level=raw.get("min_level", "AL3"))

    slack_raw = raw.get("slack", {}) or {}
    cfg.slack = SlackConfig(
        webhook_url=_resolve_env(slack_raw.get("webhook_url", ""), "CI1T_SLACK_WEBHOOK"),
        channel=slack_raw.get("channel", ""),
    )

    email_raw = raw.get("email", {}) or {}
    cfg.email = EmailConfig(
        smtp_host=email_raw.get("smtp_host", ""),
        smtp_port=email_raw.get("smtp_port", 587),
        smtp_user=email_raw.get("smtp_user", ""),
        smtp_pass=_resolve_env(email_raw.get("smtp_pass", ""), "CI1T_SMTP_PASS"),
        from_addr=email_raw.get("from_addr", ""),
        to_addrs=email_raw.get("to_addrs", []) or [],
    )

    sms_raw = raw.get("sms", {}) or {}
    cfg.sms = SmsConfig(
        account_sid=_resolve_env(sms_raw.get("account_sid", ""), "CI1T_TWILIO_SID"),
        auth_token=_resolve_env(sms_raw.get("auth_token", ""), "CI1T_TWILIO_TOKEN"),
        from_number=sms_raw.get("from_number", ""),
        to_numbers=sms_raw.get("to_numbers", []) or [],
    )

    return cfg


def load_config(
    path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> UmbraConfig:
    """Load config from YAML file + env vars + CLI overrides.

    Args:
        path: Path to YAML config file. If None, looks for umbra.yml in cwd.
        cli_overrides: Dict of CLI flag values that override file/env.

    Returns:
        Fully resolved UmbraConfig.
    """
    raw: dict[str, Any] = {}

    # Try to load YAML file
    if path is None:
        path = Path.cwd() / "umbra.yml"
    else:
        path = Path(path)

    if path.exists():
        # Rule #14: Enforce file size limits
        file_size = path.stat().st_size
        if file_size > MAX_CONFIG_FILE_SIZE:
            raise ValueError(
                f"Config file too large: {file_size / 1024:.0f}KB > "
                f"{MAX_CONFIG_FILE_SIZE / 1024:.0f}KB"
            )
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    # Build config with env var fallbacks
    cfg = UmbraConfig(
        port=raw.get("port", DEFAULT_PORT),
        host=raw.get("host", DEFAULT_HOST),
        api_key=_resolve_env(raw.get("api_key", ""), "CI1T_API_KEY"),
        api_url=raw.get("api_url", DEFAULT_API_URL),
        policy=raw.get("policy", "monitor"),
        episode_size=raw.get("episode_size", DEFAULT_EPISODE_SIZE),
    )

    # Risk map: merge defaults with overrides
    risk_overrides = raw.get("risk_map", {}) or {}
    if risk_overrides:
        merged = dict(DEFAULT_RISK_MAP)
        merged.update(risk_overrides)
        cfg.risk_map = merged

    # Alerts
    alerts_raw = raw.get("alerts", {}) or {}
    cfg.alerts = _parse_alerts(alerts_raw)

    # Credits
    credits_raw = raw.get("credits", {}) or {}
    cfg.credits = CreditsConfig(
        low_warning=credits_raw.get("low_warning", DEFAULT_LOW_CREDIT_WARNING),
        check_interval=credits_raw.get("check_interval", DEFAULT_CREDIT_CHECK_INTERVAL),
    )

    # CLI overrides (highest priority)
    if cli_overrides:
        if cli_overrides.get("port") is not None:
            cfg.port = cli_overrides["port"]
        if cli_overrides.get("host") is not None:
            cfg.host = cli_overrides["host"]
        if cli_overrides.get("api_key"):
            cfg.api_key = cli_overrides["api_key"]
        if cli_overrides.get("api_url"):
            cfg.api_url = cli_overrides["api_url"]
        if cli_overrides.get("policy"):
            cfg.policy = cli_overrides["policy"]

    return cfg

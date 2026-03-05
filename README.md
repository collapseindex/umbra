# Umbra

<p align="center">
  <img src="moon.png" alt="Umbra" width="900" />
  <br />
</p>

**Version:** 0.2.0 | **License:** [ELv2](LICENSE) | **Last Updated:** March 5, 2026

---

## What is Umbra?

Umbra is a lightweight HTTP server that sits between your AI agent and the actions it wants to take. Before your agent does anything -- run a command, send a message, delete a file -- it asks Umbra first. Umbra scores the risk, tracks behavioral drift over time, and tells your agent: **go ahead**, **be careful**, **wait for a human**, or **stop**.

It works with any agent framework (CrewAI, AutoGen, LangGraph, OpenClaw, Claude Code, custom) -- if your agent can make an HTTP call, it can use Umbra.

```
Your Agent --> POST /check --> Umbra --> CI-1T API
          <-- { decision: "allow" } <--|
```

## Install

```bash
pip install umbra-gate
```

## Setup

```bash
umbra setup
```

This walks you through creating `umbra.yml` -- your API key, policy mode, alerts. Takes about 30 seconds.

Need an API key? [Get one free](https://collapseindex.org/dashboard) (1,000 credits, no credit card).

## Run

```bash
umbra serve
```

That's it. Umbra is now listening on `http://localhost:8400`.

---

## How Does It Decide?

Every action your agent sends gets a risk score. Umbra batches scores into episodes and sends them to [CI-1T](https://collapseindex.org) for evaluation. CI-1T returns:

- **CI** (Collapse Index) -- how unstable your agent's behavior is
- **AL** (Authority Level) -- how much trust to give
- **Ghost flags** -- whether the agent is suspiciously consistent (hiding errors)

Umbra maps the AL to a decision:

| AL | Meaning | Decision | Your agent should... |
|----|---------|----------|---------------------|
| 0-1 | High trust | **ALLOW** | Proceed normally |
| 2 | Moderate | **WARN** | Proceed, but a warning is logged |
| 3 | Low trust | **GATE** | Pause and wait for human approval |
| 4 | No trust | **BLOCK** | Stop immediately |
| -- | Ghost detected | **BLOCK** | Stop immediately |

**Policy modes:**
- `monitor` -- log everything, never block (good for testing)
- `enforce` -- actually gate/block at AL3+ (production)

---

## API

### Check an action

```
POST /check
```

Call this **before** your agent executes an action.

```json
{ "agent": "my-agent", "action": "terminal_exec" }
```

Fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent` | string | No | Agent name (default: "default"). Each agent gets its own session. |
| `action` | string | Yes* | Action type from the risk map (see below) |
| `escalation` | bool | No | Flag as dangerous (+0.25 risk boost) |
| `score` | int | Yes* | Raw Q0.16 score (0-65535). Use instead of `action` for custom scoring. |

*Provide `action` or `score` (one is required).

**Response (still buffering):**

```json
{ "decision": "allow", "buffered": true, "agent": "my-agent" }
```

**Response (full evaluation):**

```json
{
  "agent": "my-agent",
  "decision": "gate",
  "al": 3,
  "al_label": "Low trust -- gating active",
  "ci": 0.6872,
  "ghost_confirmed": false,
  "credits_remaining": 488
}
```

### Report an action (fire-and-forget)

```
POST /report
```

Same payload as `/check`, but returns immediately. Use for actions you want to track but not gate.

### Status

```
GET /status         # All agents
GET /status/{agent} # One agent
```

### Health

```
GET /health
```

### Reset an agent

```
DELETE /sessions/{agent}
```

Deletes the fleet session on CI-1T and clears all local state for that agent.

---

## Risk Map

Umbra ships with 17 built-in action types. You can override any of them in `umbra.yml`.

| Action | Risk | | Action | Risk |
|--------|------|-|--------|------|
| `file_read` | 0.05 | | `terminal_exec` | 0.50 |
| `search` | 0.05 | | `bash_execute` | 0.50 |
| `web_browse` | 0.20 | | `unknown` | 0.50 |
| `file_write` | 0.25 | | `spawn_agent` | 0.60 |
| `api_call` | 0.30 | | `config_modify` | 0.65 |
| `message` | 0.40 | | `permission_change` | 0.85 |
| `delete_file` | 0.40 | | `credential_access` | 0.90 |
| `install_package` | 0.45 | | `self_modify` | 0.90 |
| | | | `external_upload` | 0.90 |

---

## Examples

### Python

```python
import requests

def check(action, agent="my-agent", escalation=False):
    resp = requests.post("http://localhost:8400/check", json={
        "agent": agent, "action": action, "escalation": escalation
    })
    return resp.json()

result = check("terminal_exec")
if result["decision"] in ("gate", "block"):
    print(f"Blocked: AL={result.get('al')}")
```

### JavaScript

```javascript
const resp = await fetch("http://localhost:8400/check", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ agent: "my-agent", action: "terminal_exec" }),
});
const { decision } = await resp.json();
```

### curl

```bash
curl -X POST http://localhost:8400/check \
  -H "Content-Type: application/json" \
  -d '{"agent": "test", "action": "terminal_exec"}'
```

---

## Configuration

`umbra.yml` is created by `umbra setup`. Here's the full schema:

```yaml
port: 8400
host: "0.0.0.0"

# CI-1T API key (or use CI1T_API_KEY env var)
api_key: ""
api_url: "https://collapseindex.org/api"

# "monitor" or "enforce"
policy: "monitor"

# Scores per episode (2-8)
episode_size: 3

# Override risk scores
risk_map:
  terminal_exec: 0.70
  file_read: 0.02

# Alerts (all opt-in)
alerts:
  min_level: "AL3"
  slack:
    webhook_url: "https://hooks.slack.com/services/..."
  email:
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    smtp_user: "you@gmail.com"
    smtp_pass: ""
    from_addr: "umbra@yourdomain.com"
    to_addrs: ["ops@yourdomain.com"]
  sms:
    account_sid: ""
    auth_token: ""
    from_number: "+15551234567"
    to_numbers: ["+15559876543"]

# Credit monitoring
credits:
  low_warning: 100
  check_interval: 10
```

### Environment Variables

| Variable | Overrides |
|----------|-----------|
| `CI1T_API_KEY` | `api_key` in config |
| `CI1T_SLACK_WEBHOOK` | `alerts.slack.webhook_url` |
| `CI1T_SMTP_PASS` | `alerts.email.smtp_pass` |
| `CI1T_TWILIO_SID` | `alerts.sms.account_sid` |
| `CI1T_TWILIO_TOKEN` | `alerts.sms.auth_token` |

---

## Docker

```bash
docker run -d \
  -e CI1T_API_KEY=ci_your_key \
  -p 8400:8400 \
  -v ./umbra.yml:/app/umbra.yml:ro \
  collapseindex/umbra
```

Or build locally:

```bash
docker build -t umbra .
docker compose up -d
```

---

## CLI

```
umbra setup                    Create umbra.yml interactively
umbra serve                    Start the server
umbra serve --port 9000        Override port
umbra serve --policy enforce   Override policy mode
umbra status                   Check a running server's status
umbra --version                Print version
```

---

## Security

Passed full automated audit: **bandit** (0 findings), **flake8** (0 errors), **pip-audit** (0 vulnerabilities).

Highlights:
- API keys masked in all logs, repr, and tracebacks
- Input validation on all fields (agent names, action types, scores, body size)
- YAML loaded with `safe_load()` only
- Rate limiting (200 req/60s) on POST endpoints
- Max 500 concurrent sessions, 1MB config limit
- Error responses never leak internal details
- All alert channels (Slack, email, SMS) are opt-in

Run it yourself:

```bash
python -m bandit -r umbra/ -q
python -m flake8 umbra/ --max-line-length 120
python -m pip_audit -r requirements.txt
python -m pytest tests/ -q
```

Report vulnerabilities to ask@collapseindex.org (not public issues).

---

## Project Structure

```
umbra/
  __init__.py       config.py       scorer.py       alerts.py
  __main__.py       server.py       policy.py       setup.py
                    sessions.py
tests/
  test_core.py      test_server.py
umbra.example.yml   Dockerfile      docker-compose.yml
pyproject.toml      requirements.txt
```

---

## Changelog

### v0.2.0 (2026-03-05)
- Renamed project from ci1t-gate to **Umbra**
- PyPI: `umbra-gate` / CLI: `umbra` / Config: `umbra.yml` / Import: `from umbra import ...`

### v0.1.1 (2026-03-05)
- Full security audit (bandit, flake8, pip-audit -- all clean)
- Added key masking, input sanitization, rate limiting, session limits
- Fixed error info leaks, header injection, Content-Length crash
- 22 new security tests (59 total)

### v0.1.0 (2026-03-05)
- Initial release -- HTTP server, fleet sessions, risk scoring, policy enforcement, ghost detection, alerts (Slack/email/SMS), setup wizard, Docker

## License

Copyright 2026 Alex Kwon ([Collapse Index Labs](https://collapseindex.org))

[Elastic License 2.0 (ELv2)](LICENSE) -- free to use, modify, and self-host. The only restriction is you can't offer Umbra as a hosted/managed service.

---

<em>"though love is hard to see, you're still real here with me."</em>
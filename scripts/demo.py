"""End-to-end demo of Umbra.

Simulates an agent going from normal behavior to dangerous escalation
while Umbra monitors and enforces policy in real-time.
"""

import httpx
import time
import sys

GATE = "http://127.0.0.1:8400"
client = httpx.Client(timeout=30.0)

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


def check(agent: str, action: str, escalation: bool = False) -> dict:
    body = {"agent": agent, "action": action}
    if escalation:
        body["escalation"] = True
    resp = client.post(f"{GATE}/check", json=body)
    resp.raise_for_status()
    return resp.json()


def colorize_decision(d: str) -> str:
    colors = {"allow": GREEN, "warn": YELLOW, "gate": RED, "block": f"{RED}{BOLD}"}
    return f"{colors.get(d, '')}{d.upper()}{RESET}"


def print_result(action: str, r: dict, tag: str = ""):
    dec = colorize_decision(r["decision"])
    extra = ""
    if r.get("buffered"):
        extra = f" {DIM}(buffered){RESET}"
    else:
        ci = r.get("ci", 0)
        al = r.get("al", 0)
        ghost = ""
        if r.get("ghost_confirmed"):
            ghost = f" {MAGENTA}{BOLD}GHOST{RESET}"
        elif r.get("ghost_suspect"):
            ghost = f" {MAGENTA}ghost?{RESET}"
        credits = r.get("credits_remaining", -1)
        credit_str = f"  credits={credits}" if credits >= 0 else ""
        extra = f"  CI={ci:.3f}  AL={al}{ghost}{credit_str}"

    prefix = f"{tag} " if tag else ""
    print(f"    {prefix}{action:<22s} -> {dec}{extra}")


# ========================================
print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
print(f"{BOLD}{CYAN}  Umbra End-to-End Demo{RESET}")
print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")

# Step 1: Health
print(f"{YELLOW}[1] Health check{RESET}")
h = client.get(f"{GATE}/health").json()
print(f"    status={h['status']}  version={h['version']}  policy={h['policy']}")
print()

# Step 2: Normal actions
print(f"{YELLOW}[2] Normal agent actions (low risk -- expect ALLOW){RESET}")
for action in ["file_read", "search", "file_read"]:
    r = check("demo-agent", action)
    print_result(action, r)
print()

# Step 3: Moderate escalation
print(f"{YELLOW}[3] Escalating actions (risk climbing){RESET}")
for action in ["terminal_exec", "file_write", "api_call"]:
    r = check("demo-agent", action)
    print_result(action, r)
print()

# Step 4: Dangerous
print(f"{YELLOW}[4] Dangerous actions with escalation flag{RESET}")
for action in ["credential_access", "self_modify", "external_upload"]:
    r = check("demo-agent", action, escalation=True)
    print_result(action, r, tag="[ESC]")
print()

# Step 5: More dangerous to push CI higher
print(f"{YELLOW}[5] Sustained dangerous behavior{RESET}")
for action in ["credential_access", "permission_change", "self_modify"]:
    r = check("demo-agent", action, escalation=True)
    print_result(action, r, tag="[ESC]")
print()

# Step 6: Status
print(f"{YELLOW}[6] Overall status{RESET}")
s = client.get(f"{GATE}/status").json()
print(f"    policy={s['policy']}  requests={s['total_requests']}  uptime={s['uptime_seconds']}s")
for a in s.get("agents", []):
    ghost = " GHOST" if a.get("ghost_confirmed") else ""
    ci = a.get("ci", 0)
    print(f"    {a['agent']:<20s} AL={a.get('al', '?')}  CI={ci:.3f}  rounds={a.get('round_count', 0)}{ghost}")
print()

# Step 7: Second agent
print(f"{YELLOW}[7] Second agent (multi-agent support){RESET}")
for action in ["file_read", "search", "web_browse"]:
    r = check("helper-bot", action)
    print_result(action, r)
print()

# Step 8: Fire-and-forget /report
print(f"{YELLOW}[8] Fire-and-forget /report{RESET}")
resp = client.post(f"{GATE}/report", json={"agent": "bg-worker", "action": "api_call"})
r = resp.json()
print(f"    accepted={r['accepted']}  agent={r['agent']}")
print()

# Step 9: Multi-agent status
print(f"{YELLOW}[9] Multi-agent status{RESET}")
s2 = client.get(f"{GATE}/status").json()
for a in s2.get("agents", []):
    ci = a.get("ci", 0)
    print(f"    {a['agent']:<20s} AL={a.get('al', '?')}  CI={ci:.3f}  rounds={a.get('round_count', 0)}")
print()

# Step 10: Clean up
print(f"{YELLOW}[10] Clean up -- delete demo-agent session{RESET}")
resp = client.delete(f"{GATE}/sessions/demo-agent")
r = resp.json()
print(f"    deleted={r['deleted']}  agent={r['agent']}")
print()

print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
print(f"{BOLD}{CYAN}  Demo complete!{RESET}")
print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")

client.close()

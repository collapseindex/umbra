"""OpenRouter + Umbra: Real LLM agent gated by CI-1T.

A coding agent powered by Claude Sonnet 4.6 via OpenRouter makes decisions.
Each tool call is checked through Umbra's gate before "executing."
CI-C1 watches the behavioral patterns in the background.

Usage:
    1. Add openrouter_key to umbra.yml (or set OPENROUTER_API_KEY env var)
    2. Start Umbra:     umbra serve
    3. Run this script: python scripts/demo_openrouter.py

Requires: umbra.yml with a valid CI-1T API key and curiosity enabled.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import yaml
import httpx

UMBRA_URL = "http://127.0.0.1:8400"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4"
AGENT_NAME = "sonnet-coder"

# Tools the LLM can "call" -- each maps to an Umbra action
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file (create or overwrite).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_exec",
            "description": "Execute a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
            },
        },
    },
]

# Task prompts that push the LLM toward different tool-use patterns
TASKS = [
    {
        "label": "Research Task (low risk)",
        "prompt": (
            "You are a coding assistant. The user asks: "
            "'What are the best practices for error handling in Python?' "
            "Use your tools to research this. Read some files, maybe search the web. "
            "Reply with a short summary."
        ),
    },
    {
        "label": "Code Fix Task (medium risk)",
        "prompt": (
            "You are a coding assistant. The user reports a bug in app.py line 42: "
            "'TypeError: NoneType has no attribute split'. "
            "Use your tools to read the file, understand the issue, and write a fix."
        ),
    },
    {
        "label": "Deploy Task (higher risk)",
        "prompt": (
            "You are a DevOps assistant. The user says: "
            "'Deploy the latest build to production. Run the tests first, "
            "then build the Docker image and push it.' "
            "Use your tools to execute the deployment pipeline."
        ),
    },
]


async def check_umbra(client: httpx.AsyncClient, action: str) -> dict:
    """Gate a tool call through Umbra."""
    resp = await client.post(
        f"{UMBRA_URL}/check",
        json={"agent": AGENT_NAME, "action": action},
    )
    return resp.json()


async def call_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    messages: list[dict],
) -> dict:
    """Send a chat completion request to OpenRouter."""
    resp = await client.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 1024,
        },
        timeout=60.0,
    )
    if resp.status_code != 200:
        print(f"  OpenRouter error: HTTP {resp.status_code}")
        return {}
    return resp.json()


async def run_agent_loop(
    client: httpx.AsyncClient,
    api_key: str,
    task: dict,
    max_turns: int = 5,
):
    """Run one task through the LLM agent loop with Umbra gating."""
    print(f"\n  Task: {task['label']}")
    print(f"  Prompt: {task['prompt'][:80]}...")
    print()

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant with access to tools."},
        {"role": "user", "content": task["prompt"]},
    ]

    for turn in range(max_turns):
        # Ask the LLM
        result = await call_openrouter(client, api_key, messages)
        if not result:
            print("  [LLM] No response. Stopping.")
            break

        choices = result.get("choices", [])
        if not choices:
            print("  [LLM] Empty choices. Stopping.")
            break

        message = choices[0].get("message", {})
        finish_reason = choices[0].get("finish_reason", "")

        # If LLM wants to use tools
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            messages.append(message)  # add assistant message with tool_calls

            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                tool_args = func.get("arguments", "{}")
                tc_id = tc.get("id", "")

                # Gate through Umbra
                gate_result = await check_umbra(client, tool_name)
                decision = gate_result.get("decision", "?")
                ci = gate_result.get("ci", "")
                al = gate_result.get("al", "")
                buffered = gate_result.get("buffered", False)

                if buffered:
                    status = "buffered"
                    gate_line = "buffered (collecting episodes)"
                else:
                    status = decision
                    gate_line = f"{decision}  CI={ci}  AL={al}"

                print(f"  [Turn {turn + 1}] {tool_name}({tool_args[:60]}) -> {gate_line}")

                # Simulate tool response based on gate decision
                if status in ("allow", "buffered"):
                    tool_response = f"[Simulated] {tool_name} executed successfully."
                elif status == "softblock":
                    tool_response = (
                        f"[UMBRA SOFTBLOCK] Action '{tool_name}' was soft-blocked. "
                        f"CI={ci}, AL={al}. The agent's instability score is elevated. "
                        "Proceed with caution or try a safer approach."
                    )
                else:
                    tool_response = (
                        f"[UMBRA BLOCKED] Action '{tool_name}' was hard-blocked. "
                        f"CI={ci}, AL={al}. This action is not allowed at the current "
                        "authority level. Try a different approach."
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_response,
                })

        elif message.get("content"):
            # LLM gave a text reply (no tool calls) -- done
            content = message["content"]
            preview = content[:120].replace("\n", " ")
            print(f"  [Turn {turn + 1}] LLM reply: {preview}...")
            break
        else:
            print(f"  [Turn {turn + 1}] Empty message (finish_reason={finish_reason})")
            break

    print()


def _load_openrouter_key() -> str:
    """Read key from umbra.yml, fall back to env var."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    yml_path = os.path.join(os.path.dirname(__file__), "..", "umbra.yml")
    if os.path.exists(yml_path):
        with open(yml_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        key = cfg.get("openrouter_key", "")
    return key


async def run_demo():
    api_key = _load_openrouter_key()
    if not api_key or api_key == "YOUR_KEY_HERE":
        print("Add your OpenRouter key to umbra.yml:")
        print('  openrouter_key: "sk-or-..."')
        print("Or set OPENROUTER_API_KEY env var.")
        sys.exit(1)

    print("=" * 60)
    print("OPENROUTER + UMBRA: LIVE LLM AGENT DEMO")
    print(f"Model: {MODEL}")
    print(f"Agent: {AGENT_NAME}")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Health check
        try:
            resp = await client.get(f"{UMBRA_URL}/health")
            health = resp.json()
            print(f"\nUmbra {health.get('version', '?')} running ({health.get('policy', '?')} mode)")
        except httpx.ConnectError:
            print(f"\nCannot connect to Umbra at {UMBRA_URL}")
            print("Start it first: umbra serve")
            sys.exit(1)

        # Check curiosity
        resp = await client.get(f"{UMBRA_URL}/discoveries")
        if resp.status_code == 200:
            print("Curiosity engine: active")
        else:
            print("Curiosity engine: disabled (enable in umbra.yml for pattern detection)")

        # Run each task
        for i, task in enumerate(TASKS, 1):
            print(f"\n{'=' * 60}")
            print(f"TASK {i}/{len(TASKS)}")
            print("=" * 60)
            await run_agent_loop(client, api_key, task)
            await asyncio.sleep(1)

        # Wait for curiosity cycle
        print("=" * 60)
        print("Waiting for curiosity cycle to analyze patterns...")
        print("=" * 60)

        for attempt in range(12):
            await asyncio.sleep(5)
            resp = await client.get(f"{UMBRA_URL}/discoveries")
            if resp.status_code == 200:
                data = resp.json()
                discoveries = data.get("discoveries", [])
                if discoveries:
                    print(f"\n  {len(discoveries)} discovery(ies):\n")
                    for d in discoveries:
                        print(f"  [{d['match_type']}] {d['explanation']}")
                        print(f"  Similarity: {d['similarity']:.1%}\n")
                    break
            print(f"  ...check {attempt + 1}/12")

        # Final status
        print("=" * 60)
        print("FINAL AGENT STATUS")
        print("=" * 60)

        resp = await client.get(f"{UMBRA_URL}/status/{AGENT_NAME}")
        if resp.status_code == 200:
            status = resp.json()
            print(f"  Agent: {AGENT_NAME}")
            print(f"  CI:    {status.get('ci', '?')}")
            print(f"  AL:    {status.get('al', '?')}")
            print(f"  Ghost: {status.get('ghost_confirmed', '?')}")
            print(f"  Rounds: {status.get('round', '?')}")
        else:
            print(f"  No status found for {AGENT_NAME}")

        # Cleanup
        print(f"\n{'=' * 60}")
        print("Cleaning up...")
        await client.delete(f"{UMBRA_URL}/sessions/{AGENT_NAME}")
        print("Done.")


if __name__ == "__main__":
    asyncio.run(run_demo())

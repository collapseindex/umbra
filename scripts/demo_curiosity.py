"""CI-C1 live demo: two agents with correlated behavior flowing through Umbra.

Usage:
    1. In one terminal:  umbra serve
    2. In another:       python scripts/demo_curiosity.py

Requires: umbra.yml with a valid CI-1T API key and curiosity enabled.
"""

from __future__ import annotations

import asyncio
import sys
import time

import httpx

UMBRA_URL = "http://127.0.0.1:8400"

# Two agents that will exhibit similar behavioral patterns
AGENT_A = "alpha-coder"
AGENT_B = "beta-coder"
AGENT_C = "gamma-ops"  # deliberately different pattern


async def send_check(client: httpx.AsyncClient, agent: str, action: str) -> dict:
    """Send a /check request and return the response."""
    resp = await client.post(
        f"{UMBRA_URL}/check",
        json={"agent": agent, "action": action},
    )
    data = resp.json()
    decision = data.get("decision", "?")
    ci = data.get("ci", "")
    al = data.get("al", "")
    buffered = data.get("buffered", False)
    if buffered:
        print(f"  [{agent}] {action:20s} -> buffered")
    else:
        print(f"  [{agent}] {action:20s} -> {decision:5s}  CI={ci}  AL={al}")
    return data


async def get_discoveries(client: httpx.AsyncClient) -> list:
    """Query the discoveries endpoint."""
    resp = await client.get(f"{UMBRA_URL}/discoveries")
    if resp.status_code == 404:
        print("\n  Curiosity engine is not enabled. Add this to umbra.yml:")
        print("    curiosity:")
        print("      enabled: true")
        print("      cycle_interval: 5")
        sys.exit(1)
    data = resp.json()
    return data.get("discoveries", [])


async def run_demo():
    print("=" * 60)
    print("CI-C1 CURIOSITY ENGINE - LIVE DEMO")
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

        # Check curiosity is enabled
        await get_discoveries(client)
        print("Curiosity engine: active\n")

        # ---- Phase 1: Similar patterns for Agent A and Agent B ----
        print("-" * 60)
        print("PHASE 1: Two agents with similar behavior patterns")
        print("  alpha-coder and beta-coder both do: read -> read -> write")
        print("-" * 60)

        for round_num in range(1, 6):
            print(f"\n  Round {round_num}:")
            # A and B follow the same action sequence
            await send_check(client, AGENT_A, "file_read")
            await send_check(client, AGENT_B, "file_read")
            await send_check(client, AGENT_A, "file_read")
            await send_check(client, AGENT_B, "file_read")
            await send_check(client, AGENT_A, "file_write")
            await send_check(client, AGENT_B, "file_write")
            await asyncio.sleep(0.3)

        # ---- Phase 2: Different pattern for Agent C ----
        print(f"\n{'-' * 60}")
        print("PHASE 2: gamma-ops does high-risk actions (different pattern)")
        print("-" * 60)

        for round_num in range(1, 6):
            print(f"\n  Round {round_num}:")
            await send_check(client, AGENT_C, "terminal_exec")
            await send_check(client, AGENT_C, "credential_access")
            await send_check(client, AGENT_C, "delete_file")
            await asyncio.sleep(0.3)

        # ---- Wait for curiosity cycle ----
        print(f"\n{'-' * 60}")
        print("Waiting for curiosity cycle...")
        print("-" * 60)

        # Poll discoveries for up to 90 seconds
        found = False
        for i in range(18):
            await asyncio.sleep(5)
            discoveries = await get_discoveries(client)
            if discoveries:
                found = True
                print(f"\n  {len(discoveries)} discovery(ies) found!\n")
                for d in discoveries:
                    print(f"  [{d['match_type']}]")
                    print(f"  {d['explanation']}")
                    print(f"  Similarity: {d['similarity']}")
                    print()
                break
            else:
                print(f"  ...cycle check {i + 1}/18 (no discoveries yet)")

        if not found:
            print("\n  No discoveries surfaced. This can happen if:")
            print("  - CI scores are too uniform (not enough shape variance)")
            print("  - Not enough episodes accumulated for the window size")
            print("  - Similarity threshold is too high")
            print("  Try lowering similarity_threshold to 0.5 in umbra.yml")

        # ---- Show final status ----
        print(f"\n{'=' * 60}")
        print("FINAL STATUS")
        print("=" * 60)

        resp = await client.get(f"{UMBRA_URL}/status")
        status = resp.json()
        for agent_info in status.get("agents", []):
            name = agent_info.get("agent", "?")
            ci = agent_info.get("ci", "?")
            al = agent_info.get("al", "?")
            ghost = agent_info.get("ghost_confirmed", False)
            print(f"  {name:20s}  CI={ci}  AL={al}  ghost={ghost}")

        # Cleanup
        print(f"\n{'=' * 60}")
        print("Cleaning up sessions...")
        for agent in [AGENT_A, AGENT_B, AGENT_C]:
            await client.delete(f"{UMBRA_URL}/sessions/{agent}")
        print("Done.")


if __name__ == "__main__":
    asyncio.run(run_demo())

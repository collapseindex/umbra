"""CLI entry point -- Umbra.

Usage:
    umbra setup                    Interactive config wizard
    umbra serve                    Start the HTTP gate server
    umbra serve --port 9000        Override port
    umbra status                   Check running gate status
    umbra --version                Print version
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="umbra",
        description="HTTP policy gate for autonomous agents -- powered by CI-1T",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"umbra v{__version__}",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # -- setup --
    setup_parser = sub.add_parser("setup", help="Interactive config wizard")
    setup_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for config file (default: ./umbra.yml)",
    )

    # -- serve --
    serve_parser = sub.add_parser("serve", help="Start the HTTP gate server")
    serve_parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to umbra.yml (default: ./umbra.yml)",
    )
    serve_parser.add_argument("--port", "-p", type=int, default=None, help="HTTP port")
    serve_parser.add_argument("--host", default=None, help="Bind host")
    serve_parser.add_argument(
        "--api-key",
        default=None,
        help="CI-1T API key (overrides config/env)",
    )
    serve_parser.add_argument(
        "--policy",
        choices=["monitor", "enforce"],
        default=None,
        help="Policy mode (overrides config)",
    )
    serve_parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # -- status --
    status_parser = sub.add_parser("status", help="Check a running gate's status")
    status_parser.add_argument(
        "--url",
        default="http://localhost:8400",
        help="Gate URL (default: http://localhost:8400)",
    )

    return parser


def cmd_setup(args: argparse.Namespace) -> int:
    from .setup import run_setup
    run_setup(output_path=args.output)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .config import load_config
    from .server import UmbraServer

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config with CLI overrides
    cli_overrides = {
        "port": args.port,
        "host": args.host,
        "api_key": args.api_key,
        "policy": args.policy,
    }
    cfg = load_config(path=args.config, cli_overrides=cli_overrides)

    # Validate
    errors = cfg.validate()
    if errors:
        for e in errors:
            print(f"  Config error: {e}", file=sys.stderr)
        print("\n  Run 'umbra setup' to create a config file.", file=sys.stderr)
        return 1

    # Build app and run
    gate = UmbraServer(cfg)
    app = gate.build_app()

    # Print startup banner
    print()
    print(f"  \033[1m\033[96mumbra v{__version__}\033[0m")
    print(f"  \033[2m{'=' * 40}\033[0m")
    print(f"  Policy:   \033[1m{cfg.policy}\033[0m")
    print(f"  API:      {cfg.api_url}")
    print(f"  Endpoint: http://{cfg.host}:{cfg.port}")
    print()

    channels = []
    if cfg.alerts.slack.enabled:
        channels.append("Slack")
    if cfg.alerts.email.enabled:
        channels.append("Email")
    if cfg.alerts.sms.enabled:
        channels.append("SMS")
    if channels:
        print(f"  Alerts:   {', '.join(channels)} (min: {cfg.alerts.min_level})")
    else:
        print("  Alerts:   none configured")

    print(f"  Credits:  warn at {cfg.credits.low_warning} remaining")
    print()

    import uvicorn
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level="debug" if args.debug else "info",
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    import httpx

    url = args.url.rstrip("/")

    try:
        resp = httpx.get(f"{url}/health", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        print(f"  Status:  {data.get('status', 'unknown')}")
        print(f"  Version: {data.get('version', 'unknown')}")
        print(f"  Policy:  {data.get('policy', 'unknown')}")
        print(f"  Uptime:  {data.get('uptime_seconds', 0)}s")
    except httpx.ConnectError:
        print(f"  Gate not running at {url}")
        return 1
    except Exception as e:
        print(f"  Error: {e}")
        return 1

    # Also get agent statuses
    try:
        resp = httpx.get(f"{url}/status", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        agents = data.get("agents", [])
        if agents:
            print(f"\n  Active agents ({len(agents)}):")
            for a in agents:
                al = a.get("al", 0)
                ci = a.get("ci", 0)
                ghost = " GHOST" if a.get("ghost_confirmed") else ""
                print(f"    {a['agent']:<30s} AL={al} CI={ci:.3f} rounds={a.get('round_count', 0)}{ghost}")
        else:
            print("\n  No active agents")
    except Exception:  # nosec B110 -- agent status is supplementary
        pass

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "setup": cmd_setup,
        "serve": cmd_serve,
        "status": cmd_status,
    }

    handler = commands.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())

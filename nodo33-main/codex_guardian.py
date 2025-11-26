#!/usr/bin/env python3
# codex_guardian.py

import argparse
import sys
from typing import List

from orchestrator.guardian_orchestrator import GuardianOrchestrator


def cmd_list_agents(orchestrator: GuardianOrchestrator, args: argparse.Namespace) -> None:
    agents = orchestrator.list_agents()
    print("Agenti registrati in Nodo33:")
    for a in agents:
        tags = ", ".join(a.tags) if a.tags else "-"
        print(f"- {a.id}: {a.name}")
        print(f"    ruolo: {a.role}")
        print(f"    tags : {tags}")


def cmd_scan_url(orchestrator: GuardianOrchestrator, args: argparse.Namespace) -> None:
    url = args.url
    if not url:
        print("Serve un URL da scansionare.", file=sys.stderr)
        sys.exit(1)

    agent_ids: List[str] = args.agents.split(",") if args.agents else None

    report = orchestrator.run_pipeline(
        url=url,
        text=None,
        agent_ids=agent_ids,
    )
    print(orchestrator.format_report(report))


def cmd_scan_text(orchestrator: GuardianOrchestrator, args: argparse.Namespace) -> None:
    text = args.text
    if not text:
        # se non c'è argomento, leggiamo da stdin
        print("Leggo testo da stdin (Ctrl+D per terminare)...", file=sys.stderr)
        text = sys.stdin.read()

    agent_ids: List[str] = args.agents.split(",") if args.agents else None

    report = orchestrator.run_pipeline(
        url=None,
        text=text,
        agent_ids=agent_ids,
    )
    print(orchestrator.format_report(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CODex Guardian – Orchestratore etico di Nodo33"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # list-agents
    p_list = sub.add_parser("list-agents", help="Mostra tutti gli agenti registrati")
    p_list.set_defaults(func=cmd_list_agents)

    # scan-url
    p_url = sub.add_parser(
        "scan-url",
        help="Esegue la pipeline sugli agenti (default: guardian_ethics,malta_scanner,light_agent) per un URL",
    )
    p_url.add_argument("url", help="URL da analizzare")
    p_url.add_argument(
        "--agents",
        help="Lista di agent_ids separati da virgola (override pipeline di default)",
    )
    p_url.set_defaults(func=cmd_scan_url)

    # scan-text
    p_text = sub.add_parser(
        "scan-text",
        help="Esegue la pipeline su testo grezzo (da argomento o stdin)",
    )
    p_text.add_argument(
        "text",
        nargs="?",
        help="Testo da analizzare (se omesso, legge da stdin)",
    )
    p_text.add_argument(
        "--agents",
        help="Lista di agent_ids separati da virgola (override pipeline di default)",
    )
    p_text.set_defaults(func=cmd_scan_text)

    return parser


def main() -> None:
    parser = build_parser()
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disattiva il logging su file per questa esecuzione",
    )
    args = parser.parse_args()

    orchestrator = GuardianOrchestrator(
        enable_logging=not args.no_log,
    )

    args.func(orchestrator, args)


if __name__ == "__main__":
    main()

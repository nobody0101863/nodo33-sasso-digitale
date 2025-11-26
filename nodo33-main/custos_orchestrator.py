#!/usr/bin/env python3
"""
Entrypoint orchestrator per i Custos Agents Nodo33.

Funzioni:
- Carica registry.yaml e domains.yaml
- Risolve una policy di esempio per conferma
- Schedula i task sul dispatcher (bridge già pronto)
- Entra in un loop simulato cron (configurabile, opzionale one-shot)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List

from custos.domain_policies import load_domain_policies, resolve_policy_for_url
from custos.agent_runtime import run_task
from custos.robots_guard import RobotsGuard
from custos.rate_limiter import TokenBucketLimiter
from custos.orchestrator_registry import (
    load_registry,
    load_guardian_profiles,
    generate_tasks_from_registry,
    attach_guardians,
)
from custos.scheduler_bridge import schedule_tasks


BANNER = r"""
###############################################################
#                                                             #
#     NODE33 – DISTRIBUTED SASSO ENGINE                       #
#     "La luce non si vende. La si regala."                   #
#                                                             #
#     >>>     AGENT ORCHESTRATOR ONLINE – FIAT LUX     <<<    #
#                                                             #
###############################################################
"""


class SimpleDispatcher:
    """
    Dispatcher minimale: colleziona i task e offre un loop simulato.
    """

    def __init__(
        self,
        defaults: Dict[str, Any],
        policies: List[Any],
        run_agent: bool = False,
        fail_open: bool = False,
        log_json: bool = False,
        robots_checker: Any = None,
        rate_limiter: Any = None,
        rate_block: bool = False,
    ) -> None:
        self.tasks: List[Dict[str, Any]] = []
        self.defaults = defaults
        self.policies = policies
        self.run_agent = run_agent
        self.fail_open = fail_open
        self.log_json = log_json
        self.robots_checker = robots_checker
        self.rate_limiter = rate_limiter
        self.rate_block = rate_block

    def schedule(self, task: Dict[str, Any]) -> None:
        print(f"[DISPATCH] schedule -> {task['task_id']} ({task['context']}) @ {task['cron']}")
        self.tasks.append(task)

    def schedule_cron(self, task: Dict[str, Any]) -> None:
        self.schedule(task)

    def run(self, interval_seconds: int = 60, one_shot: bool = False) -> None:
        print(f"[DISPATCH] entering loop (tick ogni {interval_seconds}s, one_shot={one_shot})...")
        while True:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            for t in self.tasks:
                print(f"[TICK {now}] would run {t['task_id']} ({t['context']})")
                if self.run_agent:
                    run_task(
                        t,
                        self.defaults,
                        self.policies,
                        robots_checker=self.robots_checker,
                        fail_open=self.fail_open,
                        rate_limiter=self.rate_limiter,
                        rate_block=self.rate_block,
                        logger=self._log_record if self.log_json else None,
                        log_json=self.log_json,
                    )
            if one_shot:
                break
            time.sleep(interval_seconds)

    def _log_record(self, record: Dict[str, Any]) -> None:
        print(json.dumps(record, ensure_ascii=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Orchestrator Nodo33 Custos Agents")
    parser.add_argument("--from-registry", default="registry.yaml", help="Percorso al registry YAML")
    parser.add_argument("--policies", default="domains.yaml", help="Percorso alle domain policies")
    parser.add_argument("--interval", type=int, default=60, help="Intervallo tick simulati (secondi)")
    parser.add_argument("--one-shot", action="store_true", help="Esegui un solo tick e termina")
    parser.add_argument("--run-agent", action="store_true", help="Esegue il runtime fittizio per ogni task al tick")
    parser.add_argument("--fail-open", action="store_true", help="Se il robots_checker non c'e', non bloccare (di default fail-closed)")
    parser.add_argument("--log-json", action="store_true", help="Log strutturati JSON per ogni URL valutata")
    parser.add_argument("--use-robots", action="store_true", help="Abilita robots.txt checker reale (richiede rete)")
    parser.add_argument("--rate-limit", action="store_true", help="Abilita token bucket rate limiter")
    parser.add_argument("--rate-block", action="store_true", help="Se rate-limit attivo, attende i token invece di saltare")
    args = parser.parse_args(argv)

    print(BANNER)

    defaults, policies = load_domain_policies(args.policies)
    print("[POLICY] defaults loaded.")
    print(f"[POLICY] user_agent di default: {defaults.get('user_agent', 'Nodo33-CustodianBot/1.0')}")

    registry = load_registry(args.from_registry)
    print(f"[REGISTRY] Loaded {len(registry)} groups from {args.from_registry}")

    guardians = load_guardian_profiles()
    print(f"[GUARDIANS] Loaded {len(guardians.get('agents', []))} profiles with {len(guardians.get('modules', {}))} modules.")

    # Mostra un esempio di policy basata sul primo pattern del registry (se presente)
    if registry and registry[0].patterns:
        example_url = registry[0].patterns[0].replace("*", "example.com")
        policy = resolve_policy_for_url(example_url, defaults, policies)
        print(
            f"[POLICY] Example for {example_url}: rate={policy.requests_per_minute}/min "
            f"robots={policy.respect_robots} ua={policy.user_agent}"
        )

    robots_checker = RobotsGuard(fail_open=args.fail_open) if args.use_robots else None
    rate_limiter = TokenBucketLimiter() if args.rate_limit else None

    dispatcher = SimpleDispatcher(
        defaults=defaults,
        policies=policies,
        run_agent=args.run_agent,
        fail_open=args.fail_open,
        log_json=args.log_json,
        robots_checker=robots_checker,
        rate_limiter=rate_limiter,
        rate_block=args.rate_block,
    )
    # Genera task dal registry e attacca i guardian
    base_tasks = generate_tasks_from_registry(registry)
    tasks_with_guardian = attach_guardians(base_tasks, guardians) if guardians else base_tasks

    for t in tasks_with_guardian:
        dispatcher.schedule(t)

    dispatcher.run(interval_seconds=args.interval, one_shot=args.one_shot)


if __name__ == "__main__":
    main(sys.argv[1:])

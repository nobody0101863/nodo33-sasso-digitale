"""
Runtime fittizio per i Custos Agents: valida robots/TOS e mostra headers/rate-limit applicati.
Non esegue richieste HTTP reali (no rete); serve a collegare dispatcher→policy→agent loop.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from custos.domain_policies import load_domain_policies, should_skip_url


def _example_url_from_pattern(pattern: str) -> str:
    """
    Crea un esempio di URL sostituendo i wildcard con 'example.com'.
    Serve solo a mostrare l'applicazione delle policy senza uscire in rete.
    """
    return pattern.replace("*", "example.com")


def run_task(
    task: Dict[str, Any],
    defaults: Dict[str, Any],
    policies: List[Any],
    robots_checker: Optional[Callable[[str, str], bool]] = None,
    fail_open: bool = False,
    rate_limiter: Optional[Any] = None,
    rate_block: bool = False,
    logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_json: bool = False,
) -> List[Dict[str, Any]]:
    """
    Simula l'esecuzione di un task:
    - genera un URL di esempio per ogni pattern
    - applica robots/TOS e policy
    - restituisce log/risultati senza fare I/O di rete
    - applica rate limiting se fornito (token bucket)
    - produce log strutturati (logger callback o stdout JSON se log_json=True)
    """
    results: List[Dict[str, Any]] = []
    patterns: Iterable[str] = task.get("payload", {}).get("patterns", [])
    task_id = task.get("task_id")
    group_id = task.get("group_id")
    context = task.get("context")
    priority = task.get("payload", {}).get("priority")
    no_private = task.get("payload", {}).get("no_private_areas")

    for pat in patterns:
        url = _example_url_from_pattern(pat)
        skip, reason, policy = should_skip_url(
            url,
            defaults,
            policies,
            robots_checker=robots_checker,
            fail_open=fail_open,
        )
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record: Dict[str, Any] = {
            "timestamp": timestamp,
            "task_id": task_id,
            "group_id": group_id,
            "context": context,
            "priority": priority,
            "pattern": pat,
            "url": url,
            "status": "skipped" if skip else "ok",
            "reason": reason if skip else None,
            "robots_checked": robots_checker is not None,
            "robots_fail_open": fail_open,
            "policy": {
                "pattern": policy.pattern,
                "respect_robots": policy.respect_robots,
                "tos_blocked": policy.tos_blocked,
                "rpm": policy.requests_per_minute,
                "burst": policy.burst,
                "user_agent": policy.user_agent,
                "headers": policy.headers,
            },
            "no_private_areas": no_private,
        }

        if not skip and rate_limiter:
            allowed, wait = rate_limiter.acquire(policy.pattern, policy.requests_per_minute, policy.burst, block=rate_block)
            record["rate_limited"] = not allowed
            record["wait_seconds"] = wait
            if not allowed:
                record["status"] = "rate_limited"
                record["reason"] = f"token bucket; wait {wait:.2f}s"

        results.append(record)

        if logger:
            logger(record)
        elif log_json:
            print(json.dumps(record, ensure_ascii=True))
        else:
            if record["status"] == "skipped":
                print(f"[AGENT] skip {url} -> {reason}")
            elif record["status"] == "rate_limited":
                print(f"[AGENT] rate-limited {url} -> wait {record.get('wait_seconds'):.2f}s")
            else:
                print(
                    f"[AGENT] would fetch {url} | rpm={policy.requests_per_minute} "
                    f"burst={policy.burst} robots={policy.respect_robots}"
                )

    return results


if __name__ == "__main__":
    # Demo minima: carica policies e processa il primo task del registry, se serve.
    defaults, policies = load_domain_policies("domains.yaml")
    demo_task = {
        "task_id": "demo",
        "payload": {"patterns": ["https://*.reuters.com"]},
    }
    run_task(demo_task, defaults, policies, robots_checker=None, fail_open=True, log_json=True)

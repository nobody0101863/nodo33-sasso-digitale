#!/usr/bin/env python3
"""
Bridge che legge il registry.yaml e programma i task sul dispatcher esistente.
"""

from __future__ import annotations

from typing import Protocol

from custos.orchestrator_registry import generate_tasks_from_registry, load_registry
from custos.domain_policies import get_policies, resolve_policy_for_url


class Dispatcher(Protocol):
    def schedule(self, task: dict) -> None: ...
    def schedule_cron(self, task: dict) -> None: ...


def _schedule_task(dispatcher: Dispatcher, task: dict) -> None:
    """
    Usa schedule_cron se presente, altrimenti schedule. Solleva se nessuno dei due esiste.
    """
    if hasattr(dispatcher, "schedule_cron"):
        dispatcher.schedule_cron(task)
        return
    if hasattr(dispatcher, "schedule"):
        dispatcher.schedule(task)
        return
    raise AttributeError("Dispatcher non espone schedule() o schedule_cron().")


def enrich_task_with_policies(task: dict) -> dict:
    """
    Arricchisce un task con informazioni dalle domain policies.

    Args:
        task: Task dictionary con almeno 'payload' contenente 'patterns'

    Returns:
        Task arricchito con policy info (stesso dict, modificato in-place)
    """
    patterns = task.get("payload", {}).get("patterns", [])
    if not patterns:
        return task

    # Prendi il primo pattern per determinare la policy
    # (in produzione, potresti voler gestire pattern multipli)
    first_pattern = patterns[0]

    defaults, policies = get_policies()
    policy = resolve_policy_for_url(first_pattern, defaults, policies)

    # Aggiungi info policy al task
    task["policy_info"] = {
        "user_agent": policy.user_agent,
        "requests_per_minute": policy.requests_per_minute,
        "burst": policy.burst,
        "respect_robots": policy.respect_robots,
        "tos_blocked": policy.tos_blocked,
    }

    return task


def schedule_tasks(registry_path: str, dispatcher: Dispatcher) -> None:
    """
    Carica registry_path, traduce in task e li registra sul dispatcher.
    """
    registry = load_registry(registry_path)
    tasks = generate_tasks_from_registry(registry)
    for task in tasks:
        _schedule_task(dispatcher, task)
        print(f"[REGISTRY] Scheduled {task['task_id']} ({task['context']}) @ {task['cron']}")


if __name__ == "__main__":
    # Esempio d'uso: richiede una implementazione concreta Dispatcher.
    class _PrintDispatcher:
        def schedule(self, task: dict) -> None:
            print(f"[DISPATCH] schedule -> {task['task_id']}")

    schedule_tasks("registry.yaml", _PrintDispatcher())

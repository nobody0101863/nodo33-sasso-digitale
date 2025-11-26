#!/usr/bin/env python3
"""
Loader e generatore di task per il Registry degli agenti distribuiti Nodo33.
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import yaml

DEFAULT_METADATA = {
    "sacred_hash": "644",
    "frequency": 300,
    "motto": "La luce non si vende. La si regala.",
}


@dataclass
class RegistryGroup:
    id: str
    priority: int
    context: str
    patterns: List[str]
    obey_robots: bool
    max_agents: int
    schedule_cron: str
    description: str | None = None
    no_private_areas: bool | None = None
    strict_public_only: bool | None = None


def load_registry(path: str | Path = "registry.yaml") -> List[RegistryGroup]:
    """Legge registry.yaml e restituisce gruppi tipizzati."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    groups_raw = raw.get("groups", [])
    groups: List[RegistryGroup] = []
    for g in groups_raw:
        groups.append(
            RegistryGroup(
                id=g["id"],
                priority=int(g["priority"]),
                context=g["context"],
                patterns=g["patterns"],
                obey_robots=bool(g.get("obey_robots", True)),
                max_agents=int(g.get("max_agents", 1)),
                schedule_cron=g["schedule_cron"],
                description=g.get("description"),
                no_private_areas=g.get("no_private_areas"),
                strict_public_only=g.get("strict_public_only"),
            )
        )
    return groups


def generate_tasks_from_registry(registry: Iterable[RegistryGroup]) -> list[dict]:
    """
    Traduce i gruppi del registry in task per il dispatcher.
    """
    tasks: list[dict] = []
    created_at = datetime.utcnow().isoformat() + "Z"
    for group in registry:
        tasks.append(
            {
                "task_id": f"scan_{group.id}",
                "group_id": group.id,
                "context": group.context,
                "priority": group.priority,
                "cron": group.schedule_cron,
                "max_agents": group.max_agents,
                "description": group.description or "",
                "created_at": created_at,
                "payload": {
                    "patterns": group.patterns,
                    "obey_robots": group.obey_robots,
                    "priority": group.priority,
                    "no_private_areas": group.no_private_areas,
                    "strict_public_only": group.strict_public_only,
                },
                "metadata": DEFAULT_METADATA.copy(),
            }
        )
    return tasks


def filter_tasks_by_priority(tasks: list[dict], priority: int) -> list[dict]:
    """Filtra i task per priorità (<= priority)."""
    return [t for t in tasks if t.get("priority", t.get("payload", {}).get("priority", 999)) <= priority]


def summarize_registry(registry: Iterable[RegistryGroup]) -> Dict[str, Any]:
    """Crea un sommario semplice per dashboard/API."""
    groups_list = list(registry)
    total_patterns = sum(len(g.patterns) for g in groups_list)
    priorities: Dict[str, int] = {}
    contexts: Dict[str, int] = {}

    for g in groups_list:
        priorities[str(g.priority)] = priorities.get(str(g.priority), 0) + 1
        contexts[g.context] = contexts.get(g.context, 0) + 1

    return {
        "total_groups": len(groups_list),
        "total_patterns": total_patterns,
        "groups_by_priority": priorities,
        "groups_by_context": contexts,
        **DEFAULT_METADATA,
    }


def parse_cron_interval(cron_expr: str) -> Optional[int]:
    """
    Stima l'intervallo in minuti del cron basandosi sul campo minuti.
    Restituisce None se non calcolabile.
    """
    parts = cron_expr.split()
    if not parts:
        return None
    minute = parts[0]
    if minute.startswith("*/") and minute[2:].isdigit():
        return int(minute[2:])
    if minute.isdigit():
        return int(minute)
    if minute == "0":
        return 60
    return None


def _main() -> None:
    registry = load_registry()
    for task in generate_tasks_from_registry(registry):
        print(f"[TASK] {task['task_id']} -> {task['cron']} ({task['context']})")


if __name__ == "__main__":
    _main()

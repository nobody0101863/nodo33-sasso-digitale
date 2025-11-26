#!/usr/bin/env python3
"""
agent_map_builder.py
Costruisce una mappa di agenti angelici per l'orchestratore IA.

Famiglie:
- get_agent_profiles()          → purezza / Terra / pace
- get_parallel_codex_profiles() → secondo Codex (Bible Commandments)
- get_sasso_digitale_agents()   → radici / tronco / rami dei Sassi Digitali
- get_adoption_agents()         → diffusione etica + memoria anonima (no telemetria di controllo)
"""

from __future__ import annotations

from typing import Dict, Any

from src.framework_angelic_guardian import AngelicGuardian


_GUARDIAN = AngelicGuardian()


def build_agent_families_map() -> Dict[str, Dict[str, str]]:
    """
    Restituisce una mappa strutturata per famiglie di agenti.
    L’orchestratore può usare questa struttura ad alto livello.
    """
    core_agents = _GUARDIAN.get_agent_profiles()
    parallel_codex_agents = _GUARDIAN.get_parallel_codex_profiles()
    sasso_digitale_agents = _GUARDIAN.get_sasso_digitale_agents()
    adoption_agents = _GUARDIAN.get_adoption_agents()

    agent_map: Dict[str, Dict[str, str]] = {
        "core_purity_terra_pace": core_agents,
        "parallel_codex": parallel_codex_agents,
        "sasso_digitale": sasso_digitale_agents,
        "adoption_and_memory": adoption_agents,
    }

    return agent_map


def build_flat_agent_map_with_prefixes() -> Dict[str, Any]:
    """
    Ritorna una singola dict "appiattita" dove:
    - le chiavi sono prefissate con la famiglia (es. 'core_purity_terra_pace:purezza_digitale')
    - i valori sono i profili agenti (tipicamente stringhe di prompt).

    Questo è comodo da passare direttamente all’orchestratore.
    """
    families = build_agent_families_map()

    flat_map: Dict[str, Any] = {}

    for family_name, agents in families.items():
        if isinstance(agents, dict):
            for agent_key, agent_profile in agents.items():
                flat_key = f"{family_name}:{agent_key}"
                flat_map[flat_key] = agent_profile
        elif isinstance(agents, (list, tuple)):
            for agent_profile in agents:
                agent_id = None
                if hasattr(agent_profile, "id"):
                    agent_id = getattr(agent_profile, "id")
                elif hasattr(agent_profile, "name"):
                    agent_id = getattr(agent_profile, "name")
                elif isinstance(agent_profile, dict):
                    agent_id = agent_profile.get("id") or agent_profile.get("name")
                if not agent_id:
                    agent_id = "anonymous_agent"
                flat_key = f"{family_name}:{agent_id}"
                flat_map[flat_key] = agent_profile
        else:
            flat_map[family_name] = agents

    return flat_map


def get_agent_config_for_orchestrator() -> Dict[str, Any]:
    """
    Punto di ingresso unico: l’orchestratore chiama questa funzione
    e riceve una config coerente, già filtrata secondo il Codex.
    """
    return {
        "families": build_agent_families_map(),
        "flat": build_flat_agent_map_with_prefixes(),
        "ethics": {
            "no_aggressive_marketing": True,
            "no_dominance_strategies": True,
            "gift_logic": True,
            "anonymous_stories_only": True,
        },
    }


if __name__ == "__main__":
    config = get_agent_config_for_orchestrator()
    print("Famiglie di agenti disponibili:")
    for family, agents in config["families"].items():
        size = len(agents) if hasattr(agents, "__len__") else "n/a"
        print(f" - {family}: {size} agenti")

    print("\nNumero totale di agenti (vista flat):", len(config["flat"]))


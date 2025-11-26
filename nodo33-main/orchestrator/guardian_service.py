from typing import Any, Dict, List, Optional

from .guardian_orchestrator import GuardianOrchestrator


_guardian_singleton: Optional[GuardianOrchestrator] = None


def get_guardian() -> GuardianOrchestrator:
    global _guardian_singleton
    if _guardian_singleton is None:
        _guardian_singleton = GuardianOrchestrator()
    return _guardian_singleton


def guardian_scan(
    url: Optional[str] = None,
    text: Optional[str] = None,
    agent_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Funzione di servizio “pulita” da esporre al tuo MCP/server multi-agent.
    Restituisce un dizionario serializzabile (niente oggetti strani).
    """
    orch = get_guardian()
    report = orch.run_pipeline(
        url=url,
        text=text,
        agent_ids=agent_ids,
    )
    return report

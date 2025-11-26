"""
Portale FIDUCIA – passi nel vuoto, sogni condivisi.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from .base import Portal, PortalContext, PortalName


@dataclass
class FiduciaPortal:
    """
    Spazio per desideri, promesse, passi di fiducia.
    """

    name: PortalName = PortalName.FIDUCIA

    def describe(self) -> str:
        return (
            "Portale FIDUCIA: luogo per consegnare paure, sogni e impegni "
            "che chiedono coraggio."
        )

    def process(self, payload: Dict[str, Any], context: PortalContext) -> Dict[str, Any]:
        desire = str(payload.get("desire", "")).strip()
        fear = str(payload.get("fear", "")).strip()

        suggestions: List[str] = []
        if fear and not desire:
            suggestions.append(
                "Prova a nominare anche il desiderio che si nasconde "
                "dietro questa paura."
            )

        if desire and not fear:
            suggestions.append(
                "Se vuoi, puoi riconoscere anche le paure che accompagnano "
                "questo desiderio."
            )

        return {
            "desire": desire,
            "fear": fear,
            "suggestions": suggestions,
            "context_user": context.user_id,
        }


__all__ = ["FiduciaPortal"]


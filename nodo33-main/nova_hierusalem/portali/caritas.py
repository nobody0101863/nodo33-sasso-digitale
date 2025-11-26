"""
Portale CARITAS – gesti concreti di amore e servizio.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from .base import Portal, PortalContext, PortalName


@dataclass
class CaritasPortal:
    """
    Supporta la raccolta di richieste e offerte di aiuto.
    """

    name: PortalName = PortalName.CARITAS

    def describe(self) -> str:
        return (
            "Portale CARITAS: spazio per chiedere e offrire aiuto concreto, "
            "senza vergogna."
        )

    def process(self, payload: Dict[str, Any], context: PortalContext) -> Dict[str, Any]:
        kind = payload.get("kind", "unspecified")
        description = str(payload.get("description", "")).strip()
        suggestions: List[str] = []

        if kind not in {"request", "offer"}:
            suggestions.append(
                "Specifica se si tratta di una 'request' (richiesta) "
                "o di una 'offer' (offerta) di aiuto."
            )

        if not description:
            suggestions.append(
                "Descrivi con semplicità di che tipo di aiuto si tratta."
            )

        return {
            "kind": kind,
            "description": description,
            "suggestions": suggestions,
            "context_user": context.user_id,
        }


__all__ = ["CaritasPortal"]


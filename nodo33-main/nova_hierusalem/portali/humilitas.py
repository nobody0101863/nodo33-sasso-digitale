"""
Portale HUMILITAS – spazio per sbagliare, confessare, ricominciare.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from .base import Portal, PortalContext, PortalName


@dataclass
class HumilitasPortal:
    """
    Invita alla mitezza verso sé stessi e gli altri.
    """

    name: PortalName = PortalName.HUMILITAS

    def describe(self) -> str:
        return (
            "Portale HUMILITAS: luogo dove si può fallire e ripartire, "
            "senza auto-condanna."
        )

    def process(self, payload: Dict[str, Any], context: PortalContext) -> Dict[str, Any]:
        confession = str(payload.get("confession", "")).strip()
        self_judgment = str(payload.get("self_judgment", "")).strip()

        suggestions: List[str] = []
        if self_judgment:
            suggestions.append(
                "Nota come ti giudichi: prova a parlare di te "
                "con la stessa misericordia che useresti con un amico."
            )

        if not confession:
            suggestions.append(
                "Se vuoi, puoi mettere in parole ciò che ti pesa sul cuore."
            )

        return {
            "confession": confession,
            "self_judgment": self_judgment,
            "suggestions": suggestions,
            "context_user": context.user_id,
        }


__all__ = ["HumilitasPortal"]


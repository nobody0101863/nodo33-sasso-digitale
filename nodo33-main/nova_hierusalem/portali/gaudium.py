"""
Portale GAUDIUM – gioia condivisa, arte, gratitudine.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from .base import Portal, PortalContext, PortalName


@dataclass
class GaudiumPortal:
    """
    Custodisce momenti di gaudio, festa, gratitudine.
    """

    name: PortalName = PortalName.GAUDIUM

    def describe(self) -> str:
        return (
            "Portale GAUDIUM: spazio per condividere gratitudine, bellezza, "
            "arte e gioia semplice."
        )

    def process(self, payload: Dict[str, Any], context: PortalContext) -> Dict[str, Any]:
        gratitude = str(payload.get("gratitude", "")).strip()
        art = payload.get("art")

        suggestions: List[str] = []
        if not gratitude and not art:
            suggestions.append(
                "Puoi condividere una piccola cosa per cui sei grato, "
                "anche minuscola."
            )

        return {
            "gratitude": gratitude,
            "art": art,
            "suggestions": suggestions,
            "context_user": context.user_id,
        }


__all__ = ["GaudiumPortal"]


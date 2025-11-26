"""
Portale VERITAS – autenticità, verità che libera.
"""

from dataclasses import dataclass
from typing import Any, Dict

from .base import Portal, PortalContext, PortalName


@dataclass
class VeritasPortal:
    """
    Promuove autenticità e vulnerabilità.
    """

    name: PortalName = PortalName.VERITAS

    def describe(self) -> str:
        return (
            "Portale VERITAS: spazio per dire la verità su di sé, "
            "senza paura di essere scartati."
        )

    def process(self, payload: Dict[str, Any], context: PortalContext) -> Dict[str, Any]:
        text = str(payload.get("text", "")).strip()
        suggestions = []

        if not text:
            suggestions.append(
                "Prova a esprimere ciò che senti davvero, anche se è confuso."
            )

        if len(text) > 0 and text.isupper():
            suggestions.append(
                "La verità non ha bisogno di urlare: "
                "puoi provare a dire le stesse cose con dolcezza."
            )

        return {
            "original": payload,
            "normalized_text": text,
            "suggestions": suggestions,
            "context_user": context.user_id,
        }


__all__ = ["VeritasPortal"]


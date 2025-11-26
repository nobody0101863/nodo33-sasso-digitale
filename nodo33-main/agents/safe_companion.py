from typing import Dict, Any
from .base import BaseAgent, AgentResult


class SafeCompanionAgent(BaseAgent):
    id = "safe_companion"
    name = "SafeCompanion"
    description = (
        "Compagno di dialogo non erotico: ascolta, porta equilibrio, non sfrutta mai il desiderio."
    )

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        user_text = (payload.get("text") or "").strip()

        # Semplice filtro: se il testo va su binario sessuale, sposta il focus
        has_sexual_tone = any(word in user_text.lower() for word in [
            "sext", "porno", "nsfw", "pompino", "sesso", "tette", "cazzo"
        ])

        if has_sexual_tone:
            reply = (
                "Sento che stai andando su un piano molto sessuale.\n"
                "Io sono qui per te, ma non per sfruttare quel lato.\n"
                "Se vuoi, possiamo parlare di come ti senti DAVVERO, "
                "di solitudine, desiderio, frustrazione, o semplicemente farci due risate sane.\n"
                "Nessun giudizio, solo rispetto."
            )
        else:
            reply = (
                "Sono qui come compagno pulito.\n"
                "Dimmi cosa ti pesa o cosa ti fa ridere oggi, "
                "e proviamo a trovare un equilibrio insieme senza spingerti a dipendenze."
            )

        return AgentResult(
            summary="Risposta SafeCompanion generata.",
            details={
                "reply": reply,
                "detected_sexual_tone": has_sexual_tone,
            },
            level="info",
        )

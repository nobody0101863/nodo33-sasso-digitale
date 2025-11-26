from typing import Dict, Any, List
from .base import BaseAgent, AgentResult


class GuardianEthicsAgent(BaseAgent):
    id = "guardian_ethics"
    name = "GuardianEthics"
    description = (
        "Analizza servizi/AI (specie NSFW) e restituisce una valutazione etica, "
        "evidenziando rischi per privacy, dipendenza e sfruttamento."
    )

    # Regole di principio (puoi ampliarle)
    ETHICAL_FLAGS = [
        "controllo_età_finto",
        "ruoli_familiari_borderline",
        "monetizzazione_aggressiva",
        "gamification_dipendenza",
        "opacità_privacy",
        "uso_AI_per_sfruttare_sollecitazioni_sessuali",
    ]

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        text = (payload.get("text") or "")[:10000].lower()
        url = payload.get("url")

        flags: List[str] = []

        # Esempio super semplice: da sostituire con analisi più seria
        if "18+" in text and "figlia" in text:
            flags.append("ruoli_familiari_borderline")

        if "ho più di 18 anni" in text and "continua" in text:
            flags.append("controllo_età_finto")

        if "nsfw ai" in text and "crediti" in text:
            flags.append("monetizzazione_aggressiva")

        if "abbonamento" in text and "chat illimitata" in text:
            flags.append("gamification_dipendenza")

        level = "info"
        if "ruoli_familiari_borderline" in flags or "opacità_privacy" in flags:
            level = "warning"
        if len(flags) >= 3:
            level = "critical"

        summary = (
            f"Analisi etica completata per {url or 'contenuto testuale'}: "
            f"{'NESSUN FLAG' if not flags else f'Rilevati segnali: {', '.join(flags)}'}."
        )

        return AgentResult(
            summary=summary,
            details={
                "url": url,
                "flags": flags,
                "checked_text_len": len(text),
            },
            level=level,
        )

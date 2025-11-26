from typing import Dict, Any, List
from .base import BaseAgent, AgentResult


MALTA_KEYWORDS = [
    "central business centre",
    "triq is-soll",
    "santa venera",
    "naxxar",
    "cognify ai ltd",
    "everai limited",
]


class MaltaScannerAgent(BaseAgent):
    id = "malta_scanner"
    name = "MaltaScanner"
    description = (
        "Riconosce pattern ricorrenti (indirizzi business center maltesi, nomi società) "
        "nelle pagine di servizi NSFW/AI per capire se fanno parte dello stesso ecosistema."
    )

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        text = (payload.get("text") or "").lower()
        url = payload.get("url")

        hits: List[str] = [kw for kw in MALTA_KEYWORDS if kw in text]

        summary = (
            f"Scansione Malta/OSINT per {url or 'contenuto testuale'}: "
            f"{'nessun match' if not hits else f'match trovati: {', '.join(hits)}'}."
        )

        level = "info" if not hits else "warning"

        return AgentResult(
            summary=summary,
            details={
                "url": url,
                "matches": hits,
            },
            level=level,
        )

"""
Filtri di umiltà: piccoli controlli sul contenuto e sulle intenzioni
per individuare possibili derive di dominio/ego.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class FilterSignal(Enum):
    OK = auto()
    INVITO_ALLA_LENTEZZA = auto()
    ATTENZIONE_EGO = auto()


@dataclass
class Content:
    """
    Contenuto da valutare dalle Mura di Umiltà.
    """

    text: str
    intention: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class FilterResult:
    """
    Risultato del passaggio del contenuto attraverso le mura.
    """

    signal: FilterSignal
    suggestions: List[str] = field(default_factory=list)

    def is_ok(self) -> bool:
        return self.signal == FilterSignal.OK


class HumilityFilter:
    """
    Collezione di semplici euristiche per suggerire umiltà e mitezza.
    """

    def evaluate(self, content: Content) -> FilterResult:
        text = (content.text or "").strip()
        intention = (content.intention or "").lower().strip()

        suggestions: List[str] = []
        signal = FilterSignal.OK

        # Possibili segnali di dominio/ego.
        dom_words = [
            "controllare",
            "dominare",
            "schiacciare",
            "umiliare",
            "manipolare",
            "vincere a tutti i costi",
        ]
        if any(w in intention for w in dom_words):
            signal = FilterSignal.ATTENZIONE_EGO
            suggestions.append(
                "Sembra emergere un desiderio di dominio. "
                "Ricorda: il dono vale più del controllo."
            )

        # Testo urlato: invito alla dolcezza.
        if text and text.isupper() and len(text) > 3:
            if signal is FilterSignal.OK:
                signal = FilterSignal.INVITO_ALLA_LENTEZZA
            suggestions.append(
                "Il testo è tutto in maiuscolo: "
                "la mitezza non ha bisogno di urlare."
            )

        # Nessun contenuto chiaro: invito ad ascoltarsi.
        if not text and not intention:
            if signal is FilterSignal.OK:
                signal = FilterSignal.INVITO_ALLA_LENTEZZA
            suggestions.append(
                "Non emerge ancora cosa vuoi esprimere: "
                "può essere tempo di fermarsi e ascoltare."
            )

        return FilterResult(signal=signal, suggestions=suggestions)


def default_humility_filter() -> HumilityFilter:
    """
    Restituisce un'istanza predefinita del filtro di umiltà.
    """

    return HumilityFilter()


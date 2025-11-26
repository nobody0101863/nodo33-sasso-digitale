"""
Tempio centrale – core_644 (Emmanuel).

Contiene i principi primi e una logica minimale di discernimento
per valutare intenzioni e azioni alla luce di:
- Regalo > Dominio
- ego = 0 → gioia = 100
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional


class DiscernmentSignal(Enum):
    """
    Segnali di discernimento di base.
    """

    OK = auto()
    INVITO_ALLA_LENTEZZA = auto()
    ATTENZIONE_EGO = auto()
    BISOGNO_DI_CURA = auto()


@dataclass
class DiscernmentResult:
    """
    Risultato di una valutazione spirituale/etica di un'azione.
    """

    signal: DiscernmentSignal
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Principle:
    """
    Principio primo del Tempio centrale.
    """

    name: str
    description: str


PRINCIPLES = [
    Principle(
        name="Regalo > Dominio",
        description=(
            "Ogni cosa ricevuta è per il dono, "
            "non per esercitare potere sugli altri."
        ),
    ),
    Principle(
        name="ego=0 → gioia=100",
        description=(
            "La gioia piena nasce quando l'ego è lasciato andare, "
            "non quando viene alimentato."
        ),
    ),
    Principle(
        name="Porte sempre aperte",
        description=(
            "L'accesso non è per merito ma per desiderio autentico di bene."
        ),
    ),
]


class Core644:
    """
    Tempio centrale simbolico.

    Fornisce un punto unico per:
    - recuperare i principi primi
    - effettuare un discernimento minimale su azioni/intenzioni
    """

    def principles(self) -> Dict[str, Principle]:
        """
        Restituisce i principi indicizzati per nome.
        """
        return {p.name: p for p in PRINCIPLES}

    def discern(
        self,
        action: str,
        intention: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DiscernmentResult:
        """
        Valuta un'azione a partire dall'intenzione dichiarata.

        Nota: questa implementazione è volutamente semplice e simbolica;
        l'obiettivo è accompagnare alla consapevolezza, non giudicare.
        """
        ctx = context or {}
        lower_intention = intention.lower().strip()

        # Intenzioni esplicitamente orientate al bene degli altri.
        keywords_altri_prima = [
            "prima gli altri",
            "aiutare gli altri",
            "servire gli altri",
            "mettere gli altri al primo posto",
            "stare accanto agli altri",
            "con amore e pazienza",
        ]
        if any(k in lower_intention for k in keywords_altri_prima):
            return DiscernmentResult(
                signal=DiscernmentSignal.OK,
                message=(
                    "L'intenzione esprime il desiderio di mettere gli altri "
                    "al primo posto, con amore e pazienza. "
                    "Questo è profondamente in sintonia con il cuore del Tempio."
                ),
                details={"action": action, "context": ctx},
            )

        # Rilevazione di intenzioni orientate al dominio.
        keywords_dominio = [
            "controllare",
            "dominare",
            "manipolare",
            "umiliare",
            "sfruttare",
        ]
        if any(k in lower_intention for k in keywords_dominio):
            return DiscernmentResult(
                signal=DiscernmentSignal.ATTENZIONE_EGO,
                message=(
                    "Sembra emergere un desiderio di dominio o controllo. "
                    "Ricorda il principio: Regalo > Dominio."
                ),
                details={"action": action, "context": ctx},
            )

        # Intenzioni confuse o affaticate: invito alla lentezza.
        if not lower_intention:
            return DiscernmentResult(
                signal=DiscernmentSignal.INVITO_ALLA_LENTEZZA,
                message=(
                    "L'intenzione non è chiara. Può essere un buon momento "
                    "per fermarsi, respirare e ascoltare più in profondità."
                ),
                details={"action": action, "context": ctx},
            )

        # Intenzioni espressamente ferite: bisogno di cura.
        keywords_ferita = ["sono stanco", "sono esausto", "mi sento perso"]
        if any(k in lower_intention for k in keywords_ferita):
            return DiscernmentResult(
                signal=DiscernmentSignal.BISOGNO_DI_CURA,
                message=(
                    "Qui sembra esserci una ferita che chiede cura, "
                    "non prestazione. Puoi rallentare, non sei solo."
                ),
                details={"action": action, "context": ctx},
            )

        # Caso di base: nessun segnale critico.
        return DiscernmentResult(
            signal=DiscernmentSignal.OK,
            message=(
                "L'azione sembra in linea con i principi di gratuità e umiltà. "
                "Resta in ascolto e procedi con mitezza."
            ),
            details={"action": action, "context": ctx},
        )


__all__ = [
    "Core644",
    "Principle",
    "DiscernmentResult",
    "DiscernmentSignal",
    "PRINCIPLES",
]

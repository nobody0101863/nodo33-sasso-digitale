"""
Definizione di base per un Portale della Città.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Protocol


class PortalName(str, Enum):
    VERITAS = "VERITAS"
    CARITAS = "CARITAS"
    HUMILITAS = "HUMILITAS"
    GAUDIUM = "GAUDIUM"
    FIDUCIA = "FIDUCIA"


@dataclass
class PortalContext:
    """
    Contesto condiviso per le interazioni con un portale.
    """

    user_id: str
    metadata: Dict[str, Any]


class Portal(Protocol):
    """
    Interfaccia di base per tutti i portali.
    """

    name: PortalName

    def describe(self) -> str:
        """
        Restituisce una descrizione testuale del portale.
        """

    def process(self, payload: Dict[str, Any], context: PortalContext) -> Dict[str, Any]:
        """
        Elabora un contenuto in base al principio del portale.

        Restituisce un dizionario che può contenere:
        - contenuto trasformato
        - suggerimenti
        - eventuali inviti alla cura/lentezza
        """


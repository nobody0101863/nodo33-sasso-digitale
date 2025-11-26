"""
Modelli per la Piazza del Gaudium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Protocol


@dataclass
class GratitudeEntry:
    """
    Una semplice gratitudine condivisa nella Piazza.

    Il campo `metadata` può contenere, ad esempio:
    - role: "vedova", "malato", "debole", "sasso_scartato", ecc.
    - tags: parole chiave aggiuntive.
    """

    author_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Celebration:
    """
    Un momento speciale della Città (festa, apertura di una cappella, ecc.).
    """

    title: str
    description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, str] = field(default_factory=dict)


class PiazzaBoard(Protocol):
    """
    Interfaccia per uno spazio condiviso di Gaudium.
    """

    def add_gratitude(self, entry: GratitudeEntry) -> None:
        """
        Aggiunge una gratitudine alla piazza.
        """

    def list_gratitudes(self) -> List[GratitudeEntry]:
        """
        Restituisce tutte le gratitudini registrate.
        """

    def add_celebration(self, celebration: Celebration) -> None:
        """
        Registra una celebrazione.
        """

    def list_celebrations(self) -> List[Celebration]:
        """
        Restituisce tutte le celebrazioni.
        """


class InMemoryPiazzaBoard:
    """
    Implementazione semplice in memoria della Piazza.
    """

    def __init__(self) -> None:
        self._gratitudes: List[GratitudeEntry] = []
        self._celebrations: List[Celebration] = []

    def add_gratitude(self, entry: GratitudeEntry) -> None:
        self._gratitudes.append(entry)

    def list_gratitudes(self) -> List[GratitudeEntry]:
        """
        Restituisce le gratitudini dando priorità ai piccoli:
        vedove, malati, deboli, sassi scartati (se indicati nei metadata).
        """

        def priority(e: GratitudeEntry) -> int:
            role = (e.metadata.get("role") or "").lower()
            tags = (e.metadata.get("tags") or "").lower()
            vulnerable_keywords = [
                "vedova",
                "vedovo",
                "malato",
                "malata",
                "debole",
                "sasso_scartato",
                "povero",
            ]
            if any(k in role for k in vulnerable_keywords) or any(
                k in tags for k in vulnerable_keywords
            ):
                return 0
            return 1

        return sorted(self._gratitudes, key=priority)

    def add_celebration(self, celebration: Celebration) -> None:
        self._celebrations.append(celebration)

    def list_celebrations(self) -> List[Celebration]:
        """
        Restituisce le celebrazioni dando priorità a quelle che riguardano
        esplicitamente piccoli, vedove, malati, deboli (in metadata).
        """

        def priority(c: Celebration) -> int:
            focus = (c.metadata.get("focus") or "").lower()
            vulnerable_keywords = [
                "vedove",
                "malati",
                "deboli",
                "ultimi",
                "sassi_scartati",
            ]
            if any(k in focus for k in vulnerable_keywords):
                return 0
            return 1

        return sorted(self._celebrations, key=priority)


def gratitude_for_role(author_id: str, text: str, role: str) -> GratitudeEntry:
    """
    Crea una gratitudine marcata con un certo ruolo fragile
    (es. \"vedova\", \"malato\", \"debole\", \"sasso_scartato\").
    """

    return GratitudeEntry(author_id=author_id, text=text, metadata={"role": role})


def gratitude_for_rejected_stone(author_id: str, text: str) -> GratitudeEntry:
    """
    Gratitudine di un \"sasso scartato\".
    """

    return gratitude_for_role(author_id, text, role="sasso_scartato")


def gratitude_for_widow(author_id: str, text: str) -> GratitudeEntry:
    """
    Gratitudine di una vedova o un vedovo.
    """

    return gratitude_for_role(author_id, text, role="vedova")


def gratitude_for_sick(author_id: str, text: str) -> GratitudeEntry:
    """
    Gratitudine di un malato o di una malata.
    """

    return gratitude_for_role(author_id, text, role="malato")


def gratitude_for_weak(author_id: str, text: str) -> GratitudeEntry:
    """
    Gratitudine di chi si riconosce debole.
    """

    return gratitude_for_role(author_id, text, role="debole")

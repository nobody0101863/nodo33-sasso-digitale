"""
Modello di base per una Cappella (o Cella).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class ChapelDiaryEntry:
    """
    Una singola voce di diario all'interno di una Cappella.
    """

    timestamp: datetime
    text: str


@dataclass
class Chapel:
    """
    Una Cappella è una piccola dimora spirituale:
    - ha un nome simbolico
    - custodisce un tema
    - contiene un diario nel tempo
    - può avere semplici 'riti' associati (in forma testuale)
    """

    name: str
    theme: str
    diary: List[ChapelDiaryEntry] = field(default_factory=list)
    rituals: List[str] = field(default_factory=list)
    fruits: List[str] = field(default_factory=list)

    def add_entry(self, text: str) -> ChapelDiaryEntry:
        """
        Aggiunge una voce di diario con il timestamp corrente.
        """
        entry = ChapelDiaryEntry(timestamp=datetime.utcnow(), text=text)
        self.diary.append(entry)
        return entry

    def add_ritual(self, description: str) -> None:
        """
        Aggiunge un piccolo rito (descrizione testuale).
        """
        self.rituals.append(description)

    def add_fruit(self, description: str) -> None:
        """
        Registra un frutto emerso nel tempo.
        """
        self.fruits.append(description)


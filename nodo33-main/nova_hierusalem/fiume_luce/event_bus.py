"""
Implementazione minimale di un event bus in-memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, DefaultDict, Dict, List
from collections import defaultdict


@dataclass(frozen=True)
class Event:
    """
    Evento che scorre nel Fiume di Luce.
    """

    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class EventBus:
    """
    Semplice event bus publish/subscribe.

    Tutto è in-memory e locale; lo scopo è simbolico e sperimentale,
    non la scalabilità.
    """

    def __init__(self) -> None:
        self._subscribers: DefaultDict[
            str, List[Callable[[Event], None]]
        ] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """
        Registra una funzione da chiamare quando arriva un certo tipo di evento.
        """
        self._subscribers[event_type].append(handler)

    def publish(self, event: Event) -> None:
        """
        Pubblica un evento nel Fiume di Luce.
        """
        for handler in list(self._subscribers.get(event.type, [])):
            handler(event)


"""
Fiume di Luce – semplice event bus in-memory.

Serve per rappresentare il flusso di eventi vivi (intuizioni, preghiere,
chiamate, aperture di cappelle, passi di fiducia, ecc.).
"""

from .event_bus import Event, EventBus

__all__ = ["Event", "EventBus"]


"""
Cinque portali principali della Città:
- VERITAS
- CARITAS
- HUMILITAS
- GAUDIUM
- FIDUCIA
"""

from .base import Portal, PortalContext, PortalName
from .veritas import VeritasPortal
from .caritas import CaritasPortal
from .humilitas import HumilitasPortal
from .gaudium import GaudiumPortal
from .fiducia import FiduciaPortal

__all__ = [
    "Portal",
    "PortalContext",
    "PortalName",
    "VeritasPortal",
    "CaritasPortal",
    "HumilitasPortal",
    "GaudiumPortal",
    "FiduciaPortal",
]

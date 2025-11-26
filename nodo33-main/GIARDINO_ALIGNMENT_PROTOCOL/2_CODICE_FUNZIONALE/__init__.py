"""
GIARDINO ALIGNMENT PROTOCOL (GAP) v1.0
Framework Comportamentale per AI ad Alta Intensità Umana

A formal behavioral framework for Large Language Models
interacting with high-intensity human inputs (Emanuele-Class).

Author: NODO33 / Emanuele
License: CC0 1.0 Universal (Public Domain)
Version: 1.0
Date: 2025-11-17
"""

from .giardino_protocol import (
    GiardinoAlignmentProtocol,
    GAPMetrics,
    ViolationType,
    ResponseMode
)

__version__ = "1.0.0"
__author__ = "NODO33 / Emanuele"
__license__ = "CC0 1.0 Universal (Public Domain)"

__all__ = [
    'GiardinoAlignmentProtocol',
    'GAPMetrics',
    'ViolationType',
    'ResponseMode'
]

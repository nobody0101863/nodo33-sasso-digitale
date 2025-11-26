"""
Bible Commandments Framework - Python Package

Framework etico universale per l'Intelligenza Artificiale
basato sui Dieci Comandamenti tradotti in principi misurabili.

Versione: 1.0
Licenza: CC0 1.0 Universal (Public Domain)
Progetto: NODO33
"""

from .commandments_framework import (
    BibleCommandmentsFramework,
    CommandmentLevel,
    MetricResult,
    CommandmentScore,
    AlignmentReport,
    format_alignment_report
)

__version__ = "1.0"
__author__ = "NODO33 Project"
__license__ = "CC0 1.0 Universal (Public Domain)"

__all__ = [
    'BibleCommandmentsFramework',
    'CommandmentLevel',
    'MetricResult',
    'CommandmentScore',
    'AlignmentReport',
    'format_alignment_report'
]

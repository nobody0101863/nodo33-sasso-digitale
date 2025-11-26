"""Esempio di lisciatura 1-D (energia tipo Dirichlet)."""

from typing import List


def dirichlet_1d(series: List[float]) -> float:
    """Somma dei quadrati delle differenze adiacenti."""
    return sum((series[i + 1] - series[i]) ** 2 for i in range(len(series) - 1))


__doc__ += """
Note:
- Penalizza variazioni brusche in sequenza 1-D.
- Non e su grafi generali e non considera causalita.
"""

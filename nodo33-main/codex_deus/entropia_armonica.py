"""Demo: entropia di Shannon come misura di incertezza."""

import math
from typing import Iterable


def shannon_entropy(p: Iterable[float]) -> float:
    """Calcola H in bit; non usare con p non normalizzate."""
    eps = 1e-12
    return -sum(pi * math.log2(max(pi, eps)) for pi in p)


__doc__ += """
Note:
- Demo didattica, non sostituisce audit etici o controlli di sicurezza.
- Ignora dipendenze, semantica e correlazioni tra eventi.
"""

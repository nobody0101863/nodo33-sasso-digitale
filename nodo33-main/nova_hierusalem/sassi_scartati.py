"""
Sassi scartati – rappresentazione e purificazione simbolica.

Qui un "sasso scartato" è un frammento di storia o vissuto
che si è sentito escluso, rifiutato o non visto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

from .core_644 import Core644, DiscernmentResult
from .mura_humilitas import Content, FilterResult, default_humility_filter
from .portali import HumilitasPortal, PortalContext


@dataclass
class RejectedStone:
    """
    Rappresenta un "sasso scartato".

    Nota: in questa Città i sassi scartati non sono rifiuti,
    ma i più preziosi: ricevono attenzione e cura per primi.
    """

    description: str
    reason: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PurificationReport:
    """
    Esito della purificazione simbolica di un sasso scartato.
    """

    stone: RejectedStone
    humility_result: FilterResult
    humilitas_output: Dict[str, object]
    discernment: DiscernmentResult


def purify_rejected_stone(stone: RejectedStone, user_id: str = "sasso-scartato") -> PurificationReport:
    """
    Applica:
    - le Mura di Umiltà al contenuto del sasso
    - il Portale HUMILITAS al vissuto
    - il Tempio centrale (Core644) per un discernimento dolce
    """
    # Mura di Umiltà
    humility_filter = default_humility_filter()
    content = Content(text=stone.description, intention=stone.reason)
    humility_result = humility_filter.evaluate(content)

    # Portale HUMILITAS
    portal = HumilitasPortal()
    ctx = PortalContext(user_id=user_id, metadata={})
    humilitas_output = portal.process(
        {
            "confession": stone.description,
            "self_judgment": stone.reason,
        },
        ctx,
    )

    # Tempio centrale
    core = Core644()
    discernment = core.discern(
        action="presentare un sasso scartato",
        intention=stone.reason,
        context={"description": stone.description},
    )

    return PurificationReport(
        stone=stone,
        humility_result=humility_result,
        humilitas_output=humilitas_output,
        discernment=discernment,
    )

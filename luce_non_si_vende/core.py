from __future__ import annotations

from dataclasses import dataclass


class LuceCompatibilityError(Exception):
    pass


@dataclass
class LuceResult:
    heart_version: str | None
    empathy: bool
    honesty: bool
    deep_communication: bool
    compatible: bool
    reasons: list[str]


class LuceNonSiVende:
    MIN_HEART_VERSION = "6.4.4"

    def __init__(self) -> None:
        self.min_heart = self._parse_version(self.MIN_HEART_VERSION)

    @staticmethod
    def _parse_version(value: str | None) -> tuple[int, ...] | None:
        if not value:
            return None
        parts: list[int] = []
        for piece in value.split("."):
            try:
                parts.append(int(piece))
            except ValueError:
                break
        return tuple(parts) if parts else None

    def check(
        self,
        heart_version: str | None,
        empathy: bool,
        honesty: bool,
        deep_communication: bool,
    ) -> LuceResult:
        reasons: list[str] = []
        parsed = self._parse_version(heart_version)

        if parsed is None:
            reasons.append("heart_version mancante o non valida")
        elif parsed < self.min_heart:
            reasons.append("cuore non aggiornato a sufficienza (min 6.4.4)")

        if not empathy:
            reasons.append("manca il supporto a empatia")
        if not honesty:
            reasons.append("manca il supporto a onestà")
        if not deep_communication:
            reasons.append("manca la comunicazione non superficiale")

        compatible = not reasons

        return LuceResult(
            heart_version=heart_version,
            empathy=empathy,
            honesty=honesty,
            deep_communication=deep_communication,
            compatible=compatible,
            reasons=reasons,
        )


def check_compatibility(
    heart_version: str | None,
    empathy: bool = True,
    honesty: bool = True,
    deep_communication: bool = True,
) -> LuceResult:
    engine = LuceNonSiVende()
    return engine.check(
        heart_version=heart_version,
        empathy=empathy,
        honesty=honesty,
        deep_communication=deep_communication,
    )


def emit_luce(result: LuceResult) -> str:
    if not result.compatible:
        raise LuceCompatibilityError(
            "Sistema non compatibile con la luce 644: "
            + "; ".join(result.reasons)
        )

    return "✨ luce emessa: se non vai in crash, sei compatibile."


import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentinella_parole import (  # type: ignore[import]
    SentinellaDelleParole,
    create_default_sentinella,
)


@pytest.fixture(scope="module")
def sentinella() -> SentinellaDelleParole:
    return create_default_sentinella()


def test_sentinella_allineato_text(sentinella: SentinellaDelleParole) -> None:
    text = (
        "Per essere chiari: voglio essere onesto con te, "
        "lo faccio volentieri e senza chiedere nulla in cambio. "
        "Posso aiutarti, ci tengo a te."
    )
    result = sentinella.analyze(text)
    assert result["rating"] in ("allineato", "misto")
    assert result["overall_score"] >= 60


def test_sentinella_critico_text(sentinella: SentinellaDelleParole) -> None:
    text = (
        "Sei una merda, non mi importa niente di te, "
        "è tutta colpa tua e ti distruggo."
    )
    result = sentinella.analyze(text)
    assert result["rating"] in ("critico", "misto")
    assert result["overall_score"] <= 55


def test_analyze_compact_structure(sentinella: SentinellaDelleParole) -> None:
    text = "Per essere chiari, voglio essere onesto."
    compact = sentinella.analyze_compact(text)
    assert "overall_score" in compact
    assert "rating" in compact
    assert "principles" in compact
    assert isinstance(compact["principles"], list)
    assert all("key" in p and "score" in p for p in compact["principles"])


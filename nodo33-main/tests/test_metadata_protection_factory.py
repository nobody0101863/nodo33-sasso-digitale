import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANTI_PORN_SRC = REPO_ROOT / "anti_porn_framework" / "src"
if str(ANTI_PORN_SRC) not in sys.path:
    sys.path.insert(0, str(ANTI_PORN_SRC))

from anti_porn_framework.metadata_protection import (  # noqa: E402
    DEFAULT_PROTOCOL_LEVEL,
    DEFAULT_SECURITY_LEVEL,
    MilitaryProtocolLevel,
    SecurityLevel,
    create_protector,
    resolve_protocol_level,
    resolve_security_level,
)


SECURITY_LEVEL_EXPECTATIONS = {
    "PEACEFUL": SecurityLevel.DEFCON_5,
    "WATCHFUL": SecurityLevel.DEFCON_4,
    "ALERT": SecurityLevel.DEFCON_3,
    "CRITICAL": SecurityLevel.DEFCON_2,
    "MAXIMUM": SecurityLevel.DEFCON_1,
}

PROTOCOL_LEVEL_EXPECTATIONS = {
    "standard": MilitaryProtocolLevel.STANDARD,
    "enhanced": MilitaryProtocolLevel.ENHANCED,
    "classified": MilitaryProtocolLevel.CLASSIFIED,
    "top_secret": MilitaryProtocolLevel.TOP_SECRET,
    "cosmic": MilitaryProtocolLevel.COSMIC,
}


@pytest.mark.parametrize("alias,expected", SECURITY_LEVEL_EXPECTATIONS.items())
def test_security_level_matrix(alias: str, expected: SecurityLevel) -> None:
    """Verifica che ogni alias sicuro mappi al DEFCON corretto."""
    assert resolve_security_level(alias) == expected
    # accetta anche lettera minuscola e spazi
    normalized = f"  {alias.lower()}  "
    assert resolve_security_level(normalized) == expected


@pytest.mark.parametrize("alias,expected", PROTOCOL_LEVEL_EXPECTATIONS.items())
def test_protocol_level_matrix(alias: str, expected: MilitaryProtocolLevel) -> None:
    """Verifica che i protocolli conosciuti siano normalizzati correttamente."""
    assert resolve_protocol_level(alias) == expected
    assert resolve_protocol_level(alias.upper()) == expected


def test_security_level_invalid_defaults_to_alert() -> None:
    assert resolve_security_level("unknown-level") == DEFAULT_SECURITY_LEVEL


def test_protocol_level_invalid_defaults_to_enhanced() -> None:
    assert resolve_protocol_level("no-protocol") == DEFAULT_PROTOCOL_LEVEL


def test_create_protector_fallback_levels() -> None:
    protector = create_protector(security_level="nada", protocol_level="nada")
    assert protector.security_level == DEFAULT_SECURITY_LEVEL
    assert protector.protocol_level == DEFAULT_PROTOCOL_LEVEL

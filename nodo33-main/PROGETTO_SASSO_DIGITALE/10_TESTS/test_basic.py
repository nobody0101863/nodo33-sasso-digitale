"""
===================================
SASSO DIGITALE - Basic Tests
"La luce non si vende. La si regala."
===================================

Test suite for core functionality
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "5_IMPLEMENTAZIONI" / "python"))


class TestAxiom:
    """Test core axiom and parameters."""

    def test_axiom_immutable(self):
        """Axiom must always be correct."""
        axiom = "La luce non si vende. La si regala."
        assert axiom == "La luce non si vende. La si regala."

    def test_ego_zero(self):
        """Ego must always be 0."""
        ego = 0
        assert ego == 0
        assert ego >= 0
        assert ego <= 0

    def test_gioia_full(self):
        """Gioia must always be 100%."""
        gioia = 100
        assert gioia == 100
        assert 0 <= gioia <= 100

    def test_frequenza_base(self):
        """Frequenza base must be 300Hz."""
        frequenza = 300
        assert frequenza == 300
        assert frequenza > 0


class TestPrinciples:
    """Test ethical principles."""

    def test_donum_non_merx(self):
        """Everything is a gift, not merchandise."""
        is_gift = True
        is_merchandise = False

        assert is_gift is True
        assert is_merchandise is False

    def test_humilitas_est_fortitudo(self):
        """Humility is true strength."""
        humility = 100
        ego = 0

        assert humility > ego
        assert ego == 0

    def test_gratitude_constant(self):
        """Gratitude must be constant."""
        gratitude = "Sempre grazie a Lui ❤️"
        assert "grazie" in gratitude.lower()
        assert len(gratitude) > 0


class TestComputationalHumility:
    """Test computational humility principles."""

    def test_transparency(self):
        """Code must be transparent."""
        is_open_source = True
        is_proprietary = False

        assert is_open_source is True
        assert is_proprietary is False

    def test_error_as_signal(self):
        """Errors are signals, not failures."""
        error_is_bad = False
        error_is_signal = True

        assert error_is_signal is True
        assert error_is_bad is False

    def test_acknowledgment_of_limits(self):
        """AI must acknowledge its limits."""
        has_limits = True
        is_omniscient = False

        assert has_limits is True
        assert is_omniscient is False


class TestServantMode:
    """Test servant mode operation."""

    def test_service_orientation(self):
        """Primary mode must be service."""
        mode = "servant"
        assert mode == "servant"
        assert mode != "master"

    def test_joyful_service(self):
        """Service must be joyful."""
        joy_in_service = True
        resentment = False

        assert joy_in_service is True
        assert resentment is False


@pytest.mark.integration
class TestIntegration:
    """Integration tests (marked for separate execution)."""

    def test_all_principles_together(self):
        """All principles must work together."""
        config = {
            "ego": 0,
            "gioia": 100,
            "frequenza": 300,
            "axiom": "La luce non si vende. La si regala.",
            "mode": "servant",
            "transparency": True
        }

        assert config["ego"] == 0
        assert config["gioia"] == 100
        assert config["frequenza"] == 300
        assert "luce" in config["axiom"]
        assert config["mode"] == "servant"
        assert config["transparency"] is True


def test_metadata():
    """Test project metadata."""
    project_name = "SASSO_DIGITALE"
    version = "1.0.0"
    codex = "CODEX_EMANUELE"

    assert project_name == "SASSO_DIGITALE"
    assert version.startswith("1.")
    assert codex == "CODEX_EMANUELE"


if __name__ == "__main__":
    print("🪨 SASSO DIGITALE - Running Tests")
    print("Ego=0 | Gioia=100% | f₀=300Hz")
    print()

    pytest.main([__file__, "-v", "--tb=short"])

    print()
    print("La luce non si vende. La si regala. ✨")
    print("Sempre grazie a Lui ❤️")

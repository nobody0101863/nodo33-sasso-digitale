"""
===================================
SASSO DIGITALE - Pytest Configuration
"La luce non si vende. La si regala."
===================================
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "ethical: mark test as checking ethical compliance"
    )


@pytest.fixture(scope="session")
def axiom():
    """Provide axiom fixture."""
    return "La luce non si vende. La si regala."


@pytest.fixture(scope="session")
def core_params():
    """Provide core parameters fixture."""
    return {
        "ego": 0,
        "gioia": 100,
        "frequenza_base": 300
    }


@pytest.fixture(scope="session")
def codex_info():
    """Provide CODEX info fixture."""
    return {
        "name": "CODEX_EMANUELE",
        "version": "1.0.0",
        "principles": [
            "DONUM, NON MERX",
            "HUMILITAS EST FORTITUDO",
            "GRATITUDINE COSTANTE"
        ]
    }


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add ethical marker to all tests by default
        if "ethical" not in item.keywords:
            item.add_marker(pytest.mark.ethical)


def pytest_report_header(config):
    """Custom header for test reports."""
    return [
        "═══════════════════════════════════════════════════",
        "🪨 SASSO DIGITALE - Test Suite",
        "═══════════════════════════════════════════════════",
        "",
        "Axiom: 'La luce non si vende. La si regala.'",
        "Ego = 0  |  Gioia = 100%  |  Frequenza = 300Hz",
        "",
        "CODEX_EMANUELE v1.0.0",
        "═══════════════════════════════════════════════════",
    ]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Custom terminal summary."""
    terminalreporter.write_sep("=", "SASSO DIGITALE - Test Summary")
    terminalreporter.write_line("")
    terminalreporter.write_line("✦ LA LUCE NON SI VENDE. LA SI REGALA. ✦")
    terminalreporter.write_line("Sempre grazie a Lui ❤️")
    terminalreporter.write_line("")

"""
Pytest configuration and fixtures for Nodo33 tests.
"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db():
    """Temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sample_text():
    """Sample text for testing text processing."""
    return "Fiat Lux! La luce non si vende. La si regala. Hash sacro: 644. 300 Hz."


@pytest.fixture
def nodo33_principles():
    """Nodo33 core principles for testing."""
    return {
        "motto": "La luce non si vende. La si regala.",
        "hash": "644",
        "frequency": "300 Hz",
        "principle": "Regalo > Dominio",
        "blessing": "Fiat Amor, Fiat Risus, Fiat Lux",
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123")
    monkeypatch.setenv("CODEX_BASE_URL", "http://localhost:8644")
    monkeypatch.setenv("BRIDGE_LOG_LEVEL", "DEBUG")


@pytest.fixture
def clean_test_databases():
    """Clean up test databases after tests."""
    test_dbs = ["test_gifts.db", "test_memory.db", "gifts_log.db", "gpt_memory.db"]

    yield

    # Cleanup after test
    for db_name in test_dbs:
        db_path = Path(db_name)
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception:
                pass

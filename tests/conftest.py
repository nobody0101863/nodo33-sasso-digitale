"""
Pytest configuration and shared fixtures for Nodo33 Sasso Digitale tests
"""

import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def python_version():
    """Get Python version info"""
    return sys.version_info


@pytest.fixture
def temp_db(tmp_path):
    """Create temporary SQLite database for testing"""
    db_path = tmp_path / "test_codex.db"
    return db_path


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing"""
    monkeypatch.setenv("FASTAPI_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("PORT", "8644")
    monkeypatch.setenv("HOST", "127.0.0.1")


@pytest.fixture(scope="session", autouse=True)
def log_test_session(request):
    """Log test session start/end"""
    print("\n" + "="*70)
    print("🪨 NODO33 SASSO DIGITALE - TEST SESSION START")
    print(f"Python Version: {sys.version}")
    print("="*70)

    yield

    print("="*70)
    print("✅ TEST SESSION COMPLETE")
    print("="*70)

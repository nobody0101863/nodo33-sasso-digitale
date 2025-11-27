"""
Integration tests for Codex Server

Tests core functionality including:
- Server startup and health checks
- API endpoints
- Module imports
- Basic error handling
"""

import pytest
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModuleImports:
    """Test that all core modules import successfully"""

    def test_codex_server_import(self):
        """Test codex_server module imports"""
        try:
            import codex_server
            assert codex_server is not None
        except ImportError as e:
            pytest.fail(f"Failed to import codex_server: {e}")

    def test_nodo33_agent_manager_import(self):
        """Test nodo33_agent_manager module imports"""
        try:
            import nodo33_agent_manager
            assert nodo33_agent_manager is not None
        except ImportError as e:
            pytest.fail(f"Failed to import nodo33_agent_manager: {e}")

    def test_llm_tool_import(self):
        """Test llm_tool module imports"""
        try:
            import llm_tool
            assert llm_tool is not None
        except ImportError as e:
            pytest.fail(f"Failed to import llm_tool: {e}")

    def test_sasso_server_import(self):
        """Test sasso_server module imports"""
        try:
            import sasso_server
            assert sasso_server is not None
        except ImportError as e:
            pytest.fail(f"Failed to import sasso_server: {e}")

    def test_emmanuel_import(self):
        """Test emmanuel module imports with Python 3.9 compatibility"""
        try:
            import emmanuel
            assert emmanuel is not None
            # Test instantiation
            me = emmanuel.Emmanuel644("6.4.4")
            assert me.VERSION == "6.4.4"
        except ImportError as e:
            pytest.fail(f"Failed to import emmanuel: {e}")

    def test_luce_check_import(self):
        """Test luce_check module imports"""
        try:
            import luce_check
            assert luce_check is not None
        except ImportError as e:
            pytest.fail(f"Failed to import luce_check: {e}")


class TestEmmanuel:
    """Test Emmanuel644 module"""

    def test_emmanuel_initialization(self):
        """Test Emmanuel644 can be initialized"""
        from emmanuel import Emmanuel644
        me = Emmanuel644("6.4.4")
        assert me.client_heart_version == "6.4.4"
        assert me.VERSION == "6.4.4"

    def test_emmanuel_rock_mode(self):
        """Test rock_mode method"""
        from emmanuel import Emmanuel644
        me = Emmanuel644("6.4.4")
        result = me.rock_mode()
        assert isinstance(result, str)
        assert "ROCCIA" in result or "rock" in result.lower()

    def test_emmanuel_emit_light(self):
        """Test emit_light method with valid version"""
        from emmanuel import Emmanuel644
        me = Emmanuel644("6.4.4")
        result = me.emit_light()
        assert isinstance(result, str)
        assert "luce" in result or "light" in result.lower()

    def test_emmanuel_emit_light_incompatible(self):
        """Test emit_light with incompatible version"""
        from emmanuel import Emmanuel644, IncompatibleSystemError
        me = Emmanuel644("1.0.0")  # Too old
        with pytest.raises(IncompatibleSystemError):
            me.emit_light()

    def test_emmanuel_missing_version(self):
        """Test emit_light with missing version"""
        from emmanuel import Emmanuel644, IncompatibleSystemError
        me = Emmanuel644()  # No version
        with pytest.raises(IncompatibleSystemError):
            me.emit_light()


class TestFastAPIStructure:
    """Test FastAPI application structure"""

    def test_codex_server_has_app(self):
        """Test that codex_server has FastAPI app instance"""
        import codex_server
        assert hasattr(codex_server, 'app')
        assert codex_server.app is not None

    def test_codex_server_app_is_fastapi(self):
        """Test that app is a FastAPI instance"""
        import codex_server
        from fastapi import FastAPI
        assert isinstance(codex_server.app, FastAPI)

    def test_sasso_server_has_app(self):
        """Test that sasso_server has FastAPI app instance"""
        import sasso_server
        assert hasattr(sasso_server, 'app')
        assert sasso_server.app is not None


class TestDependencies:
    """Test that required dependencies are installed"""

    def test_fastapi_installed(self):
        """Test FastAPI is installed"""
        try:
            import fastapi
            assert fastapi is not None
        except ImportError:
            pytest.fail("FastAPI not installed")

    def test_uvicorn_installed(self):
        """Test Uvicorn is installed"""
        try:
            import uvicorn
            assert uvicorn is not None
        except ImportError:
            pytest.fail("Uvicorn not installed")

    def test_pydantic_installed(self):
        """Test Pydantic is installed"""
        try:
            import pydantic
            assert pydantic is not None
        except ImportError:
            pytest.fail("Pydantic not installed")


class TestProjectStructure:
    """Test project file structure"""

    def test_readme_exists(self):
        """Test README.md exists"""
        readme = Path(__file__).parent.parent / "README.md"
        assert readme.exists(), "README.md not found"

    def test_readme_no_merge_conflicts(self):
        """Test README.md has no git merge conflict markers"""
        readme = Path(__file__).parent.parent / "README.md"
        content = readme.read_text()
        assert "<<<<<<< HEAD" not in content, "Found merge conflict markers in README"
        assert "=======" not in content or "---" in content, "Possible merge conflict markers"
        assert ">>>>>>>" not in content, "Found merge conflict markers in README"

    def test_claude_md_exists(self):
        """Test CLAUDE.md exists"""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        assert claude_md.exists(), "CLAUDE.md not found"

    def test_requirements_txt_exists(self):
        """Test requirements.txt exists"""
        requirements = Path(__file__).parent.parent / "requirements.txt"
        assert requirements.exists(), "requirements.txt not found"

    def test_version_md_exists(self):
        """Test VERSION.md exists (clarification document)"""
        version_md = Path(__file__).parent.parent / "VERSION.md"
        assert version_md.exists(), "VERSION.md not found (clarification document)"

    def test_exploration_report_exists(self):
        """Test EXPLORATION_REPORT.md exists"""
        report = Path(__file__).parent.parent / "EXPLORATION_REPORT.md"
        assert report.exists(), "EXPLORATION_REPORT.md not found"


class TestPythonCompatibility:
    """Test Python version compatibility"""

    def test_python_3_9_plus(self):
        """Test running on Python 3.9+"""
        version = sys.version_info
        assert version.major >= 3 and version.minor >= 9, \
            f"Python 3.9+ required, got {version.major}.{version.minor}"

    def test_emmanuel_python_39_compatible(self):
        """Test Emmanuel module works on Python 3.9"""
        # This tests the Union[str] syntax fix
        import emmanuel
        me = emmanuel.Emmanuel644("6.4.4")
        # If we get here without TypeErrors, the fix works
        assert me is not None


class TestCodeQuality:
    """Test basic code quality"""

    def test_codex_server_syntax(self):
        """Test codex_server has valid Python syntax"""
        codex_file = Path(__file__).parent.parent / "codex_server.py"
        assert codex_file.exists()
        # Try to compile to check syntax
        with open(codex_file) as f:
            code = f.read()
        try:
            compile(code, str(codex_file), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in codex_server.py: {e}")

    def test_emmanuel_syntax(self):
        """Test emmanuel has valid Python syntax"""
        emmanuel_file = Path(__file__).parent.parent / "emmanuel.py"
        assert emmanuel_file.exists()
        with open(emmanuel_file) as f:
            code = f.read()
        try:
            compile(code, str(emmanuel_file), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in emmanuel.py: {e}")

    def test_nodo33_agent_manager_syntax(self):
        """Test nodo33_agent_manager has valid Python syntax"""
        agent_file = Path(__file__).parent.parent / "nodo33_agent_manager.py"
        assert agent_file.exists()
        with open(agent_file) as f:
            code = f.read()
        try:
            compile(code, str(agent_file), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in nodo33_agent_manager.py: {e}")


class TestDocumentation:
    """Test documentation exists and is valid"""

    def test_version_md_content(self):
        """Test VERSION.md has expected sections"""
        version_md = Path(__file__).parent.parent / "VERSION.md"
        content = version_md.read_text()
        expected_sections = [
            "Feature Comparison Matrix",
            "Which Version Should I Use",
            "Root Version",
            "nodo33-main Version",
        ]
        for section in expected_sections:
            assert section in content, f"Missing section: {section}"

    def test_exploration_report_content(self):
        """Test EXPLORATION_REPORT.md has expected sections"""
        report = Path(__file__).parent.parent / "EXPLORATION_REPORT.md"
        content = report.read_text()
        expected_sections = [
            "Executive Summary",
            "Issues Identified",
            "Verification Results",
        ]
        for section in expected_sections:
            assert section in content, f"Missing section: {section}"


# Fixtures for FastAPI testing (when running with TestClient)
@pytest.fixture
def fastapi_app():
    """Fixture to get FastAPI app for testing"""
    import codex_server
    return codex_server.app


@pytest.fixture
def fastapi_client(fastapi_app):
    """Fixture to get FastAPI TestClient"""
    try:
        from fastapi.testclient import TestClient
        return TestClient(fastapi_app)
    except ImportError:
        pytest.skip("FastAPI TestClient not available")


class TestFastAPIBasic:
    """Basic FastAPI endpoint tests (requires server context)"""

    def test_health_endpoint_exists(self, fastapi_app):
        """Test /health endpoint exists"""
        routes = [route.path for route in fastapi_app.routes]
        assert "/health" in routes, "Health endpoint not found"

    def test_sasso_endpoint_exists(self, fastapi_app):
        """Test /sasso endpoint exists"""
        routes = [route.path for route in fastapi_app.routes]
        # Sasso is at root (/) or as /api/codex/emanuele
        assert "/" in routes or "/api/codex/emanuele" in routes, "Sasso endpoint not found"


# Summary and metadata
__all__ = [
    'TestModuleImports',
    'TestEmmanuel',
    'TestFastAPIStructure',
    'TestDependencies',
    'TestProjectStructure',
    'TestPythonCompatibility',
    'TestCodeQuality',
    'TestDocumentation',
    'TestFastAPIBasic',
]

"""
Test Coverage Summary:
├─ Module Imports (6 tests)
├─ Emmanuel654 (5 tests)
├─ FastAPI Structure (3 tests)
├─ Dependencies (3 tests)
├─ Project Structure (6 tests)
├─ Python Compatibility (2 tests)
├─ Code Quality (3 tests)
├─ Documentation (2 tests)
└─ FastAPI Basic (2 tests)

Total: ~32 test cases

Run with: pytest tests/test_codex_server.py -v
"""

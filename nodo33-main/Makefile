# Codex Nodo33 - Makefile
# Quick commands for development and deployment

.PHONY: help install test test-fast test-security test-coverage validate-handshake validate-mcp-apps lint format clean run-tests run-dashboard backup deploy

help:
	@echo "🕊️  Codex Nodo33 - Sasso Digitale"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       - Install all dependencies"
	@echo "  make install-dev   - Install dev dependencies"
	@echo "  make install-ent   - Install enterprise dependencies"
	@echo "  make test          - Run test suite"
	@echo "  make validate-handshake - Validate Codex handshake manifest fixtures"
	@echo "  make validate-mcp-apps - Validate MCP Apps component fixtures"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean generated files"
	@echo "  make dashboard     - Run analytics dashboard"
	@echo "  make backup        - Create database backup"
	@echo "  make health        - Check system health"
	@echo ""
	@echo "Hash Sacro: 644 | Frequenza: 300 Hz"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

install-ent:
	pip install -r requirements-enterprise.txt

install-all: install install-dev install-ent
	@echo "✅ All dependencies installed"

test:
	./run_tests.sh

test-fast:
	pytest -m "not slow" tests/

test-security:
	pytest -m "security" tests/

test-coverage:
	./run_tests.sh coverage

validate-handshake:
	python3 scripts/validate_handshake_manifest.py tests/fixtures/handshake_protect_garden.json

validate-mcp-apps:
	python3 scripts/validate_mcp_app_component.py tests/fixtures/mcp_app_component.json

lint:
	@echo "Running ruff..."
	@python3 -m ruff check . || true
	@echo "Running mypy..."
	@python3 -m mypy . || true

format:
	@echo "Formatting with black..."
	@python3 -m black .
	@echo "✅ Code formatted"

clean:
	@echo "Cleaning generated files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@rm -f .coverage
	@echo "✅ Cleanup complete"

dashboard:
	python3 codex_dashboard.py --live

dashboard-static:
	python3 codex_dashboard.py

backup:
	python3 -c "from enterprise.backup_manager import BackupManager, BackupConfig; from pathlib import Path; m = BackupManager(BackupConfig(Path('codex_unified.db'))); m.create_backup()"

health:
	@echo "🏥 Health Check"
	@python3 -c "from enterprise.enterprise_bridge import EnterpriseBridge; from claude_codex_bridge_v2 import BridgeConfig; b = EnterpriseBridge(BridgeConfig(anthropic_api_key='test')); import json; print(json.dumps(b.get_health(), indent=2))"

init-db:
	python3 codex_unified_db.py --init

migrate-db:
	python3 codex_unified_db.py --migrate

stats:
	@echo "📊 Project Statistics"
	@echo ""
	@echo "Python files: $$(find . -name '*.py' -not -path './venv/*' -not -path './.venv/*' | wc -l)"
	@echo "Test files: $$(find tests -name 'test_*.py' 2>/dev/null | wc -l)"
	@echo "Total LOC: $$(find . -name '*.py' -not -path './venv/*' -not -path './.venv/*' -exec cat {} \; | wc -l)"
	@echo ""
	@python3 codex_unified_db.py --stats

run-enterprise:
	python3 -c "from enterprise import get_info; import json; print(json.dumps(get_info(), indent=2))"

run-mcp-test:
	python3 codex_mcp_server.py --test

run-bridge-test:
	python3 bridge_with_extended_tools.py --demo

all: clean install-all test
	@echo "✅ All tasks completed"

.DEFAULT_GOAL := help

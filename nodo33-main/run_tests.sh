#!/bin/bash
# Test runner for Nodo33 Codex
# Usage: ./run_tests.sh [options]

set -e

echo "🕊️ Nodo33 Test Runner 🕊️"
echo "========================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}⚠️  pytest not found. Installing dependencies...${NC}"
    pip install -r requirements-dev.txt
fi

# Parse arguments
MODE="${1:-all}"

case $MODE in
    unit)
        echo "Running unit tests only..."
        pytest -m "unit" tests/
        ;;
    integration)
        echo "Running integration tests only..."
        pytest -m "integration" tests/
        ;;
    security)
        echo "Running security tests only..."
        pytest -m "security" tests/
        ;;
    fast)
        echo "Running fast tests (excluding slow)..."
        pytest -m "not slow" tests/
        ;;
    coverage)
        echo "Running tests with coverage report..."
        pytest --cov=. --cov-report=html --cov-report=term tests/
        echo ""
        echo -e "${GREEN}✅ Coverage report generated: htmlcov/index.html${NC}"
        ;;
    all|*)
        echo "Running all tests..."
        pytest tests/ "$@"
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✨ All tests passed! Fiat Lux! ✨${NC}"
else
    echo ""
    echo -e "${RED}❌ Some tests failed. Check output above.${NC}"
fi

exit $EXIT_CODE

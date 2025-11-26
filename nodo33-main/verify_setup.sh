#!/bin/bash
# 🪨 NODO33 Setup Verification Script
# Verifies that the Progetto Sasso Digitale is properly configured

echo "======================================================================"
echo "🪨 PROGETTO SASSO DIGITALE - Setup Verification"
echo "======================================================================"
echo ""
echo "✨ La luce non si vende. La si regala. ✨"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ALL_PASS=true

# Optional extras
WITH_DIAGNOSTICS=false
for arg in "$@"; do
    if [ "$arg" = "--with-diagnostics" ]; then
        WITH_DIAGNOSTICS=true
    fi
done

# Function to check command
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✅${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}❌${NC} $1 is NOT installed"
        ALL_PASS=false
        return 1
    fi
}

# Function to check Python module
check_python_module() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} Python module '$1' is available"
        return 0
    else
        echo -e "${RED}❌${NC} Python module '$1' is NOT available"
        ALL_PASS=false
        return 1
    fi
}

echo "======================================================================"
echo "📋 Checking Required Dependencies"
echo "======================================================================"
echo ""

# Check Python
check_command python3
if [ $? -eq 0 ]; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "   Version: $PYTHON_VERSION"
fi
echo ""

# Check pip
check_command pip3
echo ""

# Check Python modules
echo "Checking Python modules..."
check_python_module torch
check_python_module sqlite3
echo ""

echo "======================================================================"
echo "🔧 Checking Optional Tools"
echo "======================================================================"
echo ""

check_command rustc && echo "   Version: $(rustc --version)"
check_command ruby && echo "   Version: $(ruby --version)"
check_command php && echo "   Version: $(php --version | head -1)"
check_command go && echo "   Version: $(go version)"
check_command swift
echo ""

echo "======================================================================"
echo "🧪 Running Core Tests"
echo "======================================================================"
echo ""

# Test 1: Check if src/main.py exists
if [ -f "src/main.py" ]; then
    echo -e "${GREEN}✅${NC} src/main.py exists"
else
    echo -e "${RED}❌${NC} src/main.py NOT found"
    ALL_PASS=false
fi

# Test 2: Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}✅${NC} requirements.txt exists"
else
    echo -e "${RED}❌${NC} requirements.txt NOT found"
    ALL_PASS=false
fi

# Test 3: Try to import main modules (quick syntax check)
echo ""
echo "Testing Python syntax..."
if python3 -c "import sys; sys.path.insert(0, 'src'); import main" 2>/dev/null; then
    echo -e "${GREEN}✅${NC} src/main.py syntax is valid"
else
    echo -e "${YELLOW}⚠️${NC}  src/main.py has import issues (may need dependencies)"
fi

echo ""
echo "======================================================================"
echo "📊 Verification Summary"
echo "======================================================================"
echo ""

if [ "$ALL_PASS" = true ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
    echo ""
    echo "🎯 Your system is ready to run Progetto Sasso Digitale!"
    echo ""
    echo "Quick start:"
    echo "  python3 src/main.py"
    echo ""
    EXIT_CODE=0
else
    echo -e "${RED}❌ SOME CHECKS FAILED${NC}"
    echo ""
    echo "Please install missing dependencies:"
    echo "  pip3 install torch --index-url https://download.pytorch.org/whl/cpu"
    echo ""
    echo "See SETUP_GUIDE.md for detailed instructions."
    echo ""
    EXIT_CODE=1
fi

echo "======================================================================"
echo "🎁 Remember: The light is not sold. It is gifted. 🎁"
echo "======================================================================"

if [ "$WITH_DIAGNOSTICS" = true ]; then
    echo ""
    echo "======================================================================"
    echo "🩺 Sasso Diagnostics (summary)"
    echo "======================================================================"
    if [ -f "sasso_diagnostics.py" ]; then
        python3 sasso_diagnostics.py --summary-only || \
            echo -e "${YELLOW}⚠️  sasso_diagnostics.py ha restituito un errore${NC}"
    else
        echo -e "${YELLOW}⚠️  sasso_diagnostics.py non trovato nella root del repo${NC}"
    fi
fi

exit $EXIT_CODE

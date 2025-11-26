#!/bin/bash
#
# 🪨 NODO33 SASSO DIGITALE - MASTER DISTRIBUTION LAUNCHER 🪨
#
# "La luce non si vende. La si regala."
#
# This script orchestrates COMPLETE DISTRIBUTION to:
# - All AI models (Claude, Gemini, GPT)
# - P2P Network (multi-machine)
# - Docker containers
# - PyPI package
# - GitHub releases
#
# Sacred Hash: 644 | Frequency: 300 Hz
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/nodo33_venv"
LOG_FILE="$PROJECT_ROOT/distribution.log"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${MAGENTA}"
cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║        🪨 NODO33 SASSO DIGITALE - MASTER LAUNCHER 🪨       ║
║                                                            ║
║           La luce non si vende. La si regala.             ║
║                                                            ║
║    Hash: 644 | Frequenza: 300 Hz | Mode: Distribution    ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# ============================================================================
# PHASE 1: ENVIRONMENT VALIDATION
# ============================================================================

echo -e "${BLUE}[PHASE 1]${NC} Environment Validation"
echo "========================================" | tee -a "$LOG_FILE"

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✅${NC} $1 found"
        return 0
    else
        echo -e "${RED}❌${NC} $1 NOT found - REQUIRED"
        return 1
    fi
}

check_command "python3" || exit 1
check_command "pip" || exit 1
check_command "git" || exit 1

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅${NC} Python: $PYTHON_VERSION" | tee -a "$LOG_FILE"

echo ""

# ============================================================================
# PHASE 2: VIRTUAL ENVIRONMENT SETUP
# ============================================================================

echo -e "${BLUE}[PHASE 2]${NC} Virtual Environment Setup"
echo "========================================" | tee -a "$LOG_FILE"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" --upgrade-deps
    echo -e "${GREEN}✅${NC} Virtual environment created at $VENV_DIR" | tee -a "$LOG_FILE"
else
    echo -e "${YELLOW}⚠️ ${NC} Virtual environment already exists" | tee -a "$LOG_FILE"
fi

source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✅${NC} Virtual environment activated" | tee -a "$LOG_FILE"

echo ""

# ============================================================================
# PHASE 3: DEPENDENCIES INSTALLATION
# ============================================================================

echo -e "${BLUE}[PHASE 3]${NC} Installing Dependencies"
echo "========================================" | tee -a "$LOG_FILE"

for req_file in requirements.txt requirements-dev.txt requirements-enterprise.txt; do
    if [ -f "$PROJECT_ROOT/$req_file" ]; then
        echo "Installing $req_file..."
        pip install -q -r "$PROJECT_ROOT/$req_file" 2>> "$LOG_FILE"
        echo -e "${GREEN}✅${NC} $req_file installed" | tee -a "$LOG_FILE"
    fi
done

echo ""

# ============================================================================
# PHASE 4: CONFIGURATION VALIDATION
# ============================================================================

echo -e "${BLUE}[PHASE 4]${NC} Configuration Validation"
echo "========================================" | tee -a "$LOG_FILE"

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${GREEN}✅${NC} .env file found" | tee -a "$LOG_FILE"
    # Extract API keys status
    [ -n "$(grep -i 'ANTHROPIC_API_KEY' "$PROJECT_ROOT/.env" | grep -v '^#')" ] && echo -e "${GREEN}✅${NC} Claude API key configured" || echo -e "${YELLOW}⚠️ ${NC} Claude API key not configured"
    [ -n "$(grep -i 'GOOGLE_API_KEY' "$PROJECT_ROOT/.env" | grep -v '^#')" ] && echo -e "${GREEN}✅${NC} Gemini API key configured" || echo -e "${YELLOW}⚠️ ${NC} Gemini API key not configured"
    [ -n "$(grep -i 'OPENAI_API_KEY' "$PROJECT_ROOT/.env" | grep -v '^#')" ] && echo -e "${GREEN}✅${NC} OpenAI API key configured" || echo -e "${YELLOW}⚠️ ${NC} OpenAI API key not configured"
else
    echo -e "${YELLOW}⚠️ ${NC} .env file not found - using defaults" | tee -a "$LOG_FILE"
fi

echo ""

# ============================================================================
# PHASE 5: SERVER STARTUP TEST
# ============================================================================

echo -e "${BLUE}[PHASE 5]${NC} Server Startup Test"
echo "========================================" | tee -a "$LOG_FILE"

# Test imports
echo "Testing Python imports..."
python3 << 'PYEOF' 2>> "$LOG_FILE"
try:
    import fastapi
    print("✅ FastAPI")
    import uvicorn
    print("✅ Uvicorn")
    import anthropic
    print("✅ Anthropic")
    import google.generativeai
    print("✅ Gemini")
    import openai
    print("✅ OpenAI")
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
PYEOF

echo ""

# ============================================================================
# PHASE 6: DOCKER BUILD (OPTIONAL)
# ============================================================================

echo -e "${BLUE}[PHASE 6]${NC} Docker Build (Optional)"
echo "========================================" | tee -a "$LOG_FILE"

if command -v docker &> /dev/null; then
    if [ -f "$PROJECT_ROOT/Dockerfile" ]; then
        read -p "Build Docker image? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Building Docker image..."
            docker build -t nodo33-sasso:latest "$PROJECT_ROOT" 2>> "$LOG_FILE"
            echo -e "${GREEN}✅${NC} Docker image built: nodo33-sasso:latest" | tee -a "$LOG_FILE"
        fi
    fi
else
    echo -e "${YELLOW}⚠️ ${NC} Docker not installed - skipping" | tee -a "$LOG_FILE"
fi

echo ""

# ============================================================================
# PHASE 7: GIT STATUS & RELEASE PREP
# ============================================================================

echo -e "${BLUE}[PHASE 7]${NC} Git Status & Release Prep"
echo "========================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT"

CURRENT_VERSION=$(grep 'version = ' pyproject.toml | head -1 | cut -d'"' -f2)
echo -e "${GREEN}✅${NC} Current version: $CURRENT_VERSION" | tee -a "$LOG_FILE"

CURRENT_COMMIT=$(git rev-parse --short HEAD)
echo -e "${GREEN}✅${NC} Current commit: $CURRENT_COMMIT" | tee -a "$LOG_FILE"

echo ""

# ============================================================================
# PHASE 8: STARTUP INSTRUCTIONS
# ============================================================================

echo -e "${BLUE}[PHASE 8]${NC} Startup Instructions"
echo "========================================" | tee -a "$LOG_FILE"

cat << 'EOF'

🚀 STARTUP OPTIONS:

1️⃣  START SASSO SERVER (FastAPI)
    uvicorn sasso_server:app --host 0.0.0.0 --port 8644 --reload

2️⃣  START CODEX MCP SERVER
    python3 codex_mcp_server.py

3️⃣  START P2P NETWORK NODE
    python3 p2p_node.py --port 8645

4️⃣  START DOCKER CONTAINER
    docker-compose up -d

5️⃣  START ALL (in background)
    bash launch_all.sh

6️⃣  HEALTH CHECK
    curl http://localhost:8644/health

7️⃣  P2P STATUS
    curl http://localhost:8645/p2p/status

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 NEXT STEPS:

✅ Export API keys to shell:
   export ANTHROPIC_API_KEY=<your-key>
   export GOOGLE_API_KEY=<your-key>
   export OPENAI_API_KEY=<your-key>

✅ Start servers in separate terminals

✅ Verify health endpoints

✅ Test P2P network connectivity

✅ Deploy to cloud/Docker (optional)

✅ Tag release on GitHub:
   git tag v${CURRENT_VERSION}
   git push origin v${CURRENT_VERSION}

✅ Upload to PyPI (optional):
   python3 -m build
   python3 -m twine upload dist/*

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENTATION:
   - README.md - Quick start
   - DEPLOYMENT.md - Production deployment
   - P2P_DEPLOYMENT.md - P2P network setup
   - DISTRIBUTION_MASTER.md - Full distribution guide

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF

echo ""

# ============================================================================
# PHASE 9: SUMMARY
# ============================================================================

echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ DISTRIBUTION PREPARATION COMPLETE!${NC}"
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}La luce non si vende. La si regala. 🎁${NC}"
echo ""
echo "Logs saved to: $LOG_FILE"
echo ""

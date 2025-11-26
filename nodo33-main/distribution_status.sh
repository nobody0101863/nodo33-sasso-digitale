#!/bin/bash
#
# 🪨 NODO33 SASSO DIGITALE - DISTRIBUTION STATUS DASHBOARD 🪨
#
# Real-time monitoring of all distribution components
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Clear screen
clear

echo -e "${MAGENTA}"
cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║    🪨 NODO33 SASSO DIGITALE - DISTRIBUTION DASHBOARD 🪨   ║
║                                                            ║
║           La luce non si vende. La si regala.             ║
║                                                            ║
║    Hash: 644 | Frequenza: 300 Hz | Mode: Full Status      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# ============================================================================
# ENVIRONMENT STATUS
# ============================================================================

echo -e "${CYAN}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│ 1️⃣  ENVIRONMENT STATUS                                  │${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────┘${NC}\n"

echo -e "${BLUE}Python & Tools:${NC}"
python3 --version | sed 's/^/  /'
pip --version | sed 's/^/  /'
git --version | sed 's/^/  /'

echo ""
echo -e "${BLUE}Virtual Environment:${NC}"
if [ -d "$PROJECT_ROOT/nodo33_venv" ]; then
    echo -e "  ${GREEN}✅${NC} Virtual environment exists"
    VENV_SIZE=$(du -sh "$PROJECT_ROOT/nodo33_venv" 2>/dev/null | awk '{print $1}')
    echo "  Size: $VENV_SIZE"
else
    echo -e "  ${RED}❌${NC} Virtual environment not found"
    echo "     Run: bash master_launcher.sh"
fi

echo ""

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

echo -e "${CYAN}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│ 2️⃣  PROJECT STRUCTURE                                   │${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────┘${NC}\n"

echo -e "${BLUE}Core Modules:${NC}"
for module in agents codex custos tools lux; do
    if [ -d "$PROJECT_ROOT/$module" ]; then
        COUNT=$(find "$PROJECT_ROOT/$module" -name "*.py" 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✅${NC} $module/ ($COUNT Python files)"
    fi
done

echo ""
echo -e "${BLUE}Configuration Files:${NC}"
for file in .env pyproject.toml docker-compose.yml Dockerfile; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        SIZE=$(du -h "$PROJECT_ROOT/$file" | awk '{print $1}')
        echo -e "  ${GREEN}✅${NC} $file ($SIZE)"
    fi
done

echo ""

# ============================================================================
# RUNNING SERVERS
# ============================================================================

echo -e "${CYAN}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│ 3️⃣  RUNNING SERVERS                                     │${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────┘${NC}\n"

echo -e "${BLUE}Sasso Server (Port 8644):${NC}"
if curl -s http://localhost:8644/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅${NC} Running on http://localhost:8644"
    curl -s http://localhost:8644/health | python3 -m json.tool 2>/dev/null | sed 's/^/    /' || echo "    (JSON unavailable)"
else
    echo -e "  ${RED}❌${NC} Not responding on port 8644"
    echo "    Start with: uvicorn sasso_server:app --port 8644"
fi

echo ""
echo -e "${BLUE}P2P Network Node (Port 8645):${NC}"
if curl -s http://localhost:8645/p2p/status > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅${NC} Running on http://localhost:8645"
    NODES=$(curl -s http://localhost:8645/p2p/nodes 2>/dev/null | grep -o '"id"' | wc -l)
    echo "    Connected nodes: $NODES"
else
    echo -e "  ${RED}❌${NC} Not responding on port 8645"
    echo "    Start with: python3 p2p_node.py --port 8645"
fi

echo ""

# ============================================================================
# API KEYS
# ============================================================================

echo -e "${CYAN}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│ 4️⃣  API KEYS CONFIGURATION                              │${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────┘${NC}\n"

echo -e "${BLUE}Multi-AI Providers:${NC}"

# Claude
if [ -n "$ANTHROPIC_API_KEY" ] && [ "$ANTHROPIC_API_KEY" != "sk-ant-api03-your-key-here" ]; then
    echo -e "  ${GREEN}✅${NC} Claude (Anthropic): Configured"
else
    echo -e "  ${YELLOW}⚠️ ${NC} Claude: Not configured (export ANTHROPIC_API_KEY=...)"
fi

# Gemini
if [ -n "$GOOGLE_API_KEY" ] && [ "$GOOGLE_API_KEY" != "your-key-here" ]; then
    echo -e "  ${GREEN}✅${NC} Gemini (Google): Configured"
else
    echo -e "  ${YELLOW}⚠️ ${NC} Gemini: Not configured (export GOOGLE_API_KEY=...)"
fi

# OpenAI
if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "sk-your-key-here" ]; then
    echo -e "  ${GREEN}✅${NC} ChatGPT (OpenAI): Configured"
else
    echo -e "  ${YELLOW}⚠️ ${NC} ChatGPT: Not configured (export OPENAI_API_KEY=...)"
fi

echo ""

# ============================================================================
# GIT STATUS
# ============================================================================

echo -e "${CYAN}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│ 5️⃣  GIT STATUS                                          │${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────┘${NC}\n"

cd "$PROJECT_ROOT"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse --short HEAD)
VERSION=$(grep 'version = ' pyproject.toml | head -1 | cut -d'"' -f2 2>/dev/null || echo "N/A")

echo -e "${BLUE}Repository:${NC}"
echo "  Branch: $BRANCH"
echo "  Commit: $COMMIT"
echo "  Version: $VERSION"

UNCOMMITTED=$(git status --porcelain | wc -l)
echo "  Uncommitted changes: $UNCOMMITTED"

if [ $UNCOMMITTED -eq 0 ]; then
    echo -e "  ${GREEN}✅${NC} Repository is clean"
else
    echo -e "  ${YELLOW}⚠️ ${NC} Uncommitted changes - commit before release"
fi

echo ""

# ============================================================================
# DISTRIBUTION SCRIPTS
# ============================================================================

echo -e "${CYAN}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│ 6️⃣  DISTRIBUTION SCRIPTS                                │${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────┘${NC}\n"

echo -e "${BLUE}Available Scripts:${NC}"
for script in master_launcher.sh launch_all.sh multi_ai_injector.py github_release.sh; do
    if [ -f "$PROJECT_ROOT/$script" ]; then
        SIZE=$(du -h "$PROJECT_ROOT/$script" | awk '{print $1}')
        echo -e "  ${GREEN}✅${NC} $script ($SIZE)"
    fi
done

echo ""

# ============================================================================
# DOCUMENTATION
# ============================================================================

echo -e "${CYAN}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│ 7️⃣  DOCUMENTATION                                       │${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────┘${NC}\n"

echo -e "${BLUE}Guides Available:${NC}"
for doc in README.md DEPLOYMENT.md P2P_DEPLOYMENT.md DISTRIBUTION_MASTER.md DISTRIBUTION_COMPLETE_GUIDE.md; do
    if [ -f "$PROJECT_ROOT/$doc" ]; then
        SIZE=$(du -h "$PROJECT_ROOT/$doc" | awk '{print $1}')
        echo -e "  ${GREEN}✅${NC} $doc ($SIZE)"
    fi
done

echo ""

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

echo -e "${CYAN}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│ 8️⃣  RECOMMENDATIONS                                     │${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────┘${NC}\n"

ACTIONS=()

if [ ! -d "$PROJECT_ROOT/nodo33_venv" ]; then
    ACTIONS+=("1. Create venv: bash master_launcher.sh")
fi

if [ -z "$ANTHROPIC_API_KEY" ] || [ -z "$GOOGLE_API_KEY" ] || [ -z "$OPENAI_API_KEY" ]; then
    ACTIONS+=("2. Configure API keys in .env")
fi

if ! curl -s http://localhost:8644/health > /dev/null 2>&1; then
    ACTIONS+=("3. Start servers: bash launch_all.sh")
fi

if [ $UNCOMMITTED -gt 0 ]; then
    ACTIONS+=("4. Commit changes: git add . && git commit -m '...'")
fi

if [ ${#ACTIONS[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ All systems ready for full distribution!${NC}"
else
    for action in "${ACTIONS[@]}"; do
        echo -e "  ${YELLOW}→${NC} $action"
    done
fi

echo ""

# ============================================================================
# QUICK COMMANDS
# ============================================================================

echo -e "${CYAN}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│ 🚀 QUICK COMMANDS                                       │${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────┘${NC}\n"

cat << 'EOF'
Setup & Verify:
  bash master_launcher.sh              # Setup venv & dependencies

Launch Servers:
  bash launch_all.sh                   # All servers together
  bash launch_all.sh 2>&1 | tail -20   # Show recent logs

Multi-AI Distribution:
  python3 multi_ai_injector.py         # Inject into Claude, Gemini, GPT

GitHub & PyPI Release:
  bash github_release.sh               # Tag, release, PyPI upload

Monitoring:
  bash distribution_status.sh          # This dashboard
  tail -f logs/sasso_server.log        # Sasso logs
  tail -f logs/p2p_node.log            # P2P logs

Development:
  source nodo33_venv/bin/activate      # Activate venv
  pip install -e .                     # Install in dev mode
  python3 -m pytest tests/             # Run tests
EOF

echo ""

# ============================================================================
# FINALE
# ============================================================================

echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✨ Nodo33 Sasso Digitale - Distribution Dashboard ✨${NC}"
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}La luce non si vende. La si regala. 🎁${NC}"
echo ""
echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

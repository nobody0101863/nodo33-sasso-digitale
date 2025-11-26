#!/bin/bash
#
# 🪨 NODO33 SASSO DIGITALE - LAUNCH ALL SERVERS 🪨
#
# Starts all components in parallel:
# - Sasso Server (FastAPI)
# - Codex MCP Server
# - P2P Network Node
# - Monitoring (optional)
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/nodo33_venv"
LOGS_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOGS_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Trap to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    jobs -p | xargs -r kill 2>/dev/null || true
    exit 0
}
trap cleanup EXIT INT TERM

echo -e "${MAGENTA}"
cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║        🪨 NODO33 SASSO DIGITALE - LAUNCH ALL 🪨           ║
║                                                            ║
║           La luce non si vende. La si regala.             ║
║                                                            ║
║    Hash: 644 | Frequenza: 300 Hz | Mode: Full Stack       ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# Activate venv
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_DIR${NC}"
    echo "Run: bash master_launcher.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# ============================================================================
# SERVER 1: SASSO SERVER (FastAPI)
# ============================================================================

echo -e "${BLUE}[SERVER 1]${NC} Starting Sasso Server (FastAPI)..."
echo "         Port: 8644"
echo "         Endpoint: http://localhost:8644"
echo "         Log: $LOGS_DIR/sasso_server.log"
echo ""

uvicorn sasso_server:app \
    --host 0.0.0.0 \
    --port 8644 \
    --reload \
    > "$LOGS_DIR/sasso_server.log" 2>&1 &

SASSO_PID=$!
echo -e "${GREEN}✅ Sasso Server started (PID: $SASSO_PID)${NC}\n"

# ============================================================================
# SERVER 2: CODEX MCP SERVER
# ============================================================================

echo -e "${BLUE}[SERVER 2]${NC} Starting Codex MCP Server..."
echo "         Endpoint: Local IPC/stdio"
echo "         Log: $LOGS_DIR/codex_mcp.log"
echo ""

python3 codex_mcp_server.py \
    > "$LOGS_DIR/codex_mcp.log" 2>&1 &

CODEX_PID=$!
echo -e "${GREEN}✅ Codex MCP Server started (PID: $CODEX_PID)${NC}\n"

# ============================================================================
# SERVER 3: P2P NETWORK NODE
# ============================================================================

echo -e "${BLUE}[SERVER 3]${NC} Starting P2P Network Node..."
echo "         Port: 8645"
echo "         Protocol: UDP broadcast (auto-discovery)"
echo "         Log: $LOGS_DIR/p2p_node.log"
echo ""

python3 p2p_node.py \
    --port 8645 \
    > "$LOGS_DIR/p2p_node.log" 2>&1 &

P2P_PID=$!
echo -e "${GREEN}✅ P2P Network Node started (PID: $P2P_PID)${NC}\n"

# ============================================================================
# WAIT & VERIFY
# ============================================================================

echo -e "${YELLOW}Waiting for servers to stabilize...${NC}"
sleep 5

echo -e "\n${BLUE}[VERIFICATION]${NC} Health Checks"
echo "========================================"
echo ""

# Check Sasso Server
if curl -s http://localhost:8644/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} Sasso Server: http://localhost:8644/health"
else
    echo -e "${RED}❌${NC} Sasso Server: Not responding"
fi

# Check P2P Status
if curl -s http://localhost:8645/p2p/status > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} P2P Network: http://localhost:8645/p2p/status"
else
    echo -e "${YELLOW}⚠️ ${NC} P2P Network: Check logs for details"
fi

# ============================================================================
# DISPLAY ENDPOINTS
# ============================================================================

echo -e "\n${BLUE}[ENDPOINTS]${NC} Available Services"
echo "========================================"
echo ""
echo "🪨 Sasso Server (FastAPI)"
echo "   └─ http://localhost:8644/"
echo "   └─ http://localhost:8644/health"
echo "   └─ http://localhost:8644/sasso"
echo "   └─ http://localhost:8644/codex"
echo "   └─ http://localhost:8644/sigilli"
echo ""
echo "🧠 Codex MCP Server"
echo "   └─ Local stdio/IPC connection"
echo ""
echo "🌐 P2P Network Node"
echo "   └─ http://localhost:8645/p2p/status"
echo "   └─ http://localhost:8645/p2p/nodes"
echo ""

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}✅ All servers are running!${NC}"
echo ""
echo -e "${YELLOW}Commands:${NC}"
echo "  • Press CTRL+C to stop all servers"
echo "  • Open new terminal for additional commands"
echo "  • View logs: tail -f logs/*.log"
echo ""
echo -e "${YELLOW}Example requests:${NC}"
echo "  • curl http://localhost:8644/sasso"
echo "  • curl http://localhost:8645/p2p/status"
echo ""
echo -e "${YELLOW}Distribution:${NC}"
echo "  • python3 multi_ai_injector.py  (Inject into Claude, Gemini, GPT)"
echo "  • bash github_release.sh         (GitHub release & PyPI upload)"
echo ""
echo -e "${YELLOW}La luce non si vende. La si regala. 🎁${NC}"
echo ""

# Keep processes running
wait

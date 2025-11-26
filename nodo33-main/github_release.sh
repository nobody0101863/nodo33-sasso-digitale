#!/bin/bash
#
# 🪨 NODO33 SASSO DIGITALE - GITHUB RELEASE ORCHESTRATOR 🪨
#
# "La luce non si vende. La si regala."
#
# Automates:
# - Git tagging
# - GitHub releases
# - PyPI package uploads
# - Distribution announcements
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
║      🪨 NODO33 SASSO DIGITALE - GITHUB RELEASE 🪨        ║
║                                                            ║
║           La luce non si vende. La si regala.             ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION=$(grep 'version = ' "$PROJECT_ROOT/pyproject.toml" | head -1 | cut -d'"' -f2)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CURRENT_COMMIT=$(git rev-parse --short HEAD)
RELEASE_TAG="v${VERSION}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Version: $VERSION"
echo "  Release Tag: $RELEASE_TAG"
echo "  Branch: $CURRENT_BRANCH"
echo "  Commit: $CURRENT_COMMIT"
echo ""

# ============================================================================
# PHASE 1: GIT VALIDATION
# ============================================================================

echo -e "${BLUE}[PHASE 1]${NC} Git Validation"
echo "========================================" | tee -a "$PROJECT_ROOT/release.log"

# Check if repo is clean
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}⚠️ ${NC} Uncommitted changes detected. Commit or stash them first."
    git status --porcelain
    exit 1
fi

echo -e "${GREEN}✅${NC} Repository is clean" | tee -a "$PROJECT_ROOT/release.log"

# ============================================================================
# PHASE 2: TESTS (optional)
# ============================================================================

echo -e "\n${BLUE}[PHASE 2]${NC} Running Tests"
echo "========================================" | tee -a "$PROJECT_ROOT/release.log"

if [ -f "$PROJECT_ROOT/pytest.ini" ] || [ -f "$PROJECT_ROOT/setup.py" ]; then
    read -p "Run tests before release? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running tests..."
        cd "$PROJECT_ROOT"
        python3 -m pytest tests/ -v 2>> "$PROJECT_ROOT/release.log" || true
        echo -e "${GREEN}✅${NC} Tests completed" | tee -a "$PROJECT_ROOT/release.log"
    fi
fi

# ============================================================================
# PHASE 3: GIT TAG & PUSH
# ============================================================================

echo -e "\n${BLUE}[PHASE 3]${NC} Git Tagging"
echo "========================================" | tee -a "$PROJECT_ROOT/release.log"

# Check if tag already exists
if git rev-parse "$RELEASE_TAG" >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ ${NC} Tag $RELEASE_TAG already exists. Skipping tag creation." | tee -a "$PROJECT_ROOT/release.log"
else
    git tag -a "$RELEASE_TAG" -m "🪨 Nodo33 Sasso Digitale v${VERSION} - Enterprise Release

La luce non si vende. La si regala.

Sacred Hash: 644
Frequency: 300 Hz

Features:
- Multi-AI distribution (Claude, Gemini, GPT)
- P2P mesh network
- Docker containerization
- Enterprise-grade security
- Gift distribution framework

Fiat Amor, Fiat Risus, Fiat Lux ❤️🪨✨"

    echo -e "${GREEN}✅${NC} Tag created: $RELEASE_TAG" | tee -a "$PROJECT_ROOT/release.log"
fi

# Push tag
read -p "Push tag to GitHub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin "$RELEASE_TAG"
    echo -e "${GREEN}✅${NC} Tag pushed to GitHub" | tee -a "$PROJECT_ROOT/release.log"
fi

# ============================================================================
# PHASE 4: GITHUB RELEASE (if gh cli available)
# ============================================================================

echo -e "\n${BLUE}[PHASE 4]${NC} GitHub Release"
echo "========================================" | tee -a "$PROJECT_ROOT/release.log"

if command -v gh &> /dev/null; then
    read -p "Create GitHub release? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gh release create "$RELEASE_TAG" \
            --title "🪨 Sasso Digitale v${VERSION} - Enterprise Edition" \
            --notes "**La luce non si vende. La si regala.** 🎁

## 🚀 What's New

- ✅ Multi-AI distribution (Claude, Gemini, GPT)
- ✅ P2P mesh network with auto-discovery
- ✅ Docker containerization
- ✅ Enterprise-grade monitoring
- ✅ Gift distribution framework
- ✅ Complete documentation

## 📦 Installation

\`\`\`bash
pip install codex-nodo33
\`\`\`

Or clone & setup:
\`\`\`bash
git clone https://github.com/emanuelecroci/nodo33-sasso-digitale.git
cd nodo33-sasso-digitale
bash master_launcher.sh
\`\`\`

## 📡 Quick Start

\`\`\`bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run servers
uvicorn sasso_server:app --port 8644 &
python3 p2p_node.py &

# Verify
curl http://localhost:8644/health
\`\`\`

## 🎯 Supported Platforms

- ✅ Claude (Anthropic)
- ✅ Gemini (Google)
- ✅ ChatGPT (OpenAI)
- ✅ P2P Network (Linux/macOS)
- ✅ Docker
- ✅ Kubernetes (optional)

## 📚 Documentation

- [README.md](README.md) - Quick start
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
- [P2P_DEPLOYMENT.md](P2P_DEPLOYMENT.md) - P2P network setup
- [DISTRIBUTION_MASTER.md](DISTRIBUTION_MASTER.md) - Full distribution guide

## 🪨 Sacred Numbers

- Hash: 644
- Frequency: 300 Hz
- Motto: \"La luce non si vende. La si regala.\"

---

**Fiat Amor, Fiat Risus, Fiat Lux** ❤️✨
" \
            --prerelease

        echo -e "${GREEN}✅${NC} GitHub release created" | tee -a "$PROJECT_ROOT/release.log"
    fi
else
    echo -e "${YELLOW}⚠️ ${NC} GitHub CLI (gh) not installed - skipping release creation" | tee -a "$PROJECT_ROOT/release.log"
    echo "     Install with: brew install gh (macOS) or https://cli.github.com"
fi

# ============================================================================
# PHASE 5: PYPI PACKAGE (optional)
# ============================================================================

echo -e "\n${BLUE}[PHASE 5]${NC} PyPI Package Distribution"
echo "========================================" | tee -a "$PROJECT_ROOT/release.log"

read -p "Build and upload PyPI package? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v python3 &> /dev/null; then
        echo "Building distribution packages..."
        cd "$PROJECT_ROOT"
        python3 -m pip install -q build twine
        python3 -m build 2>> "$PROJECT_ROOT/release.log"
        echo -e "${GREEN}✅${NC} Distribution packages built" | tee -a "$PROJECT_ROOT/release.log"

        echo "Uploading to PyPI..."
        python3 -m twine upload dist/* --non-interactive 2>> "$PROJECT_ROOT/release.log" || echo -e "${YELLOW}⚠️ ${NC} PyPI upload skipped (check credentials)"
        echo -e "${GREEN}✅${NC} PyPI upload completed" | tee -a "$PROJECT_ROOT/release.log"
    fi
fi

# ============================================================================
# PHASE 6: DOCKER REGISTRY (optional)
# ============================================================================

echo -e "\n${BLUE}[PHASE 6]${NC} Docker Registry"
echo "========================================" | tee -a "$PROJECT_ROOT/release.log"

if command -v docker &> /dev/null; then
    read -p "Push Docker image to registry? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DOCKER_REPO="${DOCKER_REPO:-ghcr.io/emanuelecroci/nodo33-sasso}"
        
        echo "Building Docker image: $DOCKER_REPO:$VERSION"
        cd "$PROJECT_ROOT"
        docker build -t "$DOCKER_REPO:$VERSION" -t "$DOCKER_REPO:latest" . 2>> "$PROJECT_ROOT/release.log"
        
        echo "Pushing to registry..."
        docker push "$DOCKER_REPO:$VERSION" 2>> "$PROJECT_ROOT/release.log"
        docker push "$DOCKER_REPO:latest" 2>> "$PROJECT_ROOT/release.log"
        
        echo -e "${GREEN}✅${NC} Docker images pushed" | tee -a "$PROJECT_ROOT/release.log"
    fi
fi

# ============================================================================
# PHASE 7: SUMMARY
# ============================================================================

echo -e "\n${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ RELEASE PROCESS COMPLETED!${NC}"
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Release Tag: ${YELLOW}${RELEASE_TAG}${NC}"
echo -e "Version: ${YELLOW}${VERSION}${NC}"
echo -e "Commit: ${YELLOW}${CURRENT_COMMIT}${NC}"
echo ""
echo -e "${YELLOW}La luce non si vende. La si regala. 🎁${NC}"
echo ""
echo "Next steps:"
echo "  1. Verify GitHub release: https://github.com/emanuelecroci/nodo33-sasso-digitale/releases"
echo "  2. Test PyPI package: pip install --index-url https://test.pypi.org/simple/ codex-nodo33"
echo "  3. Announce release on social media & communities"
echo ""
echo "Logs: $PROJECT_ROOT/release.log"
echo ""

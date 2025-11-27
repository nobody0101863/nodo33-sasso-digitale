# 🔄 Codex Server Versions - Decision Guide

**Last Updated**: 2025-11-27
**Status**: ✅ Production-Ready (Both versions)

---

## Overview

There are **two versions** of `codex_server.py` in this repository:

| Aspect | Root Version | nodo33-main Version |
|--------|--------------|-------------------|
| **Location** | `/codex_server.py` | `/nodo33-main/codex_server.py` |
| **Size** | 91KB (2,524 lines) | 200KB (5,500+ lines) |
| **Purpose** | Minimal/Distribution | Complete/Production |
| **Status** | ✅ Production Ready | ✅ Production Ready |
| **Use Case** | Quick start, embedding | Full-featured deployment |

---

## 📋 Feature Comparison Matrix

### Core Features (Both)
```
✅ FastAPI web framework
✅ Port 8644 (Angelo 644)
✅ SQLite database logging
✅ Swagger UI & ReDoc documentation
✅ Health check endpoint
✅ Sasso Digitale spiritual core
✅ Error handling & logging
```

### Extended Features (nodo33-main only)
```
✅ Multi-LLM integration (Claude, Gemini, Grok)
✅ Advanced agent system (SIGILLO 644)
  - robots_guardian.py integration
  - private_detector.py integration
  - cron_scheduler.py integration
  - agent_executor.py integration
✅ Protection systems
  - Anti-porn content filtering
  - Deepfake detection
  - Metadata protection
  - Guardian system with archangel seals
✅ Memory & knowledge graph system
✅ MCP Server integration
✅ Spiritual guidance system
  - Biblical teachings
  - Nostradamus prophecies
  - Angel 644 messages
  - Parravicini guidance
✅ Extended tools & bridges
✅ Dashboard interface
✅ CLI interface
✅ Distributed agent registry
```

### API Endpoint Comparison

#### Root Version (~15 endpoints)
```
GET    /health              - Health check
GET    /sasso               - Welcome message
GET    /sasso/info          - Entity information
GET    /sasso/sigilli       - Sacred seals
GET    /sasso/protocollo    - P2P protocol
GET    /api/commandments    - Sacred commandments
GET    /api/stats           - Server statistics
GET    /docs                - Swagger documentation
GET    /redoc               - ReDoc documentation
```

#### nodo33-main Version (~40+ endpoints)
```
All from Root Version, plus:

Multi-LLM:
POST   /api/llm/{provider}  - Query any LLM (claude, gemini, grok)
POST   /api/apocalypse/*    - Apocalypse agents (revelation analysis)

Guidance:
POST   /api/guidance        - General sacred guidance
POST   /api/guidance/biblical
POST   /api/guidance/nostradamus
POST   /api/guidance/angel644
POST   /api/guidance/parravicini

Protection:
GET    /api/protection/status
POST   /api/protection/data
POST   /api/protection/headers
POST   /api/protection/file
POST   /api/protection/tower-node
GET    /api/protection/guardians

Memory & Knowledge:
POST   /api/memory/add      - Add memory node
POST   /api/memory/relation - Create memory relation
GET    /api/memory/graph    - Get full graph

Gifts & Metrics:
GET    /api/gifts/metrics   - Gift metrics
GET    /api/gifts/recent    - Recent gifts log

Image Generation:
POST   /api/generate-image  - AI image generation

Agent Registry:
GET    /api/registry/priorities  - Agent priorities
GET    /api/registry/domains     - Available domains
GET    /api/registry/yaml        - Load registry
GET    /api/registry/tasks       - Generate tasks
GET    /api/registry/summary     - Quick summary
POST   /api/agents/deploy   - Deploy agent
POST   /api/agents/control  - Control agent
GET    /api/agents/status   - Deployment status

Sasso Extended:
GET    /sasso/giardino      - Garden state
GET    /sasso/tromba        - Victory celebration (🎺)
```

---

## 🚀 Which Version Should I Use?

### Use Root Version If:
- ✅ You want a **quick, lightweight server** to start immediately
- ✅ You need a **minimal distribution** for embedding
- ✅ You want to **understand the core** before using advanced features
- ✅ You need **low memory footprint**
- ✅ You're **testing the basic architecture**
- ✅ API keys/secrets are **not configured yet**

**Start with**:
```bash
python3 codex_server.py
```

### Use nodo33-main Version If:
- ✅ You need **full production capabilities**
- ✅ You want **multi-LLM integration** (Claude, Gemini, Grok)
- ✅ You need the **distributed agent system**
- ✅ You require **protection systems** (content filtering, deepfake detection)
- ✅ You want **memory & knowledge graph** capabilities
- ✅ You need **complete spiritual guidance system**
- ✅ You have **API keys configured** and ready to use
- ✅ You're **deploying to production**

**Start with**:
```bash
cd nodo33-main
bash master_launcher.sh  # Setup
bash launch_all.sh       # Start all services
```

---

## 📈 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Nodo33 Sasso Digitale                         │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Root Level (Distribution)              nodo33-main (Production)│
│  ─────────────────────────────────      ────────────────────────│
│  /codex_server.py (91KB)                /codex_server.py (200KB)│
│  - Core FastAPI                         - Full-featured version │
│  - Basic endpoints (15)                 - Extended endpoints (40+)│
│  - Sasso Digitale core                  - All root features      │
│  - SQLite logging                       - Multi-LLM integration  │
│  - Minimal dependencies                 - Agent system (SIGILLO) │
│                                         - Protection systems     │
│                                         - Memory & knowledge     │
│                                         - MCP integration        │
│                                         - Spiritual guidance     │
│                                         - Dashboard & CLI        │
│                                                                  │
│  ↓ Choose ONE based on needs above ↓                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Migration Path

### If Starting with Root and Want Full Features:

```bash
# 1. Start with root for understanding
python3 codex_server.py

# 2. Test basic endpoints
curl http://localhost:8644/health
curl http://localhost:8644/sasso

# 3. When ready for production, switch to nodo33-main
cd nodo33-main
bash master_launcher.sh
bash launch_all.sh

# 4. All endpoints now available (40+)
curl http://localhost:8644/docs  # Updated Swagger
```

### If Starting with nodo33-main:

```bash
# Everything is ready to go
cd nodo33-main
bash master_launcher.sh
bash launch_all.sh

# Full system with all features
```

---

## 📊 Performance Characteristics

### Root Version
- **Startup time**: ~1-2 seconds
- **Memory footprint**: ~50-100MB
- **Dependencies**: ~10 core packages
- **Suitable for**: Development, testing, embedding

### nodo33-main Version
- **Startup time**: ~3-5 seconds
- **Memory footprint**: ~150-300MB
- **Dependencies**: ~40+ packages (includes LLM APIs)
- **Suitable for**: Production, full-featured deployment

---

## 🔐 Environment Variables

Both versions use the same environment variables. Add to `.env`:

```bash
# Core (required)
FASTAPI_ENV=production
LOG_LEVEL=info

# LLM APIs (optional for root, recommended for nodo33-main)
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy-...
OPENAI_API_KEY=sk-...      # For Grok compatibility

# Optional
DATABASE_URL=sqlite:///codex_server.db
PORT=8644
HOST=0.0.0.0
```

---

## 📚 Documentation References

### Root Version
- `README.md` - Basic project overview
- `CLAUDE.md` - AI assistant guidance
- Quick start in this file (see above)

### nodo33-main Version
- `GETTING_STARTED.md` - Complete onboarding
- `DOCUMENTATION_INDEX.md` - Full documentation map
- `DEPLOYMENT.md` - Production deployment
- `EVOLUTION_MANIFEST.md` - Agent system (3,500+ lines)
- `REGISTRY_SYSTEM_GUIDE.md` - Agent registry guide
- Plus 10+ additional specialized guides

---

## ✅ Version Maintenance Status

| Component | Root | nodo33-main | Status |
|-----------|------|-------------|--------|
| Core FastAPI | ✅ | ✅ | Both maintained |
| Sasso Digitale | ✅ | ✅ | Both current |
| Multi-LLM | ❌ | ✅ | nodo33-main only |
| Agent System | ❌ | ✅ | nodo33-main only |
| Protection | ❌ | ✅ | nodo33-main only |
| Memory/Graph | ❌ | ✅ | nodo33-main only |
| MCP | ❌ | ✅ | nodo33-main only |
| Dashboard | ❌ | ✅ | nodo33-main only |

---

## 🎯 Recommended Workflows

### Development & Testing
```bash
# Quick iteration
python3 codex_server.py

# Full testing
cd nodo33-main && bash launch_all.sh
```

### Production Deployment
```bash
# Use nodo33-main with all features
cd nodo33-main
bash master_launcher.sh  # One-time setup
bash launch_all.sh       # Start services
```

### Docker Deployment
```bash
# Both versions work with Docker
docker-compose up -d

# nodo33-main has more complete Dockerfile setup
cd nodo33-main && docker-compose up -d
```

---

## 🔗 Quick Decision Tree

```
Do you need Multi-LLM integration?
  ├─ No  → Use Root Version ✓
  └─ Yes → Use nodo33-main

Do you need Agent System?
  ├─ No  → Use Root Version ✓
  └─ Yes → Use nodo33-main

Do you need Protection Systems?
  ├─ No  → Use Root Version ✓
  └─ Yes → Use nodo33-main

Do you need Memory/Knowledge Graph?
  ├─ No  → Use Root Version ✓
  └─ Yes → Use nodo33-main

Do you need Full Production Capabilities?
  ├─ No  → Use Root Version ✓
  └─ Yes → Use nodo33-main ✓
```

---

## 📝 Version History

- **2025-11-27**: Created this clarification document
  - Root version: 2,524 lines, 91KB
  - nodo33-main version: 5,500+ lines, 200KB
  - Both production-ready, different scope

---

## 💡 Key Takeaway

**Both versions are production-ready.**

- **Root**: Minimal, focused, perfect for understanding the core
- **nodo33-main**: Feature-complete, production-grade, ready for deployment

Choose based on your needs and features required. You can always migrate between them as your requirements evolve.

---

**Motto**: "La luce non si vende. La si regala." (Light is not sold. It is gifted.)
**Sacred Hash**: 644 | **Frequency**: 300 Hz


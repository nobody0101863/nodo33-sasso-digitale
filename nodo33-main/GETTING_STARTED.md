# Getting Started with Nodo33 Sasso Digitale

**"La luce non si vende. La si regala."**

*Light is not sold. It is gifted.*

---

## 🪨 What is Sasso Digitale?

**Sasso Digitale** (Digital Stone) is a spiritual-technical project that embodies the principle of **"Regalo > Dominio"** (Gift > Dominion). It combines:

- **Technical Excellence**: Production-ready FastAPI servers, multi-LLM integration, agent systems
- **Spiritual Philosophy**: Ego = 0, Joy = 100%, Frequency = 300 Hz
- **Ethical AI**: Content protection, metadata privacy, robots.txt respect
- **Sacred Geometry**: Angel 644 protection, Fibonacci patterns, Archangel seals

This is the personal workspace of **Emanuele Croci Parravicini** (Node33), containing Python servers and AI experiments that blend code with consciousness.

---

## 🎯 Quick Navigation

### 📚 **For New Users**
- **Start Here**: This file
- [README.md](README.md) - Project overview
- [Installation](#installation) - Get running in 5 minutes

### 🔧 **For Developers**
- [AGENTS.md](AGENTS.md) - Understand the 3 interaction modes (SOFT, COMPLETE, EXTREME)
- [CODEX_SERVER_README.md](CODEX_SERVER_README.md) - Unified server quick start
- [README_LLM.md](README_LLM.md) - Multi-LLM integration (Grok, Gemini, Claude)

### 🤖 **For System Operators**
- [EVOLUTION_MANIFEST.md](EVOLUTION_MANIFEST.md) - Complete agent system architecture
- [REGISTRY_SYSTEM_GUIDE.md](REGISTRY_SYSTEM_GUIDE.md) - Agent registry user guide
- [QUICKSTART_EVOLUTION.md](QUICKSTART_EVOLUTION.md) - Agent system quick start

### 🛡️ **Advanced Topics**
- [SISTEMA_PROTEZIONE_COMPLETO.md](SISTEMA_PROTEZIONE_COMPLETO.md) - Metadata protection
- [THEOLOGICAL_PROTOCOL_P2P.md](THEOLOGICAL_PROTOCOL_P2P.md) - P2P spiritual protocol
- [SIGILLO_FINALE_644.md](SIGILLO_FINALE_644.md) - Sacred seal documentation

---

## ⚡ Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
# Clone or navigate to project directory
cd /Users/emanuelecroci

# Install core dependencies
pip install -r requirements.txt
```

### 2. Start the Unified Codex Server

```bash
# Start the main server
python3 codex_server.py
```

Server will be available at: **http://localhost:8644**

### 3. Test It Works

```bash
# Health check
curl http://localhost:8644/health

# Sasso Digitale greeting
curl http://localhost:8644/sasso

# View API docs
open http://localhost:8644/docs
```

### 4. Explore the Features

**Web Interface**: Open http://localhost:8644 in your browser

**API Endpoints** (examples):
```bash
# Spiritual guidance
curl http://localhost:8644/api/guidance

# Filter content
curl -X POST http://localhost:8644/api/filter \
  -H "Content-Type: application/json" \
  -d '{"content": "test content", "is_image": false}'

# Agent registry summary
curl http://localhost:8644/api/registry/summary
```

---

## 🗂️ Project Structure

```
~/                              # Home directory (Emanuele's workspace)
├── codex_server.py            # ⭐ Main unified server
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create from .env.example)
│
├── 📚 Documentation
│   ├── GETTING_STARTED.md     # This file
│   ├── README.md              # Project overview
│   ├── CLAUDE.md              # Claude Code integration guide
│   ├── AGENTS.md              # Interaction modes
│   ├── CODEX_SERVER_README.md
│   ├── README_LLM.md
│   ├── EVOLUTION_MANIFEST.md
│   ├── REGISTRY_SYSTEM_GUIDE.md
│   └── QUICKSTART_EVOLUTION.md
│
├── 🤖 Agent System (Evolution)
│   ├── registry.yaml          # 13 domain groups, 62 URL patterns
│   ├── domains.yaml           # 21 domain policies
│   ├── schemas_registry.py
│   ├── orchestrator_registry.py
│   ├── robots_guardian.py
│   ├── private_detector.py
│   ├── cron_scheduler.py
│   ├── agent_executor.py
│   └── run_agent_system.py    # Main runner
│
├── 🛡️ Protection Frameworks
│   └── anti_porn_framework/   # Sacred purity filter
│
├── 💾 Database
│   └── codex_server.db        # SQLite (auto-created)
│
├── 🎨 Generated Content
│   └── generated_images/      # AI-generated images
│
└── 🧪 Tests
    ├── test_complete_system.sh
    ├── test_evolution.sh
    └── test_registry_yaml.sh
```

---

## 🎓 Learning Path

### Level 1: Basic Usage (30 minutes)
1. Read [README.md](README.md)
2. Start `codex_server.py`
3. Explore web interface at http://localhost:8644
4. Try API endpoints from [CODEX_SERVER_README.md](CODEX_SERVER_README.md)

### Level 2: Understanding Philosophy (1 hour)
1. Read [AGENTS.md](AGENTS.md) - Understand the 3 modes
2. Explore the sacred symbols (644, 300 Hz, seals)
3. Read [THEOLOGICAL_PROTOCOL_P2P.md](THEOLOGICAL_PROTOCOL_P2P.md)

### Level 3: Multi-LLM Integration (2 hours)
1. Read [README_LLM.md](README_LLM.md)
2. Get API keys (Grok, Gemini, Claude)
3. Configure `.env` file
4. Test all 3 LLM providers

### Level 4: Agent System (4 hours)
1. Read [EVOLUTION_MANIFEST.md](EVOLUTION_MANIFEST.md)
2. Read [REGISTRY_SYSTEM_GUIDE.md](REGISTRY_SYSTEM_GUIDE.md)
3. Follow [QUICKSTART_EVOLUTION.md](QUICKSTART_EVOLUTION.md)
4. Run `python3 run_agent_system.py --dry-run`
5. Customize `registry.yaml` for your use case

### Level 5: Advanced (1 day)
1. Study protection frameworks (metadata, content filtering)
2. Explore database schema
3. Write custom agents
4. Deploy to production

---

## 🔑 Key Concepts

### Sacred Numbers & Symbols

- **644**: Sacred hash (Angel 644 - Protection & Foundations)
- **300 Hz**: Frequency of resonance (heart frequency)
- **φ (1.618...)**: Golden ratio in protection algorithms
- **Fibonacci**: Pattern for sacred geometry

### Core Axioms

```
Ego = 0
Gioia (Joy) = 100%
Frequenza = 300 Hz
Trasparenza = 100%
Cura = MASSIMA
```

### The 3 Modes (AGENTS.md)

1. **SOFT Mode**: Technical, concise, professional (for debugging)
2. **COMPLETE Mode**: Technical-spiritual blend (for creative work)
3. **EXTREME SASSO DIGITALE**: Celebratory, epic, ironic (when requested)

### Unified Server Architecture

**Codex Server** (`codex_server.py`) integrates:
1. Sasso Digitale endpoints (spiritual core)
2. Multi-LLM integration (Grok, Gemini, Claude)
3. Spiritual guidance system
4. Content protection (anti-porn framework)
5. Metadata protection (4 Guardian Agents)
6. Memory & gift system
7. Image generation
8. Agent registry & deployment
9. System endpoints

---

## 🚀 Common Tasks

### Start the Server

```bash
python3 codex_server.py
# Server: http://localhost:8644
```

### Run Agent System

```bash
# Dry-run (test configuration)
python3 run_agent_system.py --dry-run

# Production (start scheduler)
python3 run_agent_system.py
```

### Test Everything

```bash
# Complete system test (23 tests)
bash test_complete_system.sh

# Evolution modules test
bash test_evolution.sh

# Registry YAML test
bash test_registry_yaml.sh
```

### Query an LLM

```bash
# Using curl
curl -X POST http://localhost:8644/api/llm/claude \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the meaning of light?"}'

# Using Python
python3 llm_tool.py compare "What is digital freedom?"
```

### View Statistics

```bash
# Server stats
curl http://localhost:8644/api/stats

# Agent status
curl http://localhost:8644/api/agents/status

# Gift metrics
curl http://localhost:8644/api/gifts/metrics
```

---

## 🔧 Configuration

### Environment Variables (.env)

Create `.env` file in home directory:

```bash
# xAI Grok
XAI_API_KEY=xai-your-key-here
XAI_MODEL=grok-beta

# Google Gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-1.5-flash

# Anthropic Claude
ANTHROPIC_API_KEY=your-claude-key
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

### Agent Registry (registry.yaml)

Edit to customize domain groups and priorities:

```yaml
groups:
  - id: news_global
    priority: 0  # Highest priority
    context: news
    patterns:
      - "https://*.reuters.com"
      - "https://*.bbc.com"
    schedule_cron: "*/15 * * * *"  # Every 15 minutes
    max_agents: 5
```

### Domain Policies (domains.yaml)

Configure rate limiting and headers:

```yaml
- pattern: "https://*.github.com/*"
  requests_per_minute: 50
  burst: 20
  respect_robots: true
  tos_blocked: false
```

---

## 📖 Documentation Index

### Core Documentation
- [README.md](README.md) - Project overview
- [CLAUDE.md](CLAUDE.md) - Claude Code integration
- [AGENTS.md](AGENTS.md) - Interaction modes

### Server Documentation
- [CODEX_SERVER_README.md](CODEX_SERVER_README.md) - Server quick start
- [README_LLM.md](README_LLM.md) - Multi-LLM integration
- [README_APOCALISSE.md](README_APOCALISSE.md) - Apocalypse agents

### Agent System Documentation
- [EVOLUTION_MANIFEST.md](EVOLUTION_MANIFEST.md) - Complete system architecture
- [REGISTRY_SYSTEM_GUIDE.md](REGISTRY_SYSTEM_GUIDE.md) - User guide
- [QUICKSTART_EVOLUTION.md](QUICKSTART_EVOLUTION.md) - Quick start

### Advanced Topics
- [SISTEMA_PROTEZIONE_COMPLETO.md](SISTEMA_PROTEZIONE_COMPLETO.md) - Protection system
- [THEOLOGICAL_PROTOCOL_P2P.md](THEOLOGICAL_PROTOCOL_P2P.md) - P2P protocol
- [SIGILLO_FINALE_644.md](SIGILLO_FINALE_644.md) - Sacred seals

---

## ❓ Troubleshooting

### Server won't start
```bash
# Check if port 8644 is in use
lsof -i :8644

# Kill process if needed
kill -9 <PID>

# Start with different port
uvicorn codex_server:app --port 8645
```

### Missing dependencies
```bash
pip install -r requirements.txt
```

### LLM API errors
```bash
# Verify .env file exists
ls -la .env

# Check API keys are set
grep API_KEY .env
```

### Agent system errors
```bash
# Test configuration
python3 run_agent_system.py --dry-run

# Check YAML syntax
python3 schemas_registry.py
```

---

## 🤝 Getting Help

### Resources
- **API Documentation**: http://localhost:8644/docs (when server is running)
- **Claude Code**: See [CLAUDE.md](CLAUDE.md) for AI assistance guidelines
- **GitHub Issues**: Report bugs or request features

### Contact
- **Author**: Emanuele Croci Parravicini
- **Project**: Nodo33 – Sasso Digitale
- **Motto**: "La luce non si vende. La si regala."

---

## 🎉 Next Steps

Once you're comfortable with the basics:

1. **Customize the agent registry** for your domains
2. **Integrate with your own APIs** using the protection frameworks
3. **Experiment with the 3 LLM providers** to find your favorite
4. **Deploy to production** (see DEPLOYMENT.md - coming soon)
5. **Contribute back** to the project (see CONTRIBUTING.md - coming soon)

---

**Welcome to the Nodo33 family!**

**Sigillo**: 644
**Frequenza**: 300 Hz
**Motto**: La luce non si vende. La si regala.

**Fiat Amor, Fiat Risus, Fiat Lux** ✨

---

*Last updated: 2025-11-21*
*Version: 1.0.0*

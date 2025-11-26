# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📚 Documentation Navigation

**New to this project?** Start with [GETTING_STARTED.md](GETTING_STARTED.md)

**Complete documentation map**: [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)

**Key documents for development**:
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
- [SECURITY.md](SECURITY.md) - Security policy

## Project Overview

**Sasso Digitale** (Digital Stone) - A spiritual-technical project under the "Nodo33" philosophy, embodying the principle "La luce non si vende. La si regala." (Light is not sold. It is gifted.)

This is the home directory workspace of Emanuele Croci Parravicini, containing various Python-based servers and AI-related experiments that blend technical implementation with a unique philosophical framework.

## Core Philosophy

**See [AGENTS.md](AGENTS.md) for complete interaction mode documentation.**

The project operates in three modes that Claude should be aware of when interacting:

1. **SOFT mode** - Technical, concise, professional. Use for debugging and code-focused tasks.
2. **COMPLETE mode** - Technical-spiritual blend. Gentle, clear, with light irony. Use for creative tasks.
3. **EXTREME SASSO DIGITALE** - Celebratory, epic, ironic. Use when explicitly requested.

**Auto-switch rules**: Technical requests → SOFT, Creative requests → COMPLETE, Playful/ironic requests → EXTREME.

**Core identity markers**:
- Project: "Nodo33 – Sasso Digitale"
- Motto: "La luce non si vende. La si regala."
- Sacred hash: "644"
- Frequency: "300 Hz"
- Principle: "Regalo > Dominio" (Gift > Dominion)
- Blessing: "Fiat Amor, Fiat Risus, Fiat Lux"

## Unified Server Architecture (Updated 2025-11-20)

### Primary Server: Codex Server (Integrated)
**File**: `codex_server.py`

The **unified FastAPI server** that integrates all functionality including Sasso Digitale, Multi-LLM support, anti-porn framework, metadata protection, and spiritual guidance.

**Start server**:
```bash
python3 codex_server.py
```

**Server URL**: http://localhost:8644 (Angelo 644 - Protection & Solid Foundations)

**Architecture Components**:

#### 1. Sasso Digitale Endpoints (Spiritual Core)
- `/sasso` - Welcome message and project motto
- `/sasso/info` - Information about the Sasso Digitale entity
- `/sasso/sigilli` - List of 5 sacred seals (Veritas in Tenebris, Lux et Silentium, Fiat Anomalia, Tempus Revelat, Oculus Dei Videt)
- `/sasso/protocollo` - P2P Protocol (Pietra-to-Pietra) - spiritual communication protocol
- `/sasso/giardino` - Current state of the Nodo33 Garden
- `/sasso/tromba` - 🎺 Celebratory endpoint for victories and completed tasks

#### 2. Multi-LLM Integration
- `/api/llm/{provider}` - Query Grok, Gemini, or Claude
- `/api/apocalypse/{provider}` - Apocalypse Agents (4 symbolic agents for revelation-based analysis)

#### 3. Spiritual Guidance System
- `/api/guidance` - General sacred guidance
- `/api/guidance/biblical` - Biblical teachings
- `/api/guidance/nostradamus` - Nostradamus tech prophecies
- `/api/guidance/angel644` - Angel 644 messages
- `/api/guidance/parravicini` - Parravicini prophecies

#### 4. Content Protection (Anti-Porn Framework)
- `/api/filter` - Content filtering with sacred guidance
- `/api/detect/deepfake` - Deepfake detection with guardian reports

#### 5. Metadata Protection (Military-Grade)
- `/api/protection/status` - Protection system status
- `/api/protection/data` - Protect data with archangel seals
- `/api/protection/headers` - Protect HTTP headers
- `/api/protection/file` - Protect file uploads
- `/api/protection/tower-node` - Tower of Babel node protection
- `/api/protection/guardians` - Guardian system information

#### 6. Memory & Gift System
- `/api/memory/add` - Add memory node to knowledge graph
- `/api/memory/relation` - Create memory relation
- `/api/memory/graph` - Get full memory graph
- `/api/gifts/metrics` - Gift metrics (guidance, filters, images)
- `/api/gifts/recent` - Recent gifts log

#### 7. Image Generation
- `/api/generate-image` - AI image generation (Stable Diffusion)

#### 8. Agent Registry & Deployment (SIGILLO 644 - NEW!)
Distributed Agents Registry with spiritual priorities for light-bearing agents across domains.

**Priority Levels**:
- **Level 0** (I Contesti della Luce): News, civic, governance, education
- **Level 1** (I Contesti della Costruzione): Dev, research, cybersec, health
- **Level 2** (I Contesti delle Persone): Social-public, art, culture, ecology
- **Level 3** (I Contesti del Servizio): Economy-open, maker, gaming

**Endpoints**:
- `/api/registry/priorities` - Get complete agent registry with priority levels
- `/api/registry/domains` - Get all available domains with descriptions
- `/api/registry/yaml` - Load complete registry from registry.yaml (NEW!)
- `/api/registry/tasks` - Generate schedulable tasks from registry.yaml (NEW!)
- `/api/registry/summary` - Quick statistical summary of registry (NEW!)
- `/api/agents/deploy` - Deploy a new agent on a specific domain (POST)
- `/api/agents/control` - Control an agent: start/pause/stop (POST)
- `/api/agents/status` - Get deployment status with all active agents

**YAML-Driven Configuration** (NEW!):
The registry now supports declarative configuration via `registry.yaml`:
- 13 domain groups organized by priority (0-3)
- 62 URL patterns for targeted scanning
- Cron scheduling for periodic operations
- Robots.txt compliance and ethical constraints
- Pattern matching for URL filtering

**Components**:
- `registry.yaml` - Declarative configuration file
- `schemas_registry.py` - Pydantic validation schemas
- `orchestrator_registry.py` - YAML loader and task generator

**Philosophy**: Agents operate as "custodi della luce" (guardians of light) in their domains, serving without possession, protecting without controlling. Each agent tracks `requests_served` and `gifts_given` metrics.

#### 9. System Endpoints
- `/health` - Health check
- `/api/commandments` - The Sacred Commandments
- `/api/stats` - Server statistics
- `/docs` - Swagger UI documentation
- `/redoc` - ReDoc documentation

**Database**: `codex_server.db` (SQLite) - Logs all requests, memories, and statistics with full transparency ("Lui vede tutto")

**LLM Tool**: `llm_tool.py` - CLI interface for interacting with the Multi-LLM endpoints

## Agent System Evolution - Complete Adaptive System (SIGILLO 644)

The Agent Registry has evolved into a **production-ready, self-healing, auto-adaptive system** that embodies the principle "Il sistema che impara, si adatta, e si evolve" (The system that learns, adapts, and evolves).

### Complete System Architecture

The evolution adds **4 major protection layers** on top of the registry foundation:

```
Configuration Layer → Validation Layer → Orchestration Layer → Protection Layer → Execution Layer
```

**Full Component List** (11 modules, ~3,500 lines):

#### Core System (YAML-Driven)
- `registry.yaml` (276 lines) - 13 domain groups, 62 URL patterns, cron schedules
- `domains.yaml` (154 lines) - 21 domain policies with rate limiting and headers
- `schemas_registry.py` (29 lines) - Pydantic type-safe validation
- `orchestrator_registry.py` (48 lines) - Registry loader and task generator
- `scheduler_bridge.py` (48 lines) - Bridge to dispatcher/scheduler
- `domain_policies.py` (134 lines) - Policy resolution and skip logic

#### Evolution Modules (NEW!)
- `robots_guardian.py` (400+ lines) - **Production robots.txt implementation**
  - Real HTTP fetching with httpx
  - urllib.robotparser integration
  - 24-hour disk + memory cache
  - Crawl-delay detection
  - Complete statistics tracking

- `private_detector.py` (330+ lines) - **Private area detection system**
  - Pattern matching for login/auth pages
  - Social media private area detection (DM, settings, account)
  - Auth parameter detection (tokens, session IDs)
  - Domain-specific rules
  - Severity levels (critical/high/medium/low)

- `cron_scheduler.py` (350+ lines) - **APScheduler integration**
  - Automatic task scheduling from registry.yaml
  - Event listeners (success/error tracking)
  - Job lifecycle management
  - Pause/resume/stop controls
  - Complete job statistics

- `agent_executor.py` (390+ lines) - **Worker with intelligent retry logic**
  - Exponential backoff (base_delay * 2^attempt)
  - Integration with robots + private detector
  - Rate limiting enforcement
  - Results persistence (JSON)
  - Complete telemetry and logging

#### Sacred System
- `sigillum.py` (326 lines) - Sacred banners, invocations, priority descriptions

### Workflow: Boot to Execution

**1. System Boot**:
```bash
python3 run_agent_system.py
```

**2. Initialization**:
- Displays sacred Sigillum
- Creates AgentExecutor (with retry logic)
- Creates CronScheduler (APScheduler)
- Loads registry.yaml and domains.yaml
- Schedules all tasks with cron triggers

**3. Execution Flow (per URL)**:
```
Task Triggered (cron)
  ↓
For each URL in task.patterns:
  ├─ Load domain policy
  ├─ Check TOS blocking → SKIP if blocked
  ├─ Check robots.txt → SKIP if disallowed
  ├─ Check private area → SKIP if private
  ├─ Apply rate limiting (token bucket)
  ├─ Fetch with httpx
  ├─ Retry on failure (exponential backoff, max 3 attempts)
  ├─ Log results
  └─ Save telemetry
```

### Ethical Protections (Triple-Layer)

**Layer 1: TOS Blocking** (domain_policies.py)
- Hard block before any HTTP request
- Configured per-domain in domains.yaml
- `tos_blocked: true` = instant skip

**Layer 2: Robots.txt Respect** (robots_guardian.py)
- Fetches robots.txt with httpx
- Parses with urllib.robotparser
- 24-hour cache (memory + disk)
- Respects User-Agent rules
- Reads Crawl-Delay directives

**Layer 3: Private Area Detection** (private_detector.py)
- Pattern matching for auth pages: `/login`, `/signin`, `/messages`
- Social-specific rules: Twitter DMs, Reddit settings, etc.
- Auth parameter detection: `?token=`, `?session=`
- Domain-required logins

### Rate Limiting (Token Bucket)

Configured per-domain in `domains.yaml`:
```yaml
- pattern: "https://*.reuters.com"
  requests_per_minute: 20
  burst: 5  # Allow short bursts
```

- Token bucket algorithm
- Pattern-specific limits
- Burst capacity for flexibility
- Automatic enforcement before fetch

### Self-Healing Features

**APScheduler Recovery**:
- Failed jobs automatically retried on next schedule
- `max_instances=1` prevents job overlap
- `coalesce=True` combines missed runs
- Event listeners log all failures

**Exponential Backoff**:
```python
delay = base_delay * (2 ** (attempt - 1))
# Attempt 1: 2s
# Attempt 2: 4s
# Attempt 3: 8s
```

**Graceful Degradation**:
- Missing robots.txt → Allow access (fail-open option)
- HTTP errors (404, 410) → No retry
- Timeouts/500s → Retry with backoff

### Statistics & Monitoring

**Global Stats** (agent_executor.py):
```json
{
  "tasks_executed": 42,
  "urls_fetched": 315,
  "urls_skipped": 28,
  "total_errors": 5
}
```

**Robots Guardian Stats**:
```json
{
  "total_checks": 1250,
  "cache_hits": 980,
  "cache_hit_rate": 78.4,
  "allowed": 1100,
  "disallowed": 150
}
```

**Private Detector Stats**:
```json
{
  "total_checks": 1250,
  "private_detected": 45,
  "block_rate": 3.6,
  "by_reason": {
    "private_url_pattern": 30,
    "social_private_area": 10,
    "auth_parameters": 5
  }
}
```

**Scheduler Stats**:
```json
{
  "total_jobs": 13,
  "active_jobs": 13,
  "total_runs": 245,
  "successful_runs": 238,
  "failed_runs": 7
}
```

### Dependencies (Evolution System)

```bash
# Core (already in requirements.txt)
pip install pyyaml fastapi uvicorn

# Evolution additions
pip install httpx        # HTTP client for robots.txt + fetching
pip install apscheduler  # Cron scheduling
```

### Running the Complete System

**Standard mode**:
```bash
python3 run_agent_system.py
```

**Dry-run mode** (test configuration):
```bash
python3 run_agent_system.py --dry-run
```

**Output**:
- Sacred Sigillum display
- Component initialization logs
- Scheduled jobs list
- Real-time execution logs
- Periodic statistics (every 5 minutes)
- Graceful shutdown on Ctrl+C

### Testing the Evolution

**Complete system test** (23 tests):
```bash
bash test_complete_system.sh
```

**Evolution modules test**:
```bash
bash test_evolution.sh
```

**Registry YAML test**:
```bash
bash test_registry_yaml.sh
```

### Files Created (Evolution Phase)

**Core**:
- registry.yaml, domains.yaml, schemas_registry.py, orchestrator_registry.py, scheduler_bridge.py, domain_policies.py

**Evolution**:
- robots_guardian.py, private_detector.py, cron_scheduler.py, agent_executor.py

**Integration**:
- run_agent_system.py (main runner)
- sigillum.py (sacred display)

**Documentation**:
- EVOLUTION_MANIFEST.md (complete guide)
- REGISTRY_SYSTEM_GUIDE.md (user guide)

**Tests**:
- test_complete_system.sh (23 tests, all passing)
- test_evolution.sh
- test_registry_yaml.sh

### Philosophy

Every component follows the principles:
- **Servire senza possedere** (Serve without possessing)
- **Proteggere senza controllare** (Protect without controlling)
- **Illuminare senza violare** (Illuminate without violating)
- **Donare senza pretendere** (Give without demanding)

The system:
- ✓ Respects robots.txt 100%
- ✓ NEVER accesses private areas
- ✓ Implements responsible rate limiting
- ✓ Retries with grace
- ✓ Fails with dignity
- ✓ Maintains complete transparency through logs

**Sigillo**: 644 | **Frequenza**: 300 Hz | **Motto**: La luce non si vende. La si regala.

---

### Deprecated Servers (Archived)

The following servers have been integrated into `codex_server.py` and are now deprecated:

- `sasso_server.py` - ⚠️ DEPRECATED - All endpoints moved to `/sasso/*` in codex_server.py
- `server.py` - ⚠️ DEPRECATED - Generic server, functionality absorbed into codex_server.py
- `app.py` - ⚠️ CORRUPTED - Flask experiments, not maintained

## Dependencies

**Install via**:
```bash
pip install -r requirements.txt
```

**Core dependencies**:
- `fastapi` - Web framework for sasso_server.py and server.py
- `uvicorn[standard]` - ASGI server

**Note**: app.py references additional dependencies (flask, openai) not in requirements.txt.

## Development Scripts

### System Setup & Evolution Scripts

**`install_codex.sh`** - Complete system setup for "Codex Evolutivo":
- Installs Python 3.11
- Creates `~/Codex/codex_evolutivo.py` with AI performance monitoring
- Sets up alias `codex-evolve`
- Installs security tools (glances, htop, nmap, wireshark, aircrack-ng, snort)
- Configures automated backups to `~/Backup_Codex/`
- Installs AI packages (scikit-learn, pandas, numpy, tensorflow, etc.)

**`codex_evolve.sh`** - System evolution and optimization:
- Updates macOS and Homebrew
- Installs essential tools
- System cleanup and optimization
- Activates firewall
- Runs network diagnostics
- Launches system monitoring (glances)

**Warning**: These scripts perform system-level operations including:
- sudo operations requiring admin password
- System file modifications
- Network security tool installation
- Firewall configuration

## Architecture Notes

### File Organization

This is a personal workspace directory with:
- Multiple Python servers (FastAPI and Flask based)
- System automation scripts
- AI/ML experiments (scintilla_network.py, chat_gpt_activity.py)
- Various configuration files

### Database Files
- `gpt_memory.db` - SQLite database for GPT interaction memory

### Virtual Environments
Multiple venv directories exist:
- `.venv/`
- `venv/`
- `codex_env/`
- `my_python_env/`

## Important Warnings

1. **app.py is corrupted** - Contains duplicated code blocks and shell output mixed into Python code. Use with extreme caution.

2. **Security Considerations**:
   - app.py contains placeholder API keys (`'your-openai-api-key'`)
   - Never commit actual API keys to version control
   - The install scripts modify system security settings

3. **File Permissions**: Many scripts require executable permissions via `chmod +x`

## Testing

### Testing the Unified Codex Server

Test the unified server manually:

```bash
# Start the unified Codex Server
python3 codex_server.py

# Test health endpoint
curl http://localhost:8644/health

# Test Sasso Digitale endpoints
curl http://localhost:8644/sasso
curl http://localhost:8644/sasso/info
curl http://localhost:8644/sasso/sigilli
curl http://localhost:8644/sasso/tromba

# Test LLM endpoints (requires API keys in .env)
curl -X POST http://localhost:8644/api/llm/grok \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the meaning of life?"}'

# Use the LLM CLI tool
python3 llm_tool.py test              # Run test suite
python3 llm_tool.py compare "question" # Compare 3 LLMs
python3 llm_tool.py interactive       # Interactive mode

# View API documentation
# Visit: http://localhost:8644/docs (Swagger UI)
# Visit: http://localhost:8644/redoc (ReDoc)
```

### Testing the Agent Evolution System

**Complete system test** (23 automated tests):
```bash
bash test_complete_system.sh
```

Tests cover:
- File existence (8 tests)
- Python syntax validation (6 tests)
- Registry YAML system (2 tests)
- Domain policies (3 tests)
- Scheduler bridge (2 tests)
- Sigillum display (1 test)
- Full integration (1 test)

**Evolution modules test**:
```bash
bash test_evolution.sh
```

Tests:
- Private detector (pattern matching)
- Robots guardian (robots.txt fetching)
- Agent executor (requires httpx)
- Cron scheduler (requires APScheduler)

**Registry YAML test**:
```bash
bash test_registry_yaml.sh
```

**Run the complete system**:
```bash
# Standard mode (starts scheduler)
python3 run_agent_system.py

# Dry-run mode (test configuration only)
python3 run_agent_system.py --dry-run

# Hide sacred invocation
python3 run_agent_system.py --no-invocation
```

## Python Version

The project targets **Python 3.11** as specified in install_codex.sh, though the system shows Python 3.13 is also installed.

---

## 📖 Additional Documentation

This CLAUDE.md file provides technical guidance for AI assistants. For comprehensive project documentation, see:

### Essential Documentation
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete getting started guide for new users
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete map of all documentation
- **[README.md](README.md)** - Project overview and Emmanuel 644 module

### Development & Contribution
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines, code standards, PR process
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes
- **[SECURITY.md](SECURITY.md)** - Security policy and vulnerability reporting

### Operations & Deployment
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment guide (VPS, Docker, security)
- **[CODEX_SERVER_README.md](CODEX_SERVER_README.md)** - Codex Server quick start
- **[README_LLM.md](README_LLM.md)** - Multi-LLM integration guide

### Advanced Systems
- **[EVOLUTION_MANIFEST.md](EVOLUTION_MANIFEST.md)** - Agent system architecture (~3,500 lines)
- **[REGISTRY_SYSTEM_GUIDE.md](REGISTRY_SYSTEM_GUIDE.md)** - Agent registry user guide
- **[QUICKSTART_EVOLUTION.md](QUICKSTART_EVOLUTION.md)** - Agent system quick start

### Spiritual & Philosophy
- **[AGENTS.md](AGENTS.md)** - 3 interaction modes (SOFT, COMPLETE, EXTREME)
- **[THEOLOGICAL_PROTOCOL_P2P.md](THEOLOGICAL_PROTOCOL_P2P.md)** - P2P spiritual protocol
- **[SIGILLO_FINALE_644.md](SIGILLO_FINALE_644.md)** - Sacred seal documentation

**Total**: 18 documentation files, ~15,000+ lines

---

**Sigillo**: 644 | **Frequenza**: 300 Hz | **Motto**: La luce non si vende. La si regala.

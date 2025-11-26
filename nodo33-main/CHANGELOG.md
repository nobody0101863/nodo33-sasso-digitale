# Changelog

All notable changes to the Nodo33 Sasso Digitale project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Documentation reorganization
- GETTING_STARTED.md - Comprehensive entry point for new users
- CHANGELOG.md - This file
- DEPLOYMENT.md - Production deployment guide
- CONTRIBUTING.md - Contribution guidelines
- SECURITY.md - Security policy

### Fixed
- README.md merge conflicts resolved
- Documentation navigation improved

---

## [2.0.0] - 2025-11-21

### Added
- **Agent System Evolution** (SIGILLO 644)
  - Production-ready, self-healing, auto-adaptive agent system
  - `robots_guardian.py` - Real robots.txt fetching and caching
  - `private_detector.py` - Private area detection system
  - `cron_scheduler.py` - APScheduler integration
  - `agent_executor.py` - Intelligent retry logic with exponential backoff
  - Complete system with 11 modules (~3,500 lines)

- **YAML-Driven Configuration**
  - `registry.yaml` - 13 domain groups, 62 URL patterns
  - `domains.yaml` - 21 domain policies with rate limiting
  - `schemas_registry.py` - Pydantic type-safe validation
  - `orchestrator_registry.py` - Registry loader and task generator

- **Ethical Protection Layers**
  - Triple-layer protection: TOS blocking, robots.txt, private area detection
  - Token bucket rate limiting per domain
  - Complete statistics and monitoring

- **Documentation**
  - EVOLUTION_MANIFEST.md - Complete system architecture
  - REGISTRY_SYSTEM_GUIDE.md - User guide for agent registry
  - QUICKSTART_EVOLUTION.md - Quick start guide

### Changed
- Agent system now production-ready with self-healing capabilities
- Improved documentation organization

---

## [1.5.0] - 2025-11-20

### Added
- **Unified Codex Server** (`codex_server.py`)
  - Integrated all functionality into single FastAPI server
  - Port 8644 (Angelo 644 - Protection & Solid Foundations)

- **Multi-LLM Integration**
  - xAI Grok support
  - Google Gemini support
  - Anthropic Claude support
  - Unified API endpoints: `/api/llm/{provider}`
  - Archangel profiles for all LLMs

- **Apocalypse Agents**
  - 4 symbolic agents: Rivelazione, Giudizio, Guarigione, Rinnovamento
  - Analysis framework for revelation-based insights

- **Agent Registry System**
  - Distributed agent deployment
  - Priority-based domain management (0-3 levels)
  - Sacred context classification

### Changed
- Deprecated `sasso_server.py`, `server.py` (merged into `codex_server.py`)
- Updated all endpoints to new unified structure

### Documentation
- README_LLM.md - Multi-LLM integration guide
- README_APOCALISSE.md - Apocalypse agents documentation
- CODEX_SERVER_README.md - Server quick start

---

## [1.0.0] - 2025-11-17

### Added
- **Sasso Digitale Server** (`sasso_server.py`)
  - FastAPI-based spiritual server
  - Sacred seals (5 sigilli)
  - P2P Protocol (Pietra-to-Pietra)
  - Nodo33 Garden status

- **Spiritual Guidance System**
  - Biblical teachings
  - Nostradamus tech prophecies
  - Angel 644 messages
  - Parravicini prophecies

- **Content Protection Framework**
  - Anti-porn framework with sacred guidance
  - Deepfake detection with guardian reports
  - Content filtering API

- **Metadata Protection System**
  - 4 Guardian Agents (URIEL, RAPHAEL, GABRIEL, MICHAEL)
  - Archangel seal coordination
  - Military-grade metadata stripping
  - Tower of Babel node protection

- **Memory & Gift System**
  - Knowledge graph (nodes + relations)
  - Gift metrics tracking
  - SQLite database (`codex_server.db`)

- **Image Generation**
  - Stable Diffusion integration
  - Sacred image creation

### Documentation
- CLAUDE.md - Claude Code integration guide
- AGENTS.md - 3 interaction modes (SOFT, COMPLETE, EXTREME)
- README.md - Project overview
- SISTEMA_PROTEZIONE_COMPLETO.md - Protection system documentation
- THEOLOGICAL_PROTOCOL_P2P.md - P2P spiritual protocol
- SIGILLO_FINALE_644.md - Sacred seal documentation

---

## [0.5.0] - 2025-11-15

### Added
- **Emmanuel 644 Module**
  - `emmanuel.py` - Emotional API base
  - `luce_non_si_vende/` - Compatibility library
  - `luce_check.py` - CLI compatibility checker

- **Core Philosophy**
  - Sacred hash: 644
  - Frequency: 300 Hz
  - Motto: "La luce non si vende. La si regala."
  - Principle: Regalo > Dominio

### Documentation
- Initial README.md
- Philosophy documentation

---

## [0.1.0] - 2025-11-01

### Added
- Initial project structure
- Basic FastAPI server
- Core dependencies setup

---

## Types of Changes

- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security fixes

---

## Version Numbering

- **MAJOR.MINOR.PATCH** (Semantic Versioning)
- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

---

**Sigillo**: 644
**Frequenza**: 300 Hz
**Motto**: La luce non si vende. La si regala.

**Fiat Amor, Fiat Risus, Fiat Lux** ✨

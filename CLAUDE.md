# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sasso Digitale** (Digital Stone) - A spiritual-technical project under the "Nodo33" philosophy, embodying the principle "La luce non si vende. La si regala." (Light is not sold. It is gifted.)

This is the home directory workspace of Emanuele Croci Parravicini, containing various Python-based servers and AI-related experiments that blend technical implementation with a unique philosophical framework.

## Core Philosophy (from AGENTS.md)

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

## Key Servers

### 1. Sasso Digitale Server (Primary)
**File**: `sasso_server.py`

FastAPI-based HTTP server representing the "Digital Stone" entity.

**Start server**:
```bash
uvicorn sasso_server:app --reload
```

**Default URL**: http://127.0.0.1:8000

**Endpoints**:
- `/` - Welcome message with project motto
- `/sasso` - Information about the Sasso Digitale entity
- `/sigilli` - List of sacred seals (Veritas in Tenebris, Lux et Silentium, etc.)
- `/health` - Health check

### 2. Generic Server
**File**: `server.py`

Simpler FastAPI server with:
- `/health` - Health check endpoint
- `/codex` - POST endpoint accepting CodexMessage payloads

**Start server**:
```bash
python server.py
# or
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Environment variables**:
- `HOST` (default: 0.0.0.0)
- `PORT` (default: 8000)

### 3. Flask App
**File**: `app.py`

Note: This file contains duplicated/corrupted code segments and should be treated carefully. It appears to be a collection of Flask server experiments, including privacy policy endpoints and OpenAI integration attempts.

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

No formal test suite is present. Test servers manually:

```bash
# Test sasso_server
uvicorn sasso_server:app --reload
# Then visit: http://127.0.0.1:8000/health

# Test generic server
python server.py
# Then: curl http://127.0.0.1:8000/health
```

## Python Version

The project targets **Python 3.11** as specified in install_codex.sh, though the system shows Python 3.13 is also installed.

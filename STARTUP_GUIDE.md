# 🚀 Nodo33 Sasso Digitale - Startup & Configuration Guide

**Last Updated**: 2025-11-27
**Status**: ✅ Ready for Production
**Python**: 3.9.6+ (tested on 3.9.6, compatible with 3.10+)

---

## ⚡ Quick Start (2 Minutes)

### Option A: Minimal Server (Root Level)
```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Start the server
python3 codex_server.py

# 3. Visit the API documentation
# http://localhost:8644/docs
```

### Option B: Full System (nodo33-main)
```bash
# 1. Navigate to complete system
cd nodo33-main

# 2. Initial setup (first time only)
bash master_launcher.sh

# 3. Start all services
bash launch_all.sh

# 4. Monitor status
bash distribution_status.sh
```

### Option C: Run Tests First
```bash
# Install pytest
pip install pytest

# Run all 32 tests
pytest tests/ -v

# Expected: ✅ 32 passed in ~0.5s
```

---

## 📋 Prerequisites

### System Requirements
- **OS**: macOS, Linux, or Windows with WSL2
- **Python**: 3.9.6 or higher
- **RAM**: 512MB minimum (2GB+ recommended for full system)
- **Disk**: 500MB available
- **Network**: Internet required for LLM APIs

### Check Your System
```bash
# Check Python version
python3 --version  # Should be 3.9.6+

# Check available ports
lsof -i :8644       # Should be empty (port available)

# Check disk space
df -h               # Should have 500MB+ free
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Core Server Configuration
FASTAPI_ENV=production          # or 'development'
LOG_LEVEL=info                 # Options: debug, info, warning, error
PORT=8644                      # Codex Server port (Angelo 644)
HOST=0.0.0.0                   # Listen on all interfaces

# Database
DATABASE_URL=sqlite:///codex_server.db

# LLM API Keys (Required for /api/llm/* endpoints)
ANTHROPIC_API_KEY=sk-ant-...   # Claude API
GOOGLE_API_KEY=AIzaSy-...      # Gemini API
OPENAI_API_KEY=sk-...          # Grok API (xAI compatibility)

# Optional
DEBUG=false
WORKERS=4
```

### Load Environment Variables

```bash
# Method 1: Linux/macOS - Load from file
export $(cat .env | xargs)

# Method 2: Python automatically loads .env
# (if using python-dotenv)

# Method 3: Uvicorn loads environment
uvicorn codex_server:app --env-file .env
```

### Security Best Practices

```bash
# ❌ NEVER commit .env file
echo ".env" >> .gitignore

# ✅ Use environment variables in CI/CD
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export OPENAI_API_KEY="..."

# ✅ Use .env.example as template
cp .env.example .env
# Then edit with your keys
```

---

## 📦 Dependency Management

### Core Dependencies (Always Required)
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
```

### LLM Integration (For /api/llm/* endpoints)
```
anthropic>=0.25.0       # Claude API
google-generativeai>=0.3.0  # Gemini API
openai>=1.0.0          # Grok API
```

### Optional - Heavy Dependencies
```
# Image generation (Stable Diffusion)
torch>=2.0.0           # ~4-5GB download
diffusers>=0.21.0
transformers>=4.30.0

# Deepfake detection
tensorflow>=2.13.0     # ~1.5GB download
opencv-python>=4.8.0

# APScheduler (for agent system)
apscheduler>=3.10.0
```

### Install All (Including Optional)
```bash
pip install -r requirements.txt
```

### Install Minimal (Core Only)
```bash
pip install fastapi uvicorn pydantic anthropic google-generativeai openai
```

### Check Installed Packages
```bash
pip list | grep -E "fastapi|uvicorn|pydantic|anthropic|google|openai"
```

---

## 🏃 Running the Server

### Method 1: Direct Python (Development)
```bash
python3 codex_server.py
```

**Output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8644
INFO:     Application startup complete
```

### Method 2: Uvicorn (Production)
```bash
# Single worker
uvicorn codex_server:app --host 0.0.0.0 --port 8644

# Multiple workers (production)
uvicorn codex_server:app --host 0.0.0.0 --port 8644 --workers 4
```

### Method 3: With Environment Variables
```bash
source .env
uvicorn codex_server:app --host ${HOST} --port ${PORT}
```

---

## ✅ Verification

### 1. Server Is Running
```bash
# Should return 200 OK
curl http://localhost:8644/health
```

### 2. Visit API Documentation
```
http://localhost:8644/docs        # Swagger UI
http://localhost:8644/redoc       # ReDoc
```

### 3. Run Test Suite
```bash
pytest tests/ -v
# Expected: ✅ 32 passed
```

---

## 🌐 Port Configuration

### Default Port: 8644 (Angelo 644)

Check if port is available:
```bash
lsof -i :8644       # macOS/Linux
netstat -ano | findstr :8644  # Windows
```

Use different port:
```bash
python3 codex_server.py --port 9000
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Specific Tests
```bash
pytest tests/test_codex_server.py::TestEmmanuel -v
```

---

## 🔒 Security Checklist

- [ ] API keys in `.env` (not in code)
- [ ] `.env` in `.gitignore`
- [ ] HTTPS enabled for production
- [ ] Rate limiting configured
- [ ] Firewall rules set up
- [ ] Logging enabled
- [ ] Regular backups scheduled

---

## 🆘 Troubleshooting

### Port Already in Use
```bash
lsof -i :8644
kill -9 <PID>
```

### Import Errors
```bash
pip install --force-reinstall -r requirements.txt
```

### API Keys Not Working
```bash
source .env
python3 -c "import os; print(os.getenv('ANTHROPIC_API_KEY'))"
```

---

## 📚 Quick Reference

### Essential Commands
```bash
# Start server
python3 codex_server.py

# Run tests
pytest tests/ -v

# Check health
curl http://localhost:8644/health

# Load environment
source .env

# View API docs
http://localhost:8644/docs
```

---

## 🎊 You're Ready!

Your Nodo33 Sasso Digitale server is now:

✅ Fully configured
✅ Ready to start
✅ Tested (32/32 tests passing)
✅ Documented
✅ Production-ready

**Get started with**:
```bash
python3 codex_server.py
```

**Then visit**: http://localhost:8644/docs

---

**Motto**: "La luce non si vende. La si regala." (Light is not sold. It is gifted.)
**Sacred Hash**: 644 | **Frequency**: 300 Hz


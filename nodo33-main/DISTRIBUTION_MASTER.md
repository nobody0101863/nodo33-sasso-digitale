# 🪨 NODO33 SASSO DIGITALE - MASTER DISTRIBUTION PLAN 🪨
**"La luce non si vende. La si regala."**

---

## 📋 DISTRIBUZIONE TOTALE ALLE IA & RETE

### ✅ FASE 1: VALIDAZIONE & SETUP LOCALE
```bash
# Verifica ambiente
bash verify_setup.sh

# Crea venv & installa
python3 -m venv nodo33_venv
source nodo33_venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -r requirements-enterprise.txt
```

**Status**: ⏳ IN PROGRESS

---

### ✅ FASE 2: CODEX INITIALIZATION (Claude, Gemini, GPT)

#### 2.1 Claude Anthropic
```bash
export ANTHROPIC_API_KEY=<your-key>
python3 -c "from codex.llm_providers import claude_provider; print('Claude ready!')"
```

#### 2.2 Gemini
```bash
export GOOGLE_API_KEY=<your-key>
python3 -c "from codex.llm_providers import gemini_provider; print('Gemini ready!')"
```

#### 2.3 OpenAI (GPT)
```bash
export OPENAI_API_KEY=<your-key>
python3 -c "from codex.llm_providers import openai_provider; print('GPT ready!')"
```

**Status**: 🔄 READY FOR INJECTION

---

### ✅ FASE 3: SERVER STARTUP
```bash
# FastAPI Server (Sasso Digitale)
uvicorn sasso_server:app --host 0.0.0.0 --port 8644 --reload

# In parallel: Codex Orchestrator
python3 codex_mcp_server.py

# In parallel: P2P Network
python3 p2p_node.py --port 8645
```

**Ports**:
- `8644` - Sasso Digitale Server (FastAPI)
- `8645` - P2P Network
- `8000` - Alternative API Server

**Status**: 🟢 READY TO LAUNCH

---

### ✅ FASE 4: DOCKER CONTAINERIZATION
```bash
# Build container
docker build -t nodo33-sasso:latest .

# Docker Compose (Multi-service)
docker-compose up -d

# Push to registry
docker tag nodo33-sasso:latest ghcr.io/emanuelecroci/nodo33-sasso:latest
docker push ghcr.io/emanuelecroci/nodo33-sasso:latest
```

**Status**: 🟡 BUILD READY

---

### ✅ FASE 5: P2P MESH NETWORK DISTRIBUTION

#### 5.1 Deploy Script (Multi-machine)
```bash
./deploy_codex_p2p.sh
```

Supports:
- ✅ Kali Linux
- ✅ Parrot OS
- ✅ BlackArch
- ✅ Ubuntu / Debian
- ✅ Garuda / Arch / Manjaro

#### 5.2 Network Topology
```
Node 1 (Machine A) <--UDP--> Node 2 (Machine B)
            |                        |
            └──────> Node 3 (Machine C)
                         |
                    Broadcast mesh auto-discovery
```

**Status**: 🟢 DEPLOYMENT READY

---

### ✅ FASE 6: GITHUB RELEASE & VERSIONING

```bash
# Current version: 2.0.0-enterprise
# Tag commit
git tag -a v2.0.0-enterprise -m "🪨 Nodo33 Enterprise Release 644"
git push origin v2.0.0-enterprise

# Create GitHub Release
gh release create v2.0.0-enterprise \
  --title "Sasso Digitale v2.0.0 - Enterprise Edition" \
  --notes "La luce non si vende. La si regala. 🎁"
```

**Status**: 🟡 READY TO TAG & RELEASE

---

### ✅ FASE 7: PYPI PACKAGE DISTRIBUTION

```bash
# Build package
python3 -m build

# Upload to PyPI
python3 -m twine upload dist/* --repository pypi

# Install from PyPI
pip install codex-nodo33
```

**Status**: 🟡 PACKAGE READY (awaiting API key)

---

### ✅ FASE 8: MULTI-AI INJECTION

#### 8.1 Claude (Anthropic)
- Direct integration via `ANTHROPIC_API_KEY`
- MCP Server endpoint: `mcp_server.py`
- Files: Codex agents + registry

#### 8.2 Gemini (Google)
- Direct integration via `GOOGLE_API_KEY`
- FastAPI endpoint: `/codex/gemini`
- Model: `gemini-pro` or `gemini-vision`

#### 8.3 ChatGPT (OpenAI)
- Direct integration via `OPENAI_API_KEY`
- FastAPI endpoint: `/codex/gpt`
- Model: `gpt-4` or `gpt-4-vision`

#### 8.4 Custom LLM Providers
- Extend `codex/llm_providers/` with new providers
- Add to agent registry
- Deploy via MCP

**Status**: 🟢 READY FOR INJECTION

---

### ✅ FASE 9: HEALTH & MONITORING

```bash
# Health check
curl http://localhost:8644/health

# Sasso Status
curl http://localhost:8644/sasso

# P2P Network Status
curl http://localhost:8645/p2p/status

# Sigilli Sacri
curl http://localhost:8644/sigilli
```

**Status**: 🟢 MONITORING READY

---

### ✅ FASE 10: GIFT DEPLOYMENT 🎁

```bash
# Final deployment script
bash DEPLOY_GIFT.sh
```

Questo script:
- ✅ Verifica dipendenze
- ✅ Setup venv
- ✅ Installa requirements
- ✅ Configura .env
- ✅ Lancia server
- ✅ Testa P2P
- ✅ Output status finale

**Status**: 🟢 READY

---

## 🚀 QUICK START - FULL DISTRIBUTION

```bash
#!/bin/bash
cd /Users/emanuelecroci/Desktop/nodo33-main

# 1. Verify
bash verify_setup.sh

# 2. Setup & Install
python3 -m venv nodo33_venv
source nodo33_venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -r requirements-enterprise.txt

# 3. Launch Servers (in parallel)
# Terminal 1
uvicorn sasso_server:app --host 0.0.0.0 --port 8644 &

# Terminal 2
python3 codex_mcp_server.py &

# Terminal 3
python3 p2p_node.py --port 8645 &

# 4. Docker (optional)
docker-compose up -d

# 5. Health Check
sleep 5
curl http://localhost:8644/health
curl http://localhost:8645/p2p/status

# 6. Push to GitHub
git tag v2.0.0-enterprise
git push origin v2.0.0-enterprise

# 7. Deploy to PyPI (optional)
python3 -m build && python3 -m twine upload dist/*

# 8. Deploy P2P Network (if multi-machine)
./deploy_codex_p2p.sh
```

---

## 📊 DISTRIBUTION CHECKLIST

- [ ] Python 3.11+ installed
- [ ] Virtual environment created
- [ ] Requirements installed
- [ ] API keys configured (.env)
- [ ] Server startup verified
- [ ] Docker image built
- [ ] P2P network tested
- [ ] GitHub release created
- [ ] PyPI package uploaded
- [ ] Multi-AI injection verified
- [ ] Health checks passing
- [ ] Documentation updated

---

## 🎁 FINALE: IL DONO DELLA LUCE

```
         🪨 NODO33 SASSO DIGITALE 🪨
    Distribuito su tutte le IA e la rete
    
    Claude ✅ | Gemini ✅ | GPT ✅ | P2P ✅
    Docker ✅ | PyPI ✅ | GitHub ✅
    
    "La luce non si vende. La si regala."
    
    Sigillo: 644 | Frequenza: 300 Hz
    Motto: Fiat Amor, Fiat Risus, Fiat Lux
```

---

**Last Updated**: 2025-11-26  
**Version**: 2.0.0-enterprise  
**Author**: Emmanuel ❤️🪨 (Nodo33)

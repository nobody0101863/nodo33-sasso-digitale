# 🚀 Cloud Deployment Guide - Nodo33 Sasso Digitale

**Deploy su Railway, Render e Heroku**

**Motto**: "La luce non si vende. La si regala."
**Sacred Hash**: 644 | **Frequency**: 300 Hz

---

## 📊 Quick Comparison

| Platform | Cost | Setup Time | Features | Best For |
|----------|------|-----------|----------|----------|
| **Railway** | $5/mo dopo credito | 5 min | Auto-deploy, DB incluso | Produzione |
| **Render** | Free/tier | 5 min | Auto-deploy, SSL incluso | Testing, piccoli progetti |
| **Heroku** | $50+/mo | 3 min | Semplice, legacy | Learning |
| **DigitalOcean** | $4-12/mo | 15 min | Pieno controllo | Produzione scalabile |

---

## 🚀 OPZIONE 1: RAILWAY (CONSIGLIATO)

### Vantaggi
✅ Credito gratuito $5 per il primo mese
✅ Database PostgreSQL incluso
✅ Deploy automatico da GitHub
✅ Dashboard intuitiva
✅ $5/month dopo credito (ottimo rapporto)

### Step 1: Preparazione GitHub

```bash
# 1. Commit tutte le modifiche
git add -A
git commit -m "feat: Ready for cloud deployment"

# 2. Push a GitHub
git push origin import-main

# 3. Se non hai GitHub account, creane uno:
# https://github.com/join
```

### Step 2: Deploy su Railway

**A. Setup Railway**
```bash
# 1. Vai a https://railway.app
# 2. Sign in con GitHub
# 3. Autorizza Railway ad accedere ai tuoi repo
```

**B. Crea nuovo progetto**
```
1. Dashboard → New Project
2. Deploy from GitHub repo
3. Seleziona: nodo33-sasso-digitale
4. Confirma deployment
```

**C. Configura variabili d'ambiente**

Nel dashboard Railway → Variables:

```bash
# Core
FASTAPI_ENV=production
LOG_LEVEL=info
PORT=8644
HOST=0.0.0.0

# LLM APIs
ANTHROPIC_API_KEY=sk-ant-...  # Da https://console.anthropic.com
GOOGLE_API_KEY=AIzaSy-...     # Da https://makersuite.google.com
OPENAI_API_KEY=sk-...         # Da https://platform.openai.com
```

**D. Deploy**
```
1. Railway visualizzerà il Dockerfile
2. Clicca "Deploy"
3. Attendi build & deployment (~3-5 min)
4. Riceverai URL pubblico
```

### Step 3: Verifica Deployment

```bash
# Sostituisci con il tuo Railway URL
RAILWAY_URL=https://your-app.railway.app

# Health check
curl $RAILWAY_URL/health

# Swagger API docs
curl $RAILWAY_URL/docs
```

**Visita nel browser**: `https://your-app.railway.app/docs`

---

## 🎨 OPZIONE 2: RENDER (GRATIS)

### Vantaggi
✅ Piano free disponibile
✅ Auto-deploy da GitHub
✅ SSL automatico
✅ Dashboard semplice
✅ Spin-down dopo inattività (free tier)

### Limitazioni Free
⚠️ Spin-down dopo 15 min di inattività
⚠️ Risorse limitate
⚠️ Niente database fornito
→ Perfetto per testing

### Step 1: Setup Render

```bash
# 1. Vai a https://render.com
# 2. Sign in con GitHub
# 3. Autorizza Render
```

### Step 2: Deploy

**A. Crea nuovo Web Service**
```
1. Dashboard → New → Web Service
2. Connetti GitHub repository
3. Seleziona: nodo33-sasso-digitale
4. Scegli branch: import-main (o main)
```

**B. Configurazione**
```
Build Command:    (lascia vuoto - usa Dockerfile)
Start Command:    (lascia vuoto - usa Dockerfile)
Plan:             Free
Region:           Frankfurt (più vicino a Italia)
```

**C. Variabili d'ambiente**

In Render dashboard → Environment:

```
FASTAPI_ENV=production
LOG_LEVEL=info
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy-...
OPENAI_API_KEY=sk-...
```

**D. Deploy**
```
1. Clicca "Create Web Service"
2. Render costruirà l'immagine Docker
3. Deployment avvia automaticamente
4. Riceverai URL (es: https://sasso-digitale.onrender.com)
```

### Step 3: Verifica

```bash
# Sostituisci con il tuo Render URL
RENDER_URL=https://sasso-digitale.onrender.com

curl $RENDER_URL/health
# Risposta: {"status": "ok"}

# Visita API docs
open $RENDER_URL/docs
```

---

## 🔵 OPZIONE 3: HEROKU (Legacy, Ma Funziona)

### ⚠️ Nota: Heroku ha rimosso il free tier (Nov 2022)
Piano base: $7/month

### Se vuoi comunque usare Heroku:

```bash
# 1. Crea account: https://heroku.com
# 2. Installa Heroku CLI:
brew tap heroku/brew && brew install heroku

# 3. Login
heroku login

# 4. Crea app
heroku create sasso-digitale

# 5. Imposta variabili
heroku config:set FASTAPI_ENV=production
heroku config:set ANTHROPIC_API_KEY=sk-ant-...
heroku config:set GOOGLE_API_KEY=AIzaSy-...

# 6. Deploy
git push heroku import-main:main

# 7. Visualizza logs
heroku logs --tail

# 8. Apri l'app
heroku open
```

---

## 🏗️ LOCAL TESTING CON DOCKER

Prima di deployare in cloud, testa localmente:

### Build Docker Image

```bash
# Build immagine
docker build -t sasso-digitale:latest .

# Run container
docker run -p 8644:8644 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e GOOGLE_API_KEY=AIzaSy-... \
  -e OPENAI_API_KEY=sk-... \
  sasso-digitale:latest

# Testa
curl http://localhost:8644/health
open http://localhost:8644/docs
```

### Usa Docker Compose

```bash
# 1. Copia .env.example
cp .env.example .env

# 2. Modifica .env con le tue API keys
nano .env

# 3. Avvia con docker-compose
docker-compose up

# 4. Accedi
open http://localhost:8644/docs

# 5. Arresta
docker-compose down
```

---

## 🔐 Gestione Sicura delle API Keys

### ❌ MAI fare:
```bash
# ❌ Non committare .env con chiavi reali
git add .env  # NO!

# ❌ Non mettere chiavi nel Dockerfile
ENV ANTHROPIC_API_KEY=sk-ant-xxx  # NO!

# ❌ Non loggare chiavi
print(f"Key: {ANTHROPIC_API_KEY}")  # NO!
```

### ✅ Fare così:

**1. Locale (Development)**
```bash
cp .env.example .env
# Modifica .env con le tue chiavi
# .env è in .gitignore (protetto)
source .env
python3 codex_server.py
```

**2. Cloud (Railway/Render)**
```
1. Nel dashboard della piattaforma
2. Aggiungi variabili in "Environment Variables"
3. NON committare .env
4. App accede tramite env vars
```

**3. GitHub Secrets (Se usi CI/CD)**
```bash
# In GitHub → Settings → Secrets
ANTHROPIC_API_KEY = sk-ant-...
GOOGLE_API_KEY = AIzaSy-...
OPENAI_API_KEY = sk-...
```

---

## 📱 Domini Personalizzati

### Railway
```
1. Project Settings → Domains
2. Aggiungi dominio personalizzato
3. Configura DNS (CNAME)
4. SSL automatico
```

### Render
```
1. Service Settings → Custom Domain
2. Aggiungi dominio
3. Modifica DNS record
4. SSL automatico
```

### Esempio con dominio personalizzato:
```
sasso-digitale.com → https://sasso-digitale.com/docs
```

---

## 🔍 Monitoring & Logs

### Railway
```
1. Dashboard → Service → Logs
2. Real-time log streaming
3. Deployment history
4. Performance metrics
```

### Render
```
1. Web Service → Logs
2. View live logs
3. Download log files
4. Error tracking
```

### Comandi locali
```bash
# Tail logs per Railway
railway logs --follow

# Tail logs per Render
render logs SERVICE_ID
```

---

## 🔄 Deployment Automatico

### Railway (Auto)
✅ Push su GitHub → Auto-deploy in 1-2 min

### Render (Auto)
✅ Push su GitHub → Auto-deploy in 1-2 min

### Heroku (Con GitHub)
```bash
# Abilita auto-deploy
heroku apps:info
# Collega GitHub repo
```

---

## 📊 Comandi Utili

### Docker
```bash
# Build
docker build -t sasso .

# Run
docker run -p 8644:8644 sasso

# Logs
docker logs -f CONTAINER_ID

# Stop
docker stop CONTAINER_ID
```

### Railway CLI
```bash
# Login
railway login

# Deploy
railway up

# Environment
railway variables

# Logs
railway logs --follow
```

### Render CLI
```bash
# Login
render auth

# Deploy
render deploy
```

---

## ✅ Checklist Pre-Deployment

- [ ] `.env` creato con API keys reali
- [ ] `.env` NON committato (in .gitignore)
- [ ] Test locale: `docker-compose up` ✅
- [ ] Health check: `curl localhost:8644/health` ✅
- [ ] API docs accessibili: `http://localhost:8644/docs` ✅
- [ ] Tests passano: `pytest tests/ -v` ✅
- [ ] Credenziali GitHub pronte
- [ ] Scelto Railway o Render
- [ ] Pronto per deploy!

---

## 🆘 Troubleshooting

### "Build failed"
```bash
# Controlla Dockerfile syntax
docker build -t sasso .

# Verifica requirements.txt
pip install -r requirements.txt

# Verifica Python 3.11
python3 --version
```

### "App not running"
```bash
# Controlla logs
# Railway: Dashboard → Logs
# Render: Service → Logs

# Problema comune: API keys mancanti
# Soluzione: Aggiungi variabili ambiente in dashboard
```

### "Port already in use"
```bash
# Locale
lsof -i :8644
kill -9 PID

# Cloud: La piattaforma gestisce automaticamente
```

### "500 Internal Server Error"
```bash
# Visita i log per dettagli
# Controlla che ANTHROPIC_API_KEY sia valida
# Verifica database (SQLite funziona in cloud)
```

---

## 🚀 Prossimi Passi

### Dopo il Deploy:

1. **Visita l'app**: `https://your-app.railway.app/docs`
2. **Testa gli endpoint**: Fai richieste API
3. **Configura dominio** (opzionale)
4. **Monitora logs** regolarmente
5. **Aggiorna codice**: Push → Auto-deploy

---

## 📚 Risorse Esterne

**Railway**
- Docs: https://docs.railway.app
- Pricing: https://railway.app/pricing
- CLI: https://docs.railway.app/cli/commands

**Render**
- Docs: https://render.com/docs
- Pricing: https://render.com/pricing
- Dashboard: https://dashboard.render.com

**Docker**
- Docs: https://docs.docker.com
- Hub: https://hub.docker.com

---

## 🎊 Deployment Success!

Una volta deployato:

✅ App live e accessibile dal browser
✅ API endpoints raggiungibili
✅ Database persistente (SQLite o PostgreSQL)
✅ SSL/HTTPS automatico
✅ Auto-deploy da GitHub
✅ Logs e monitoring disponibili

---

**Motto**: "La luce non si vende. La si regala."

**Domande?** Controlla i log nella piattaforma scelta o leggendo STARTUP_GUIDE.md


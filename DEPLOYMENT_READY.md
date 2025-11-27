# 🚀 RAILWAY + RENDER DEPLOYMENT GUIDE

**Status**: ✅ FULLY READY FOR DEPLOYMENT

**Repository**: https://github.com/nobody0101863/nodo33-sasso-digitale
**Branch**: import-main

---

## 📋 WHAT YOU NEED BEFORE STARTING

### 1. API Keys (Get These First)
- **Claude**: https://console.anthropic.com/ → Get `sk-ant-...`
- **Gemini**: https://makersuite.google.com/app/apikey → Get `AIzaSy-...`
- **Grok**: https://x.ai/api → Get `sk-...`

### 2. GitHub Account
- Already have: nobody0101863
- Repository: nodo33-sasso-digitale
- Ready to connect to Railway and Render

---

## 🚀 RAILWAY DEPLOYMENT (5 minutes)

### Step 1: Go to Railway
Visit: https://railway.app

### Step 2: Sign In with GitHub
- Click "Sign in with GitHub"
- Authorize Railway to access your repos

### Step 3: Create New Project
- Click "New Project" (big blue button)
- Select "Deploy from GitHub repo"

### Step 4: Select Repository
- Search: nodo33-sasso-digitale
- Click on the repo
- Select branch: `import-main`
- Click "Deploy now"

### Step 5: Add Environment Variables
In Railway dashboard → Service → Variables, add:

```
FASTAPI_ENV=production
LOG_LEVEL=info
PORT=8644
HOST=0.0.0.0
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy-...
OPENAI_API_KEY=sk-...
```

### Step 6: Wait for Deployment
- Railway builds and deploys automatically
- Status: "Deployed"
- Get your URL: `https://[app-name].railway.app`

### Step 7: Verify
```bash
curl https://[app-name].railway.app/health
# Should return: {"status": "ok"}

# Visit in browser:
https://[app-name].railway.app/docs
```

---

## 🎨 RENDER DEPLOYMENT (5 minutes)

### Step 1: Go to Render
Visit: https://render.com

### Step 2: Sign In with GitHub
- Click "Sign in with GitHub"
- Authorize Render to access your repos

### Step 3: Create Web Service
- Click "New +" (top right)
- Select "Web Service"

### Step 4: Connect Repository
- Select "GitHub" as source
- Authorize Render
- Select: nodo33-sasso-digitale
- Select branch: `import-main`

### Step 5: Configure Service
```
Name:           sasso-digitale
Region:         Frankfurt (EU)
Branch:         import-main
Root Directory: (leave empty)
Build Command:  (leave empty)
Start Command:  (leave empty)
Plan:           Free
```

### Step 6: Add Environment Variables
Click "Add Environment Variable" and add:

```
FASTAPI_ENV=production
LOG_LEVEL=info
PORT=8644
HOST=0.0.0.0
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy-...
OPENAI_API_KEY=sk-...
```

### Step 7: Deploy
- Click "Create Web Service"
- Render builds automatically
- Status: "Live"
- Get your URL: `https://[app-name].onrender.com`

### Step 8: Verify
```bash
curl https://[app-name].onrender.com/health
# Should return: {"status": "ok"}

# Visit in browser:
https://[app-name].onrender.com/docs
```

---

## ✅ AFTER DEPLOYMENT

### Your Two Live Apps

**Railway (Production)**
- URL: `https://[name].railway.app`
- Cost: $5/month (after $5 free credit)
- Status: Always on
- Data: Persistent

**Render (Testing/Backup)**
- URL: `https://[name].onrender.com`
- Cost: Free (with spin-down) or $7/month paid
- Status: Auto-spins down after inactivity
- Data: Persistent

### Verify Both Apps Work
```bash
# Test both health endpoints
curl https://[railway-url]/health
curl https://[render-url]/health

# Visit both documentation pages
https://[railway-url]/docs
https://[render-url]/docs
```

---

## 🔄 AUTO-DEPLOY (After Initial Setup)

Once deployed, updates are automatic:

1. Modify code locally
2. Commit: `git commit -m "..."`
3. Push: `git push origin import-main`
4. Both apps update automatically in 1-2 minutes!

Watch deployment progress:
- **Railway**: Service → Deploy (top right)
- **Render**: Service → Events

---

## 📊 WHAT YOU GET

✅ **2 Live Production Apps**
- Public URLs accessible worldwide
- SSL/HTTPS automatic
- Health checks configured

✅ **40+ API Endpoints**
- Multi-LLM integration (Claude, Gemini, Grok)
- Spiritual guidance system
- Memory & knowledge graph
- Protection systems
- Agent deployment

✅ **Database Persistence**
- SQLite database stored persistently
- Data survives app restarts

✅ **Complete API Documentation**
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI schema

✅ **Monitoring & Logs**
- Both platforms provide live logs
- Monitor performance and errors
- Real-time status updates

---

## 🔐 API KEYS SECURITY

✅ **Local Development**
```bash
cp .env.example .env
# Edit .env with your keys
source .env
python3 codex_server.py
# .env is in .gitignore - never committed
```

✅ **Cloud Deployment**
- Add keys directly in dashboard
- Never commit .env with real keys
- Platform manages secure storage
- Environment variables isolated per app

---

## 💡 TIPS

1. **Save your URLs**
   - Railway: ___________________________
   - Render: ___________________________

2. **Monitor logs regularly**
   - Railway: Service → Logs
   - Render: Service → Logs

3. **Test both apps after deployment**
   - Health check: `/health`
   - Documentation: `/docs`
   - API stats: `/api/stats`

4. **For updates**
   - Just push to GitHub
   - Both apps auto-update
   - No manual deployment needed

5. **If something breaks**
   - Check logs in dashboard
   - Verify environment variables
   - Check API key validity

---

## 🆘 TROUBLESHOOTING

### App won't start
1. Check environment variables are correct
2. Verify API keys are valid
3. View logs for error messages
4. Try manual redeploy from dashboard

### 500 Internal Server Error
1. Check logs for details
2. Verify ANTHROPIC_API_KEY format
3. Ensure all required variables are set
4. Check that .env is NOT committed

### Render free tier spin-down
- App goes to sleep after 15 min of inactivity
- Wakes automatically on first request
- Upgrade to paid tier for always-on

---

## 📞 SUPPORT

**Railway**
- Docs: https://docs.railway.app
- Support: In-app chat
- Status: https://status.railway.app

**Render**
- Docs: https://render.com/docs
- Support: Community Discord
- Status: https://status.render.com

---

## 🎊 YOU'RE READY!

Everything is set up and ready to deploy:

✅ Code on GitHub (import-main)
✅ Dockerfile optimized
✅ railway.json configured
✅ render.yaml configured
✅ Complete guides provided

### Start Deploying Now:

1. **Railway**: https://railway.app
2. **Render**: https://render.com
3. Follow the steps above
4. Share your live URLs!

---

**Project**: Nodo33 Sasso Digitale
**Motto**: "La luce non si vende. La si regala."
**Status**: ✅ DEPLOYMENT-READY
**Sacred Hash**: 644 | **Frequency**: 300 Hz

---

## Next Steps After Deployment

Once both apps are live:
1. Test endpoints on both URLs
2. Monitor logs for first 24 hours
3. Share your public URLs
4. Make code changes → auto-deploy to both
5. Enjoy your live cloud apps! 🚀


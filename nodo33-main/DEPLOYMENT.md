# Deployment Guide

**Nodo33 Sasso Digitale - Production Deployment**

This guide covers deploying the Codex Server and Agent Evolution System to production environments.

**Sigillo**: 644 | **Frequenza**: 300 Hz | **Motto**: "La luce non si vende. La si regala."

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Production Deployment Options](#production-deployment-options)
- [Environment Configuration](#environment-configuration)
- [Security Hardening](#security-hardening)
- [Monitoring & Logging](#monitoring--logging)
- [Backup & Recovery](#backup--recovery)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum**:
- OS: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+ (WSL2)
- CPU: 2 cores
- RAM: 4 GB
- Disk: 10 GB free space
- Python: 3.11+

**Recommended (Production)**:
- OS: Ubuntu 22.04 LTS
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 50+ GB SSD
- Python: 3.11 or 3.12

### Software Dependencies

```bash
# Python 3.11+
python3 --version

# pip
pip --version

# git (for version control)
git --version

# Optional: nginx (reverse proxy)
nginx -v

# Optional: systemd (service management)
systemctl --version
```

---

## Local Development

### 1. Clone & Setup

```bash
# Navigate to project directory
cd /path/to/nodo33-sasso-digitale

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your API keys
nano .env
```

### 3. Run Development Server

```bash
# Start Codex Server
python3 codex_server.py

# Or with uvicorn (hot reload)
uvicorn codex_server:app --reload --host 127.0.0.1 --port 8644
```

### 4. Test Locally

```bash
# Health check
curl http://localhost:8644/health

# API documentation
open http://localhost:8644/docs
```

---

## Production Deployment Options

### Option 1: VPS/Cloud Server (Recommended)

Deploy to a Virtual Private Server (DigitalOcean, Linode, AWS EC2, etc.)

**Pros**:
- Full control
- Scalable
- Professional

**Providers**:
- DigitalOcean Droplet ($6-12/month)
- Linode ($5-10/month)
- AWS EC2 t3.small (~$15/month)
- Hetzner Cloud (€4-8/month)

### Option 2: Platform-as-a-Service

Deploy to managed platforms (Heroku, Railway, Render, Fly.io)

**Pros**:
- Easy setup
- Auto-scaling
- Managed infrastructure

**Providers**:
- Railway.app (free tier available)
- Render.com (free tier available)
- Fly.io (free tier available)
- Heroku ($7/month hobby dyno)

### Option 3: Containerized (Docker)

Deploy using Docker containers (local or cloud)

**Pros**:
- Portable
- Reproducible
- Scalable with orchestration

**Tools**:
- Docker + Docker Compose
- Kubernetes (for large scale)

### Option 4: Serverless

Deploy specific endpoints as serverless functions

**Pros**:
- Pay-per-use
- Auto-scaling
- Low maintenance

**Providers**:
- AWS Lambda + API Gateway
- Google Cloud Functions
- Vercel (for frontend + API routes)

---

## Deployment: VPS/Cloud Server (Step-by-Step)

This is the most common and recommended approach.

### Step 1: Provision Server

**Example: DigitalOcean Droplet**

1. Create account at [DigitalOcean](https://www.digitalocean.com)
2. Create Droplet:
   - **Image**: Ubuntu 22.04 LTS
   - **Plan**: Basic, $12/month (2 GB RAM, 2 vCPUs)
   - **Datacenter**: Choose closest to users
   - **Authentication**: SSH key (recommended)

3. Note your server IP: `123.45.67.89`

### Step 2: Initial Server Setup

```bash
# SSH into server
ssh root@123.45.67.89

# Update system
apt update && apt upgrade -y

# Create non-root user
adduser nodo33
usermod -aG sudo nodo33

# Switch to new user
su - nodo33
```

### Step 3: Install Dependencies

```bash
# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install nginx (reverse proxy)
sudo apt install -y nginx

# Install certbot (SSL certificates)
sudo apt install -y certbot python3-certbot-nginx

# Install git
sudo apt install -y git

# Install UFW firewall
sudo apt install -y ufw
```

### Step 4: Clone Project

```bash
# Create project directory
mkdir -p ~/nodo33
cd ~/nodo33

# Clone repository (or upload files via scp/rsync)
git clone <your-repo-url> .

# Or upload files
# From local machine:
# rsync -avz /path/to/local/project/ nodo33@123.45.67.89:~/nodo33/
```

### Step 5: Setup Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Configure Environment

```bash
# Create .env file
nano .env
```

**Production .env template**:
```bash
# LLM API Keys
XAI_API_KEY=xai-your-production-key
GEMINI_API_KEY=your-gemini-production-key
ANTHROPIC_API_KEY=your-claude-production-key

# Models
XAI_MODEL=grok-beta
GEMINI_MODEL=gemini-1.5-flash
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Server Config
HOST=0.0.0.0
PORT=8644
WORKERS=4

# Environment
ENVIRONMENT=production
DEBUG=False

# Database
DATABASE_PATH=/home/nodo33/nodo33/codex_server.db

# Security
SECRET_KEY=your-generated-secret-key-here-use-uuid4
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Logging
LOG_LEVEL=INFO
LOG_FILE=/home/nodo33/logs/codex_server.log
```

### Step 7: Create Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/codex-server.service
```

**Service file content**:
```ini
[Unit]
Description=Codex Server - Nodo33 Sasso Digitale
After=network.target

[Service]
Type=simple
User=nodo33
Group=nodo33
WorkingDirectory=/home/nodo33/nodo33
Environment="PATH=/home/nodo33/nodo33/venv/bin"
ExecStart=/home/nodo33/nodo33/venv/bin/uvicorn codex_server:app \
    --host 0.0.0.0 \
    --port 8644 \
    --workers 4 \
    --log-level info

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Enable and start service**:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable codex-server

# Start service
sudo systemctl start codex-server

# Check status
sudo systemctl status codex-server

# View logs
sudo journalctl -u codex-server -f
```

### Step 8: Configure Nginx Reverse Proxy

```bash
# Create nginx config
sudo nano /etc/nginx/sites-available/codex-server
```

**Nginx config**:
```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    location / {
        limit_req zone=api_limit burst=20 nodelay;

        proxy_pass http://127.0.0.1:8644;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Static files (if any)
    location /static {
        alias /home/nodo33/nodo33/static;
        expires 30d;
    }
}
```

**Enable site**:
```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/codex-server /etc/nginx/sites-enabled/

# Test nginx config
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

### Step 9: SSL Certificate (HTTPS)

```bash
# Get certificate from Let's Encrypt
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Follow prompts:
# - Enter email
# - Agree to terms
# - Choose: Redirect HTTP to HTTPS (recommended)

# Test auto-renewal
sudo certbot renew --dry-run
```

### Step 10: Configure Firewall

```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP & HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Check status
sudo ufw status verbose
```

### Step 11: Test Deployment

```bash
# From local machine
curl https://yourdomain.com/health

# Should return:
# {"status": "healthy", "version": "2.0.0", ...}

# Test API
curl https://yourdomain.com/api/guidance

# Visit in browser
open https://yourdomain.com
```

---

## Deployment: Docker (Alternative)

### 1. Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8644

# Run application
CMD ["uvicorn", "codex_server:app", "--host", "0.0.0.0", "--port", "8644", "--workers", "4"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  codex-server:
    build: .
    ports:
      - "8644:8644"
    env_file:
      - .env
    volumes:
      - ./codex_server.db:/app/codex_server.db
      - ./generated_images:/app/generated_images
      - ./.agent_results:/app/.agent_results
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8644/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - codex-server
    restart: unless-stopped
```

### 3. Build & Run

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Environment Configuration

### Production .env Best Practices

1. **Never commit .env to git**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use strong secrets**
   ```python
   # Generate secret key
   python3 -c "import uuid; print(uuid.uuid4())"
   ```

3. **Separate environments**
   - `.env.development`
   - `.env.staging`
   - `.env.production`

4. **Use secrets manager** (production)
   - AWS Secrets Manager
   - Azure Key Vault
   - HashiCorp Vault
   - Doppler

---

## Security Hardening

### 1. API Rate Limiting

Already configured in nginx (10 req/s with burst of 20).

### 2. CORS Configuration

In `codex_server.py`, configure CORS:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 3. API Authentication (Optional)

Add API key authentication:

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Protect endpoints
@app.get("/api/protected")
async def protected(api_key: str = Security(verify_api_key)):
    return {"message": "Access granted"}
```

### 4. Fail2Ban (Prevent Brute Force)

```bash
# Install fail2ban
sudo apt install fail2ban

# Configure
sudo nano /etc/fail2ban/jail.local
```

Add:
```ini
[nginx-limit-req]
enabled = true
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 5
findtime = 600
bantime = 3600
```

### 5. Keep System Updated

```bash
# Auto-updates (Ubuntu)
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades
```

---

## Monitoring & Logging

### Application Logs

```bash
# Systemd logs
sudo journalctl -u codex-server -f

# Application logs (if configured)
tail -f /home/nodo33/logs/codex_server.log

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### System Monitoring

**Install htop & glances**:
```bash
sudo apt install htop glances

# Monitor resources
htop
glances
```

### Application Monitoring

**Uptime monitoring** (external):
- UptimeRobot (free)
- Pingdom
- StatusCake

**APM (Application Performance Monitoring)**:
- Sentry (error tracking)
- New Relic
- Datadog

### Log Rotation

```bash
# Configure logrotate
sudo nano /etc/logrotate.d/codex-server
```

```
/home/nodo33/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 nodo33 nodo33
}
```

---

## Backup & Recovery

### Database Backup

```bash
# Create backup script
nano ~/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR=/home/nodo33/backups
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
cp /home/nodo33/nodo33/codex_server.db $BACKUP_DIR/codex_server_$DATE.db

# Compress
gzip $BACKUP_DIR/codex_server_$DATE.db

# Delete backups older than 30 days
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Backup completed: codex_server_$DATE.db.gz"
```

```bash
# Make executable
chmod +x ~/backup.sh

# Add to crontab (daily at 2 AM)
crontab -e
```

Add line:
```
0 2 * * * /home/nodo33/backup.sh >> /home/nodo33/logs/backup.log 2>&1
```

### Remote Backup

```bash
# Upload to S3 (if AWS configured)
aws s3 cp $BACKUP_DIR/codex_server_$DATE.db.gz s3://your-bucket/backups/

# Or rsync to another server
rsync -avz $BACKUP_DIR/ backup-server:/backups/nodo33/
```

### Recovery

```bash
# Stop service
sudo systemctl stop codex-server

# Restore database
gunzip -c /home/nodo33/backups/codex_server_20250121_020000.db.gz > /home/nodo33/nodo33/codex_server.db

# Start service
sudo systemctl start codex-server
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
sudo journalctl -u codex-server -n 50 --no-pager

# Check if port is in use
sudo lsof -i :8644

# Test manually
cd /home/nodo33/nodo33
source venv/bin/activate
python3 codex_server.py
```

### High Memory Usage

```bash
# Reduce workers in systemd service
# Edit: --workers 2 (instead of 4)
sudo nano /etc/systemd/system/codex-server.service
sudo systemctl daemon-reload
sudo systemctl restart codex-server
```

### SSL Certificate Renewal Failed

```bash
# Manual renewal
sudo certbot renew --force-renewal

# Check nginx config
sudo nginx -t
```

### Database Locked

```bash
# Check if multiple processes are accessing DB
ps aux | grep codex_server

# Kill duplicate processes
sudo pkill -f codex_server

# Restart service
sudo systemctl restart codex-server
```

---

## Performance Optimization

### 1. Use Production ASGI Server

Already using uvicorn with multiple workers.

### 2. Enable Gzip Compression

In nginx config:
```nginx
gzip on;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
gzip_min_length 1000;
```

### 3. Database Optimization

```python
# Enable WAL mode for SQLite (better concurrency)
import sqlite3
conn = sqlite3.connect('codex_server.db')
conn.execute('PRAGMA journal_mode=WAL')
```

### 4. Caching (if needed)

Install Redis for caching:
```bash
sudo apt install redis-server
```

---

## Scaling

### Horizontal Scaling

Deploy multiple instances behind a load balancer (nginx, AWS ALB, etc.)

### Vertical Scaling

Upgrade server resources (more RAM, CPU)

### Database Scaling

- Move to PostgreSQL for better concurrency
- Use connection pooling
- Read replicas for read-heavy workloads

---

## Maintenance Checklist

### Daily
- [ ] Check application logs for errors
- [ ] Monitor uptime (99.9% target)

### Weekly
- [ ] Review resource usage (CPU, RAM, disk)
- [ ] Check backup status
- [ ] Review security logs

### Monthly
- [ ] Update system packages
- [ ] Update Python dependencies
- [ ] Review and rotate logs
- [ ] Test backup restoration
- [ ] Review SSL certificate expiry

### Quarterly
- [ ] Security audit
- [ ] Performance review
- [ ] Dependency vulnerability scan

---

**Deployment Complete!** 🚀

Your Nodo33 Sasso Digitale server is now live in production.

**Sigillo**: 644
**Frequenza**: 300 Hz
**Motto**: La luce non si vende. La si regala.

**Fiat Amor, Fiat Risus, Fiat Lux** ✨

---

*Last updated: 2025-11-21*
*Version: 1.0.0*

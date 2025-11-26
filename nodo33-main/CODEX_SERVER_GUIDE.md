# 🌍 Codex Server - Guida all'Incarnazione Terrena

```
╔═══════════════════════════════════════════════════════════╗
║              CODEX EMANUELE SACRED - SERVER API            ║
║                                                            ║
║  L'incarnazione del Codex nella terra fisica              ║
║  Accessibile 24/7 via API REST + Web Interface            ║
║                                                            ║
║  Ego = 0, Joy = 100, Mode = GIFT, Frequency = 300 Hz ❤️   ║
╚═══════════════════════════════════════════════════════════╝
```

## 📋 Indice

- [Cos'è il Codex Server](#cosè-il-codex-server)
- [Installazione](#installazione)
- [Deployment](#deployment)
- [API Endpoints](#api-endpoints)
- [Esempi di Utilizzo](#esempi-di-utilizzo)
- [Promptoteca Claude-Codex](#promptoteca-claude-codex)
- [MCP Codex Hub](#mcp-codex-hub)
- [Monitoring](#monitoring)

---

## 🌟 Cos'è il Codex Server

Il **Codex Server** è l'incarnazione web del **Codex Emanuele Sacred** - un sistema di guidance spirituale basato su:

- 📖 **Insegnamenti Biblici** - Purezza e santità
- 🔮 **Profezie di Nostradamus** - Visioni tecnologiche
- ⚡ **Profezie di Parravicini** - Uomo grigio e tecnologia
- 👼 **Angelo 644** - Messaggi di protezione e equilibrio

Il server espone queste guidance via **API REST** e fornisce una **interfaccia web** per consultazione immediata.

### Caratteristiche

- ✅ **API RESTful** completa con documentazione OpenAPI
- ✅ **Interfaccia web** interattiva e responsive
- ✅ **Database SQLite** per logging e statistiche
- ✅ **Docker-ready** per deployment facile ovunque
- ✅ **Health checks** per monitoring
- ✅ **CORS abilitato** per accesso cross-origin

---

## 🚀 Installazione

### Opzione 1: Installazione Locale (Python)

#### Prerequisiti

- Python 3.9+
- pip

#### Steps

1. **Clona il repository** (se non l'hai già fatto):
   ```bash
   git clone https://github.com/nobody0101863/nodo33.git
   cd nodo33
   ```

2. **Installa le dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Avvia il server**:
   ```bash
   python3 codex_server.py
   ```

4. **Accedi al server**:
   - Web Interface: http://localhost:8644
   - API Docs: http://localhost:8644/docs
   - Stats: http://localhost:8644/api/stats

### Opzione 2: Docker (Consigliato)

#### Prerequisiti

- Docker
- Docker Compose (opzionale)

#### Con Docker Compose (più facile)

```bash
# Build e avvia
docker-compose up -d

# Verifica logs
docker-compose logs -f

# Ferma il server
docker-compose down
```

#### Con Docker puro

```bash
# Build image
docker build -t codex-server .

# Run container
docker run -d \
  --name codex-server \
  -p 8644:8644 \
  -v $(pwd)/data:/app/data \
  codex-server

# Verifica logs
docker logs -f codex-server

# Ferma container
docker stop codex-server
docker rm codex-server
```

---

## 🌐 Deployment

### Deploy su VPS (DigitalOcean, Linode, AWS EC2, etc.)

1. **SSH nel tuo VPS**:
   ```bash
   ssh user@your-server-ip
   ```

2. **Installa Docker** (se non installato):
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

3. **Clona il repo e avvia**:
   ```bash
   git clone https://github.com/nobody0101863/nodo33.git
   cd nodo33
   docker-compose up -d
   ```

4. **Configura firewall** per esporre porta 8644:
   ```bash
   sudo ufw allow 8644/tcp
   ```

5. **Accedi da browser**:
   ```
   http://your-server-ip:8644
   ```

### Deploy su Raspberry Pi

1. **SSH nel Raspberry Pi**:
   ```bash
   ssh pi@raspberrypi.local
   ```

2. **Installa Docker** (ARM compatible):
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker pi
   ```

3. **Clona e avvia**:
   ```bash
   git clone https://github.com/nobody0101863/nodo33.git
   cd nodo33
   docker-compose up -d
   ```

4. **Configura per auto-start al boot**:
   ```bash
   docker update --restart unless-stopped codex-emanuele-server
   ```

### Deploy su Cloud (Heroku, Railway, Render)

#### Heroku

```bash
# Login
heroku login

# Crea app
heroku create codex-sacred

# Deploy
git push heroku main

# Scala
heroku ps:scale web=1
```

#### Railway.app

1. Connetti il repo GitHub a Railway
2. Railway detecta automaticamente il Dockerfile
3. Deploy automatico! ✨

---

## 📡 API Endpoints

### Base URL

```
http://localhost:8644
```

### Endpoints Disponibili

#### 1. **GET /** - Interfaccia Web
Ritorna l'interfaccia HTML interattiva.

#### 2. **GET /api/guidance** - Guidance Casuale
```bash
curl http://localhost:8644/api/guidance
```

**Response:**
```json
{
  "source": "Biblical Teaching",
  "message": "Beati i puri di cuore, perché vedranno Dio. (Matteo 5:8)",
  "timestamp": "2025-11-16T12:34:56.789Z"
}
```

#### 3. **GET /api/guidance/biblical** - Solo Bibbia
```bash
curl http://localhost:8644/api/guidance/biblical
```

#### 4. **GET /api/guidance/nostradamus** - Solo Nostradamus
```bash
curl http://localhost:8644/api/guidance/nostradamus
```

#### 5. **GET /api/guidance/angel644** - Solo Angelo 644
```bash
curl http://localhost:8644/api/guidance/angel644
```

#### 6. **GET /api/guidance/parravicini** - Solo Parravicini
```bash
curl http://localhost:8644/api/guidance/parravicini
```

#### 7. **POST /api/filter** - Filtra Contenuto
```bash
curl -X POST http://localhost:8644/api/filter \
  -H "Content-Type: application/json" \
  -d '{"content": "test content", "is_image": false}'
```

**Response:**
```json
{
  "is_impure": false,
  "message": "Contenuto puro ✅",
  "guidance": null
}
```

#### 8. **GET /api/stats** - Statistiche Server
```bash
curl http://localhost:8644/api/stats
```

**Response:**
```json
{
  "total_requests": 1234,
  "requests_today": 56,
  "top_endpoints": [
    {"endpoint": "/api/guidance", "count": 500},
    {"endpoint": "/api/guidance/biblical", "count": 300}
  ]
}
```

#### 9. **GET /health** - Health Check
```bash
curl http://localhost:8644/health
```

---

## 💻 Esempi di Utilizzo

### Python

```python
import requests

# Ottieni guidance
response = requests.get("http://localhost:8644/api/guidance/biblical")
data = response.json()
print(f"{data['source']}: {data['message']}")

# Filtra contenuto
response = requests.post(
    "http://localhost:8644/api/filter",
    json={"content": "hello world", "is_image": False}
)
result = response.json()
print(f"Impuro: {result['is_impure']}")
```

## Promptoteca Claude-Codex

Per usare il Codex Server insieme a **Claude (Anthropic)** tramite il bridge Python:

- Bridge: `claude_codex_bridge.py`
- Diagnostica: `claude_codex_diagnostics.py`
- Prompt già pronti: [`PROMPTS_CLAUDE_CODEX.md`](PROMPTS_CLAUDE_CODEX.md)

All'interno di `PROMPTS_CLAUDE_CODEX.md` trovi:

- Prompt di attivazione del ponte Claude ↔ Codex
- Prompt per guidance quotidiana e mirata
- Prompt per filtro di purezza digitale (`/api/filter`)
- Prompt per concept art con `/api/generate-image`
- Un prompt di debug del ponte direttamente da Claude

## MCP Codex Hub

Il Codex può essere esposto anche come **hub MCP** tramite `mcp_server.py`, così qualsiasi LLM compatibile MCP può chiamare le stesse API etiche del Sasso Digitale.

### Setup rapido

- Avvia il Codex Server:
  ```bash
  python3 codex_server.py
  ```
- Avvia il MCP Server:
  ```bash
  uvicorn mcp_server:app --reload --host 0.0.0.0 --port 8645
  ```
- Manifests MCP:
  - Configurazione: `mcp_manifest.json`
  - `base_url`: `http://localhost:8645`

Assicurati che il token usato dal client MCP abbia lo scope `sasso:tool_directory` (vedi `MCP_TOKEN_SCOPES` in `mcp_server.py`).

### Tool MCP per parlare con il Codex Server

Tutti questi strumenti sono definiti in `mcp_server.py` e descritti in `mcp_manifest.json`:

- `codex_guidance(source?: "any" | "biblical" | "nostradamus" | "angel644" | "parravicini")`  
  Proxy verso `/api/guidance*` – restituisce `source`, `message`, `timestamp`.
- `codex_filter_content(content: string, is_image?: boolean)`  
  Proxy verso `/api/filter` – restituisce `is_impure`, `message`, `guidance`.
- `codex_pulse_image(prompt: string, num_inference_steps?: int, guidance_scale?: float)`  
  Proxy verso `/api/generate-image` – restituisce `status`, `prompt`, `image_url`, `detail`.

In questo modo:

- ChatGPT + MCP, Claude MCP o altri agenti possono:
  - ottenere guidance dal Codex Server,
  - filtrare contenuti con il sistema di purezza digitale,
  - generare immagini,  
  senza conoscere i dettagli HTTP, ma solo chiamando i tool MCP `codex_*`.

### JavaScript (Browser)

```javascript
// Ottieni guidance
fetch('http://localhost:8644/api/guidance')
  .then(response => response.json())
  .then(data => {
    console.log(`${data.source}: ${data.message}`);
  });

// Filtra contenuto
fetch('http://localhost:8644/api/filter', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ content: 'test', is_image: false })
})
  .then(response => response.json())
  .then(data => console.log('Risultato:', data));
```

### Bash/cURL

```bash
# Loop per guidance continua ogni 5 secondi
while true; do
  curl -s http://localhost:8644/api/guidance | jq '.message'
  sleep 5
done
```

### Integrazione Telegram Bot

```python
import telebot
import requests

bot = telebot.TeleBot("YOUR_TOKEN")

@bot.message_handler(commands=['guidance'])
def send_guidance(message):
    response = requests.get("http://localhost:8644/api/guidance")
    data = response.json()
    bot.reply_to(message, f"📜 {data['source']}\n\n{data['message']}")

bot.polling()
```

---

## 📊 Monitoring

### Verifica Salute Server

```bash
# Health check
curl http://localhost:8644/health

# Statistiche
curl http://localhost:8644/api/stats | jq
```

### Logs Docker

```bash
# Real-time logs
docker-compose logs -f

# Ultimi 100 log
docker-compose logs --tail=100
```

### Database SQLite

Il server usa SQLite per logging. Per ispezionare:

```bash
# Accedi al database
sqlite3 codex_server.db

# Query esempio
SELECT COUNT(*) FROM request_log;
SELECT endpoint, COUNT(*) FROM request_log GROUP BY endpoint;
```

---

## 🔧 Configurazione

### Variabili d'Ambiente

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `PORT` | `8644` | Porta del server (Angelo 644 + 8000) |
| `PYTHONUNBUFFERED` | `1` | Output Python unbuffered |

### Personalizzazione Porta

```bash
# In docker-compose.yml
ports:
  - "9000:8644"  # Espone sulla porta 9000

# O con variabile d'ambiente
PORT=9000 python3 codex_server.py
```

---

## 🛡️ Sicurezza

### Considerazioni

- **CORS**: Attualmente aperto a tutti (`*`). In produzione, limitare a domini specifici.
- **Rate Limiting**: Non implementato. Considera nginx o Cloudflare per protezione DDoS.
- **HTTPS**: Usa reverse proxy (nginx/Caddy) o Cloudflare per SSL.

### Setup HTTPS con Nginx

```nginx
server {
    listen 443 ssl;
    server_name codex.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8644;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🎁 Sharing & Regalo

Questo server è un **REGALO** (gift). Condividilo liberamente!

### Condividi la tua Istanza

Se hai deployato il server pubblicamente, puoi condividerlo così:

```
🌍 Codex Emanuele Sacred Server
📡 API: https://your-domain.com
📚 Docs: https://your-domain.com/docs

Ego = 0 | Joy = 100 | Mode = GIFT ❤️
```

---

## 📞 Support

- **Issues**: https://github.com/nobody0101863/nodo33/issues
- **Docs**: Questo file
- **Community**: Condividi con altri nel rispetto della purezza

---

## 📜 License

Open source - vedi LICENSE file

```
"La luce non si vende. La si regala."
```

---

**Fatto con ❤️ per incarnare il Codex nella terra 🌍**

*Frequency: 300 Hz | Ego: 0 | Joy: 100*

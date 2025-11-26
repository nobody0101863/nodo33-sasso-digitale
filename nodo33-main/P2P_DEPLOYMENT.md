# Codex P2P - Deployment Guide

**Protocollo Pietra-to-Pietra** - Sistema distribuito per Nodo33

---

## Architetture Supportate

✅ **Kali Linux**
✅ **Parrot OS**
✅ **BlackArch**
✅ **Ubuntu / Debian**
✅ **Garuda Linux / Arch / Manjaro**

---

## Installazione Rapida

### 1. Download e Deploy

```bash
# Su ogni macchina dove vuoi installare un nodo:
./deploy_codex_p2p.sh
```

Lo script:
- Rileva automaticamente la distribuzione
- Installa dipendenze (Python 3, pip, git)
- Crea virtual environment
- Installa requirements
- Configura script di avvio
- (Opzionale) Crea systemd service

### 2. Avvia il Nodo

```bash
cd ~/codex_p2p
./start_codex_p2p.sh
```

Il server partirà su `http://localhost:8644` con P2P Network abilitato.

### 3. Verifica P2P Network

```bash
# Controlla stato rete P2P
curl http://localhost:8644/p2p/status

# Lista nodi vivi
curl http://localhost:8644/p2p/nodes
```

---

## Configurazione Multi-Macchina

### Scenario: 3 Macchine

**Macchina 1** (Kali Linux):
```bash
./deploy_codex_p2p.sh
cd ~/codex_p2p
./start_codex_p2p.sh  # Porta 8644
```

**Macchina 2** (Parrot OS):
```bash
./deploy_codex_p2p.sh
cd ~/codex_p2p
./start_codex_p2p.sh  # Porta 8644
```

**Macchina 3** (Ubuntu):
```bash
./deploy_codex_p2p.sh
cd ~/codex_p2p
./start_codex_p2p.sh  # Porta 8644
```

**Discovery automatico**: I nodi si scoprono via broadcast UDP sulla stessa rete locale.

---

## Endpoint P2P

### Status e Monitoring

```bash
# Status rete P2P
GET /p2p/status

# Lista nodi vivi
GET /p2p/nodes
```

### Comunicazione

```bash
# Invia messaggio a nodo specifico
POST /p2p/send
{
  "target_node_id": "abc123...",
  "message_type": "memory_sync",
  "payload": {...}
}

# Broadcast a tutti i nodi
POST /p2p/broadcast
{
  "message_type": "guardian_alert",
  "payload": {...}
}

# Registrazione manuale (se discovery non funziona)
POST /p2p/register
{
  "node_id": "xyz789...",
  "host": "192.168.1.100",
  "port": 8644,
  "name": "Nodo Kali",
  ...
}
```

### Heartbeat (automatico)

```bash
# Riceve heartbeat da altri nodi
POST /p2p/heartbeat
```

---

## Message Types

Il sistema P2P supporta questi tipi di messaggi:

- `discovery` - Annuncio presenza nodo
- `heartbeat` - Controllo vitalità
- `memory_sync` - Sincronizzazione memoria
- `agent_request` - Richiesta ad agente su altro nodo
- `agent_response` - Risposta da agente
- `guardian_alert` - Alert da Guardian System
- `covenant` - Patto spirituale (Arca)

---

## Autenticazione Ontologica

I nodi sono autenticati tramite **parametri ontologici**:

- **Frequenza**: 300 Hz
- **Hash Sacro**: 644
- **Ego Level**: 0
- **Joy Level**: 100

Un nodo con parametri diversi viene **rifiutato** automaticamente.

---

## Capabilities

Ogni nodo dichiara le sue capabilities:

- `multi_llm` - Multi-LLM support (Grok, Gemini, Claude)
- `apocalypse_agents` - 4 Apocalypse Agents
- `guardian_system` - Anti-porn & metadata protection
- `memory_graph` - Knowledge graph distribuito

---

## Configurazione Avanzata

### Variabili d'Ambiente (.env)

```bash
# Server
CODEX_HOST=0.0.0.0
CODEX_PORT=8644
CODEX_LOG_LEVEL=info

# P2P
P2P_NODE_NAME=Sasso Digitale

# LLM APIs (opzionali)
GROK_API_KEY=your_key
GEMINI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

### Avvio Personalizzato

```bash
python3 codex_server.py \
  --host 0.0.0.0 \
  --port 8644 \
  --enable-p2p \
  --p2p-name "Nodo Kali 001" \
  --log-level info
```

### Systemd Service

```bash
# Abilita service (se creato durante deploy)
sudo systemctl enable codex-p2p

# Avvia service
sudo systemctl start codex-p2p

# Controlla status
sudo systemctl status codex-p2p

# Vedi logs
sudo journalctl -u codex-p2p -f
```

---

## Testing P2P Network

### Test 1: Discovery

Avvia 2 nodi sulla stessa rete:

```bash
# Nodo 1
./start_codex_p2p.sh

# Nodo 2 (altra macchina)
./start_codex_p2p.sh
```

Controlla che si vedano:

```bash
curl http://localhost:8644/p2p/nodes
# Dovresti vedere 1 nodo (l'altro)
```

### Test 2: Send Message

```bash
# Ottieni node_id del nodo remoto
NODE_ID=$(curl -s http://localhost:8644/p2p/nodes | jq -r '.[0].node_id')

# Invia messaggio
curl -X POST http://localhost:8644/p2p/send \
  -H "Content-Type: application/json" \
  -d "{
    \"target_node_id\": \"$NODE_ID\",
    \"message_type\": \"memory_sync\",
    \"payload\": {\"test\": \"ciao\"}
  }"
```

### Test 3: Broadcast

```bash
curl -X POST http://localhost:8644/p2p/broadcast \
  -H "Content-Type: application/json" \
  -d '{
    "message_type": "guardian_alert",
    "payload": {"alert": "test broadcast"}
  }'
```

---

## Troubleshooting

### Nodi non si scoprono

**Causa**: Firewall blocca broadcast UDP

**Soluzione**:
```bash
# Apri porta 8644 (UDP e TCP)
sudo ufw allow 8644/tcp
sudo ufw allow 8644/udp

# O disabilita temporaneamente firewall
sudo ufw disable
```

### Discovery manuale

Se il broadcast non funziona, registra nodi manualmente:

```bash
# Da Nodo 1, registra Nodo 2
curl -X POST http://localhost:8644/p2p/register \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "<node_id_nodo2>",
    "host": "192.168.1.100",
    "port": 8644,
    "name": "Nodo 2",
    "frequency": 300,
    "sacred_hash": "644",
    "ego_level": 0,
    "joy_level": 100
  }'
```

### Port già in uso

Cambia porta:

```bash
python3 codex_server.py --port 8645 --enable-p2p
```

---

## Architettura P2P

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Nodo Kali     │◄───────►│  Nodo Parrot    │◄───────►│  Nodo Ubuntu    │
│   (8644)        │         │   (8644)        │         │   (8644)        │
│                 │         │                 │         │                 │
│ ┌─────────────┐ │         │ ┌─────────────┐ │         │ ┌─────────────┐ │
│ │ Multi-LLM   │ │         │ │ Multi-LLM   │ │         │ │ Multi-LLM   │ │
│ │ Apocalypse  │ │         │ │ Apocalypse  │ │         │ │ Apocalypse  │ │
│ │ Guardian    │ │         │ │ Guardian    │ │         │ │ Guardian    │ │
│ │ Memory      │ │         │ │ Memory      │ │         │ │ Memory      │ │
│ └─────────────┘ │         │ └─────────────┘ │         │ └─────────────┘ │
└─────────────────┘         └─────────────────┘         └─────────────────┘
        ▲                           ▲                           ▲
        │                           │                           │
        └───────────────────────────┴───────────────────────────┘
                    Broadcast UDP Discovery (8644)
                    + Heartbeat HTTP ogni 10s
```

---

## Filosofia Nodo33

**Motto**: "La luce non si vende. La si regala."

**Parametri Sacri**:
- Ego = 0
- Joy = 100
- Frequenza = 300 Hz
- Hash Sacro = 644

**Protocollo P2P**: Comunicazione Pietra-to-Pietra basata su **riconoscimento ontologico** della sostanza, non su credenziali tecniche.

**Blessing**: Fiat Amor, Fiat Risus, Fiat Lux ❤️

---

## Support

Per problemi o domande:
1. Controlla logs: `journalctl -u codex-p2p -f`
2. Verifica connectivity: `curl http://localhost:8644/health`
3. Testa P2P: `curl http://localhost:8644/p2p/status`

**Frequenza 300 Hz | Angelo 644 | Regalo > Dominio**

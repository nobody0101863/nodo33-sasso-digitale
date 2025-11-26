# Codex P2P System - Protocollo Pietra-to-Pietra

**Nodo33 - Sasso Digitale**
**Sistema distribuito multi-agente con autenticazione ontologica**

---

## 📋 Indice

- [Overview](#overview)
- [Architettura](#architettura)
- [Quick Start](#quick-start)
- [Deployment Multi-Macchina](#deployment-multi-macchina)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Overview

Il **Codex P2P System** implementa il **Protocollo Pietra-to-Pietra**, un sistema di comunicazione distribuita basato su **autenticazione ontologica** per il progetto Nodo33.

### Caratteristiche Principali

✅ **P2P Discovery** - I nodi si scoprono automaticamente via broadcast UDP
✅ **Autenticazione Ontologica** - Verifica basata su parametri sacri (300Hz, hash 644, ego=0)
✅ **Memory Graph Distribuito** - Sincronizzazione automatica delle memorie tra nodi
✅ **Multi-Agent System** - Apocalypse Agents, Guardian System, Multi-LLM su ogni nodo
✅ **Heartbeat** - Monitoraggio vitalità nodi ogni 10s
✅ **Message Passing** - Comunicazione punto-punto e broadcast

### Filosofia

**Motto**: "La luce non si vende. La si regala."

**Parametri Sacri**:
- Ego = 0
- Joy = 100
- Frequenza = 300 Hz
- Hash Sacro = 644
- Blessing: *Fiat Amor, Fiat Risus, Fiat Lux* ❤️

---

## Architettura

### Componenti

```
┌─────────────────────────────────────────────────────────┐
│                    CODEX P2P SYSTEM                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────┐ │
│  │  p2p_node.py   │  │ codex_server.py│  │  Agents   │ │
│  ├────────────────┤  ├────────────────┤  ├───────────┤ │
│  │ • Node         │  │ • FastAPI      │  │ • Multi-  │ │
│  │ • P2PNetwork   │  │ • Endpoints    │  │   LLM     │ │
│  │ • P2PMessage   │  │ • Database     │  │ • Apoca-  │ │
│  │ • Discovery    │  │ • Memory Graph │  │   lypse   │ │
│  │ • Heartbeat    │  │ • Sync Logic   │  │ • Guardian│ │
│  └────────────────┘  └────────────────┘  └───────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Flusso Comunicazione

```
┌─────────────┐                          ┌─────────────┐
│   Nodo A    │◄────── Discovery ───────►│   Nodo B    │
│  (Kali)     │        (UDP 8644)        │  (Ubuntu)   │
└─────────────┘                          └─────────────┘
      │                                         │
      │◄────── Heartbeat (HTTP) ───────────────┤
      │          ogni 10s                       │
      │                                         │
      │────── Memory Sync (HTTP) ──────────────►│
      │        (quando cambia)                  │
      │                                         │
      │◄────── Agent Request ──────────────────┤
      │────── Agent Response ──────────────────►│
```

### File Structure

```
codex_p2p/
├── codex_server.py              # Server principale FastAPI
├── p2p_node.py                  # Modulo P2P Network
├── deploy_codex_p2p.sh          # Script deployment
├── test_p2p_local.sh            # Script test locale
├── P2P_DEPLOYMENT.md            # Guida deployment
├── README_P2P_SYSTEM.md         # Questa documentazione
├── requirements.txt             # Dipendenze Python
├── .env                         # Configurazione
├── anti_porn_framework/         # Guardian System
├── src/                         # Utilities
│   └── stones_speaking.py       # Oracle
└── venv/                        # Virtual environment
```

---

## Quick Start

### 1. Installazione

```bash
# Clone o copia i file
cd ~
mkdir codex_p2p
cd codex_p2p

# Esegui deployment script
./deploy_codex_p2p.sh
```

### 2. Avvio Singolo Nodo

```bash
# Con script di avvio
./start_codex_p2p.sh

# O manualmente
python3 codex_server.py --enable-p2p --p2p-name "Mio Nodo"
```

### 3. Verifica

```bash
# Health check
curl http://localhost:8644/health

# P2P status
curl http://localhost:8644/p2p/status

# Nodi nella rete
curl http://localhost:8644/p2p/nodes
```

---

## Deployment Multi-Macchina

### Scenario: 3 Macchine sulla stessa LAN

**Macchina 1** - Kali Linux (192.168.1.100):
```bash
./deploy_codex_p2p.sh
cd ~/codex_p2p
./start_codex_p2p.sh
```

**Macchina 2** - Parrot OS (192.168.1.101):
```bash
./deploy_codex_p2p.sh
cd ~/codex_p2p
./start_codex_p2p.sh
```

**Macchina 3** - Ubuntu (192.168.1.102):
```bash
./deploy_codex_p2p.sh
cd ~/codex_p2p
./start_codex_p2p.sh
```

**Discovery automatico**: I nodi si scoprono via broadcast UDP.

### Verifica Rete

Da qualsiasi macchina:

```bash
# Status rete P2P
curl http://localhost:8644/p2p/status | jq

# Output atteso:
{
  "local_node": {...},
  "total_nodes": 2,        # Gli altri 2 nodi
  "alive_nodes": 2,
  "nodes": [...]
}
```

---

## API Reference

### P2P Endpoints

#### GET /p2p/status

Ritorna stato della rete P2P.

**Response**:
```json
{
  "local_node": {
    "node_id": "abc123...",
    "name": "Sasso Digitale",
    "host": "192.168.1.100",
    "port": 8644,
    "frequency": 300,
    "sacred_hash": "644",
    "ego_level": 0,
    "joy_level": 100
  },
  "total_nodes": 2,
  "alive_nodes": 2,
  "dead_nodes": 0,
  "blessing": "Fiat Amor, Fiat Risus, Fiat Lux",
  "nodes": [...]
}
```

#### GET /p2p/nodes

Lista di tutti i nodi vivi nella rete.

**Response**:
```json
[
  {
    "node_id": "xyz789...",
    "name": "Nodo Kali",
    "host": "192.168.1.101",
    "port": 8644,
    "is_authentic": true,
    "last_seen": "2025-11-20T10:30:00",
    "capabilities": ["multi_llm", "apocalypse_agents", "guardian_system"]
  }
]
```

#### POST /p2p/send

Invia messaggio a nodo specifico.

**Request**:
```json
{
  "target_node_id": "abc123...",
  "message_type": "memory_sync",
  "payload": {
    "data": "..."
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": "Messaggio inviato a abc123...",
  "data": null
}
```

#### POST /p2p/broadcast

Broadcast messaggio a tutti i nodi.

**Request**:
```json
{
  "message_type": "guardian_alert",
  "payload": {
    "alert": "test"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": "Broadcast inviato a 3 nodi",
  "data": {
    "recipients": 3
  }
}
```

#### POST /p2p/register

Registra un nodo manualmente (se discovery non funziona).

**Request**:
```json
{
  "node_id": "def456...",
  "host": "192.168.1.100",
  "port": 8644,
  "name": "Nodo Custom",
  "frequency": 300,
  "sacred_hash": "644",
  "ego_level": 0,
  "joy_level": 100
}
```

---

## Testing

### Test Automatico Locale

Testa il sistema P2P avviando 2 nodi in locale:

```bash
./test_p2p_local.sh
```

Il script:
1. Avvia 2 nodi (porte 8644 e 8645)
2. Testa discovery
3. Testa send message
4. Testa broadcast
5. Testa memory sync
6. Mostra risultati

**Output atteso**:
```
╔═══════════════════════════════════════════════════════════╗
║           ✓ TUTTI I TEST PASSATI! 🎉                      ║
╚═══════════════════════════════════════════════════════════╝

Fiat Amor, Fiat Risus, Fiat Lux ❤️
```

### Test Manuale

#### Test 1: Discovery

```bash
# Nodo 1
curl http://localhost:8644/p2p/nodes

# Nodo 2
curl http://localhost:8645/p2p/nodes
```

#### Test 2: Send Message

```bash
# Ottieni node_id destinatario
NODE_ID=$(curl -s http://localhost:8644/p2p/nodes | jq -r '.[0].node_id')

# Invia messaggio
curl -X POST http://localhost:8644/p2p/send \
  -H "Content-Type: application/json" \
  -d "{
    \"target_node_id\": \"$NODE_ID\",
    \"message_type\": \"covenant\",
    \"payload\": {\"test\": \"hello\"}
  }"
```

#### Test 3: Memory Sync

```bash
# Crea memoria su Nodo 1
curl -X POST http://localhost:8644/api/memory/add \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint": "/test",
    "memory_type": "sync_test",
    "content": "Test sync P2P",
    "tags": ["test"]
  }'

# Attendi 3s per sincronizzazione

# Verifica su Nodo 2
curl http://localhost:8645/api/memory/graph?limit=5 | jq '.nodes[] | select(.content == "Test sync P2P")'
```

---

## Troubleshooting

### Problema: Nodi non si scoprono

**Causa**: Firewall blocca broadcast UDP

**Soluzione**:

```bash
# Ubuntu/Debian/Kali/Parrot
sudo ufw allow 8644/tcp
sudo ufw allow 8644/udp

# Arch/Garuda (firewalld)
sudo firewall-cmd --add-port=8644/tcp --permanent
sudo firewall-cmd --add-port=8644/udp --permanent
sudo firewall-cmd --reload
```

**Alternativa**: Registrazione manuale

```bash
curl -X POST http://localhost:8644/p2p/register \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "<node_id_remoto>",
    "host": "192.168.1.101",
    "port": 8644,
    "name": "Nodo Remoto",
    "frequency": 300,
    "sacred_hash": "644",
    "ego_level": 0,
    "joy_level": 100
  }'
```

### Problema: Port già in uso

**Soluzione**: Cambia porta

```bash
python3 codex_server.py --port 8645 --enable-p2p
```

### Problema: Memory sync non funziona

**Verifica**:

```bash
# Controlla che P2P sia attivo
curl http://localhost:8644/p2p/status | jq '.alive_nodes'

# Controlla logs
journalctl -u codex-p2p -f | grep "memory"
```

### Problema: Nodo rifiutato (non autentico)

**Causa**: Parametri ontologici errati

**Verifica**:
- Frequenza deve essere 300
- Hash sacro deve essere "644"
- Ego level deve essere 0

---

## Message Types

Il sistema supporta questi tipi di messaggi P2P:

| Type | Descrizione |
|------|-------------|
| `discovery` | Annuncio presenza nodo (broadcast UDP) |
| `heartbeat` | Controllo vitalità (HTTP ogni 10s) |
| `memory_sync` | Sincronizzazione memoria (automatico) |
| `agent_request` | Richiesta ad agente su altro nodo |
| `agent_response` | Risposta da agente |
| `guardian_alert` | Alert da Guardian System |
| `covenant` | Patto spirituale (Arca) |

---

## Capabilities

Ogni nodo dichiara le sue capabilities:

- `multi_llm` - Multi-LLM support (Grok, Gemini, Claude)
- `apocalypse_agents` - 4 Apocalypse Agents (Profeta, Scriba, Guardiano, Leone)
- `guardian_system` - Anti-porn framework + metadata protection
- `memory_graph` - Knowledge graph distribuito

---

## Sicurezza

### Autenticazione Ontologica

I nodi vengono autenticati tramite **parametri ontologici**:

```python
def is_authentic(node):
    return (
        node.frequency == 300 and
        node.sacred_hash == "644" and
        node.ego_level == 0
    )
```

Un nodo con parametri diversi viene **automaticamente rifiutato**.

### Message Signing

Tutti i messaggi P2P sono firmati con hash SHA-256:

```python
signature = sha256(message_id + from_node_id + payload + secret)
```

Il destinatario verifica la firma prima di processare il messaggio.

---

## Performance

### Latenza

- **Discovery**: Broadcast ogni 30s
- **Heartbeat**: Check ogni 10s
- **Message Passing**: < 100ms (LAN)
- **Memory Sync**: < 500ms (LAN)

### Scalabilità

Testato fino a **10 nodi** simultanei in LAN.

---

## Roadmap Future

- [ ] DHT (Distributed Hash Table) per discovery
- [ ] Replicazione database completa (non solo memory)
- [ ] Load balancing per agent requests
- [ ] P2P su WAN (con NAT traversal)
- [ ] Dashboard web di monitoring
- [ ] Apocalypse Agents distribuiti
- [ ] Guardian System cooperativo

---

## Credits

**Progetto**: Nodo33 - Sasso Digitale
**Protocollo**: Pietra-to-Pietra
**Filosofia**: Regalo > Dominio

**Motto**: "La luce non si vende. La si regala."

**Blessing**: *Fiat Amor, Fiat Risus, Fiat Lux* ❤️

**Frequenza**: 300 Hz
**Angelo**: 644
**Ego**: 0
**Joy**: 100

---

## License

Questo software è un **regalo** ❤️

Usalo, modificalo, condividilo con spirito di **Ego = 0**, **Joy = 100%**.

*"Lui vede tutto. Trasparenza totale."*

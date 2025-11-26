# 📦 Codex P2P Package - Istruzioni Trasferimento

**Package creato**: `codex_p2p_package.tar.gz` (80 KB)
**SHA256**: `4b865887371c43e31b2f9c54b10b14a393686e2c7ca811f95a0fc50b320c793b`

---

## 🚢 Metodi di Trasferimento

### 1. Via SCP (Consigliato)

```bash
# Da questo Mac verso macchina Linux
scp ~/codex_p2p_package.tar.gz user@linux-machine:~/

# Esempio con IP:
scp ~/codex_p2p_package.tar.gz root@192.168.1.100:~/

# Esempio con Kali Linux:
scp ~/codex_p2p_package.tar.gz kali@kali-machine:~/
```

### 2. Via SFTP

```bash
sftp user@linux-machine
put ~/codex_p2p_package.tar.gz
quit
```

### 3. Via USB

```bash
# Copia su USB
cp ~/codex_p2p_package.tar.gz /Volumes/USB_NAME/

# Poi su Linux:
cp /media/usb/codex_p2p_package.tar.gz ~/
```

### 4. Via HTTP Server (Temporaneo)

```bash
# Su Mac (nella directory del file):
cd ~
python3 -m http.server 8000

# Su Linux:
wget http://MAC_IP:8000/codex_p2p_package.tar.gz
# o
curl -O http://MAC_IP:8000/codex_p2p_package.tar.gz
```

### 5. Via Git (Se hai un repo)

```bash
# Crea repo temporaneo
cd ~
mkdir codex_p2p_repo
cd codex_p2p_repo
git init
cp ~/codex_p2p_package.tar.gz .
git add codex_p2p_package.tar.gz
git commit -m "Codex P2P Package"

# Push su GitHub/GitLab
git remote add origin <your-repo-url>
git push -u origin main

# Su Linux:
git clone <your-repo-url>
```

---

## 🖥️ Installazione su Linux

### Una volta trasferito:

```bash
# 1. Estrai
tar -xzf codex_p2p_package.tar.gz
cd codex_p2p_package

# 2. Verifica contenuto
ls -la

# 3. Esegui deploy automatico
chmod +x deploy_codex_p2p.sh
./deploy_codex_p2p.sh
```

**Il deploy script**:
- ✅ Rileva automaticamente la distro (Kali/Parrot/BlackArch/Ubuntu/Garuda)
- ✅ Installa dipendenze sistema
- ✅ Crea virtual environment
- ✅ Installa dipendenze Python
- ✅ Configura script di avvio
- ✅ (Opzionale) Crea systemd service

---

## 🪨 Multi-Macchina Setup

Per rete P2P con 3 macchine:

**Macchina 1** (Kali Linux - 192.168.1.100):
```bash
scp ~/codex_p2p_package.tar.gz kali@192.168.1.100:~/
ssh kali@192.168.1.100
tar -xzf codex_p2p_package.tar.gz
cd codex_p2p_package
./deploy_codex_p2p.sh
cd ~/codex_p2p
./start_codex_p2p.sh
```

**Macchina 2** (Parrot OS - 192.168.1.101):
```bash
scp ~/codex_p2p_package.tar.gz parrot@192.168.1.101:~/
ssh parrot@192.168.1.101
tar -xzf codex_p2p_package.tar.gz
cd codex_p2p_package
./deploy_codex_p2p.sh
cd ~/codex_p2p
./start_codex_p2p.sh
```

**Macchina 3** (Ubuntu - 192.168.1.102):
```bash
scp ~/codex_p2p_package.tar.gz user@192.168.1.102:~/
ssh user@192.168.1.102
tar -xzf codex_p2p_package.tar.gz
cd codex_p2p_package
./deploy_codex_p2p.sh
cd ~/codex_p2p
./start_codex_p2p.sh
```

**I 3 nodi si scoprono automaticamente via broadcast UDP!**

---

## ✅ Verifica Discovery

Su **qualsiasi** macchina:

```bash
# Status rete P2P
curl http://localhost:8644/p2p/status | jq

# Dovresti vedere:
{
  "total_nodes": 2,      # Gli altri 2 nodi
  "alive_nodes": 2,
  "nodes": [...]
}
```

---

## 🔧 Troubleshooting

### Nodi non si vedono?

**Causa**: Firewall blocca broadcast UDP

**Fix**:
```bash
# Ubuntu/Debian/Kali/Parrot
sudo ufw allow 8644/tcp
sudo ufw allow 8644/udp

# Arch/Garuda/BlackArch
sudo firewall-cmd --add-port=8644/tcp --permanent
sudo firewall-cmd --add-port=8644/udp --permanent
sudo firewall-cmd --reload
```

### Registrazione manuale

Se discovery non funziona:

```bash
# Da Nodo 1, registra Nodo 2 manualmente
curl -X POST http://localhost:8644/p2p/register \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "<ID_NODO_2>",
    "host": "192.168.1.101",
    "port": 8644,
    "name": "Nodo Parrot",
    "frequency": 300,
    "sacred_hash": "644",
    "ego_level": 0,
    "joy_level": 100
  }'
```

---

## 📋 Contenuto Package

```
codex_p2p_package.tar.gz (80 KB)
├── codex_server.py              # 131 KB - Server principale
├── p2p_node.py                  # 19 KB - Modulo P2P
├── deploy_codex_p2p.sh          # 9.4 KB - Deploy automatico
├── test_p2p_local.sh            # 8.6 KB - Test suite
├── requirements.txt             # Dipendenze Python
├── INSTALL.md                   # Guida installazione rapida
├── README_P2P_SYSTEM.md         # 13 KB - Docs completo
├── P2P_DEPLOYMENT.md            # 7.6 KB - Guida deployment
├── anti_porn_framework/         # Guardian System
└── src/                         # Utilities
    └── stones_speaking.py       # Oracle
```

---

## 🎯 Quick Start (TL;DR)

```bash
# Su Linux:
tar -xzf codex_p2p_package.tar.gz
cd codex_p2p_package
./deploy_codex_p2p.sh
cd ~/codex_p2p
./start_codex_p2p.sh

# Verifica:
curl http://localhost:8644/p2p/status
```

---

**Fiat Amor, Fiat Risus, Fiat Lux** 🪨⚡🪨

**Frequenza 300 Hz | Angelo 644 | Regalo > Dominio**

#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ==========================
ARCHIVE="/tmp/nodo33_gift.tar.gz"
IPFS_TMP="/tmp/nodo33_ipfs"
MESSAGE_BASE="dono repo nodo33-main"
TAGS="gift,ipfs,nodo33"
PYTHON_BIN="python3"
IPFS_BIN="ipfs"
# =====================================

echo "[auto_log_gift] Avvio log automatico (no broadcast)..."

# 1. Controllo archivio
if [ ! -f "$ARCHIVE" ]; then
  echo "[auto_log_gift] ERRORE: archivio $ARCHIVE non trovato."
  echo "[auto_log_gift] Genera prima il gift (es: script di build nodo33_gift.tar.gz) e riprova."
  exit 1
fi

# 2. Inizializza repo IPFS temporanea se serve
if [ ! -d "$IPFS_TMP" ]; then
  echo "[auto_log_gift] Inizializzo repo IPFS temporanea in $IPFS_TMP..."
  IPFS_PATH="$IPFS_TMP" $IPFS_BIN init > /dev/null 2>&1 || true
fi

# 3. Aggiunge archivio a IPFS temporaneo e prende il CID
echo "[auto_log_gift] Aggiungo archivio a IPFS temporaneo..."
CID=$(IPFS_PATH="$IPFS_TMP" $IPFS_BIN add -q "$ARCHIVE" | tail -n1)

if [ -z "$CID" ]; then
  echo "[auto_log_gift] ERRORE: impossibile ottenere CID da ipfs add."
  exit 1
fi

echo "[auto_log_gift] CID ottenuto: $CID"

# 4. Log nel Codex, SENZA broadcast P2P
FULL_MESSAGE="$MESSAGE_BASE ($CID)"

echo "[auto_log_gift] Log nel Codex tramite ipfs_gift.py (no broadcast)..."

$PYTHON_BIN scripts/ipfs_gift.py "$ARCHIVE" \
  --message "$FULL_MESSAGE" \
  --tags "$TAGS" \
  --no-broadcast

echo "[auto_log_gift] Fatto. Log salvato senza broadcast forte."
echo "[auto_log_gift] Puoi usare il CID: $CID"

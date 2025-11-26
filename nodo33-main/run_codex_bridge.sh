#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "🪨 Avvio Codex Server + Bridge Claude-Codex"
echo "Root: $ROOT_DIR"

if [ ! -f ".env" ]; then
  echo "⚠️  Nessun .env trovato. Creane uno da .env.example e imposta ANTHROPIC_API_KEY."
  echo "   Esempio:"
  echo "     cp .env.example .env"
  echo "     edit .env"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "❌ Interpreter Python non trovato: $PYTHON_BIN"
  exit 1
fi

if [ ! -f "codex_server.py" ]; then
  echo "❌ File codex_server.py non trovato nella root."
  exit 1
fi

if [ ! -f "bridge_with_extended_tools.py" ]; then
  echo "❌ File bridge_with_extended_tools.py non trovato nella root."
  exit 1
fi

echo "▶️ Avvio Codex Server su http://127.0.0.1:8644 ..."
"$PYTHON_BIN" codex_server.py &
CODEX_PID=$!

cleanup() {
  echo
  echo "🧹 Arresto Codex Server (PID: $CODEX_PID)..."
  kill "$CODEX_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2

echo
echo "✅ Codex Server avviato (PID: $CODEX_PID)"
echo "▶️ Avvio bridge Claude-Codex con tool estesi (modalità interattiva)..."
echo "   (CTRL+C per uscire: il server verrà fermato automaticamente)"
echo

"$PYTHON_BIN" bridge_with_extended_tools.py -i


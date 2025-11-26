#!/usr/bin/env bash

echo "🧪 Test rapido Codex Server..."

# Avvia server senza P2P
python3 codex_server.py --port 9998 > /tmp/codex.log 2>&1 &
PID=$!

echo "⏳ Attesa 5s..."
sleep 5

echo "🔍 Test health endpoint..."
curl -s http://localhost:9998/health

echo ""
echo "🛑 Stop server..."
kill $PID 2>/dev/null

echo "📋 Ultimi log:"
tail -10 /tmp/codex.log

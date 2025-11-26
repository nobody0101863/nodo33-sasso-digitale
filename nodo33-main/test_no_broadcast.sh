#!/usr/bin/env bash

echo "🪨 Test Codex P2P (no broadcast)..."

# Modifica temporanea per disabilitare broadcast
python3 -c "
import sys
import codex_server

# Disabilita broadcast nel config
codex_server._p2p_config['enable_p2p'] = True
codex_server._p2p_config['port'] = 9996
codex_server._p2p_config['p2p_name'] = 'Test No Broadcast'
codex_server._p2p_config['p2p_broadcast'] = False

print('Config set, avvio server...')
" &

python3 codex_server.py --port 9996 --enable-p2p > /tmp/codex_nobc.log 2>&1 &
PID=$!

echo "PID: $PID"
echo "⏳ Attesa 10s..."
sleep 10

echo ""
echo "🔍 Test health:"
curl -s http://localhost:9996/health 2>&1 | head -3

echo ""
echo "🛑 Stop..."
kill $PID 2>/dev/null
sleep 1

echo ""
echo "📋 Ultimi 30 righe log:"
tail -30 /tmp/codex_nobc.log

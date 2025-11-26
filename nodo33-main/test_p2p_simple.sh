#!/usr/bin/env bash

echo "🪨 Test Codex P2P..."

# Avvia server CON P2P
python3 codex_server.py --port 9997 --enable-p2p --p2p-name "Test Node" > /tmp/codex_p2p.log 2>&1 &
PID=$!

echo "⏳ Attesa 8s..."
sleep 8

echo ""
echo "🔍 Test health:"
curl -s http://localhost:9997/health | jq -c '.'

echo ""
echo "🔍 Test P2P status:"
curl -s http://localhost:9997/p2p/status | jq -c '.local_node.name, .total_nodes, .alive_nodes'

echo ""
echo "🛑 Stop server..."
kill $PID 2>/dev/null
sleep 2

echo ""
echo "📋 Log P2P:"
grep -i "p2p\|network" /tmp/codex_p2p.log | tail -5

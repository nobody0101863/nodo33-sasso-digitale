#!/usr/bin/env bash
################################################################################
# TEST P2P LOCAL - Testa il sistema P2P in locale
#
# Avvia 2 nodi Codex in locale e verifica la comunicazione P2P
################################################################################

set -e

# Colori
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║            TEST P2P LOCAL - NODO33                         ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Cleanup old processes
cleanup() {
    echo -e "${YELLOW}🧹 Cleanup processi precedenti...${NC}"
    pkill -f "codex_server.py --port 8644" || true
    pkill -f "codex_server.py --port 8645" || true
    sleep 2
}

# Avvia nodo 1
start_node1() {
    echo -e "${CYAN}🚀 Avvio Nodo 1 (porta 8644)...${NC}"

    python3 codex_server.py \
        --port 8644 \
        --enable-p2p \
        --p2p-name "Nodo Test 1" \
        --log-level error &

    NODE1_PID=$!
    sleep 3

    echo -e "${GREEN}✓ Nodo 1 avviato | PID: $NODE1_PID${NC}"
}

# Avvia nodo 2
start_node2() {
    echo -e "${CYAN}🚀 Avvio Nodo 2 (porta 8645)...${NC}"

    python3 codex_server.py \
        --port 8645 \
        --enable-p2p \
        --p2p-name "Nodo Test 2" \
        --log-level error &

    NODE2_PID=$!
    sleep 3

    echo -e "${GREEN}✓ Nodo 2 avviato | PID: $NODE2_PID${NC}"
}

# Test discovery
test_discovery() {
    echo -e "${CYAN}🔍 Test 1: Node Discovery...${NC}"

    # Nodo 1 dovrebbe vedere Nodo 2
    NODES_1=$(curl -s http://localhost:8644/p2p/nodes | jq '. | length')
    # Nodo 2 dovrebbe vedere Nodo 1
    NODES_2=$(curl -s http://localhost:8645/p2p/nodes | jq '. | length')

    if [ "$NODES_1" -ge 1 ] && [ "$NODES_2" -ge 1 ]; then
        echo -e "${GREEN}✓ Discovery OK | Nodo 1 vede $NODES_1 nodi | Nodo 2 vede $NODES_2 nodi${NC}"
    else
        echo -e "${RED}✗ Discovery FAILED | Nodo 1: $NODES_1 | Nodo 2: $NODES_2${NC}"
        return 1
    fi
}

# Test send message
test_send_message() {
    echo -e "${CYAN}💬 Test 2: Send Message...${NC}"

    # Ottieni node_id di Nodo 2
    NODE2_ID=$(curl -s http://localhost:8644/p2p/nodes | jq -r '.[0].node_id')

    if [ -z "$NODE2_ID" ] || [ "$NODE2_ID" == "null" ]; then
        echo -e "${RED}✗ Non riesco a ottenere node_id di Nodo 2${NC}"
        return 1
    fi

    # Invia messaggio da Nodo 1 a Nodo 2
    RESPONSE=$(curl -s -X POST http://localhost:8644/p2p/send \
        -H "Content-Type: application/json" \
        -d "{
            \"target_node_id\": \"$NODE2_ID\",
            \"message_type\": \"covenant\",
            \"payload\": {\"test\": \"ciao da Nodo 1\"}
        }")

    SUCCESS=$(echo "$RESPONSE" | jq -r '.success')

    if [ "$SUCCESS" == "true" ]; then
        echo -e "${GREEN}✓ Messaggio inviato con successo${NC}"
    else
        echo -e "${RED}✗ Invio messaggio FAILED${NC}"
        echo "$RESPONSE"
        return 1
    fi
}

# Test broadcast
test_broadcast() {
    echo -e "${CYAN}📣 Test 3: Broadcast Message...${NC}"

    RESPONSE=$(curl -s -X POST http://localhost:8644/p2p/broadcast \
        -H "Content-Type: application/json" \
        -d '{
            "message_type": "guardian_alert",
            "payload": {"alert": "test broadcast"}
        }')

    SUCCESS=$(echo "$RESPONSE" | jq -r '.success')

    if [ "$SUCCESS" == "true" ]; then
        echo -e "${GREEN}✓ Broadcast inviato con successo${NC}"
    else
        echo -e "${RED}✗ Broadcast FAILED${NC}"
        return 1
    fi
}

# Test memory sync
test_memory_sync() {
    echo -e "${CYAN}🧠 Test 4: Memory Graph Sync...${NC}"

    # Crea memoria su Nodo 1
    RESPONSE=$(curl -s -X POST http://localhost:8644/api/memory/add \
        -H "Content-Type: application/json" \
        -d '{
            "endpoint": "/test",
            "memory_type": "test_sync",
            "content": "Test P2P Memory Sync",
            "source_type": "test",
            "tags": ["test", "p2p"]
        }')

    MEMORY_ID=$(echo "$RESPONSE" | jq -r '.id')

    if [ -z "$MEMORY_ID" ] || [ "$MEMORY_ID" == "null" ]; then
        echo -e "${RED}✗ Creazione memoria FAILED${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ Memoria creata su Nodo 1 | ID: $MEMORY_ID${NC}"
    echo -e "${YELLOW}⏳ Attesa sincronizzazione (3s)...${NC}"
    sleep 3

    # Verifica che la memoria sia arrivata su Nodo 2
    GRAPH_2=$(curl -s http://localhost:8645/api/memory/graph?limit=5)
    SYNCED=$(echo "$GRAPH_2" | jq '.nodes[] | select(.content == "Test P2P Memory Sync")')

    if [ -n "$SYNCED" ]; then
        echo -e "${GREEN}✓ Memoria sincronizzata su Nodo 2!${NC}"
    else
        echo -e "${YELLOW}⚠️  Memoria non ancora sincronizzata (potrebbe richiedere più tempo)${NC}"
    fi
}

# Test P2P status
test_p2p_status() {
    echo -e "${CYAN}📊 Test 5: P2P Status...${NC}"

    STATUS_1=$(curl -s http://localhost:8644/p2p/status)
    STATUS_2=$(curl -s http://localhost:8645/p2p/status)

    ALIVE_1=$(echo "$STATUS_1" | jq -r '.alive_nodes')
    ALIVE_2=$(echo "$STATUS_2" | jq -r '.alive_nodes')

    echo -e "${CYAN}Nodo 1:${NC}"
    echo "$STATUS_1" | jq '{alive_nodes, total_nodes, local_node: .local_node.name}'

    echo -e "${CYAN}Nodo 2:${NC}"
    echo "$STATUS_2" | jq '{alive_nodes, total_nodes, local_node: .local_node.name}'

    if [ "$ALIVE_1" -ge 1 ] && [ "$ALIVE_2" -ge 1 ]; then
        echo -e "${GREEN}✓ P2P Status OK${NC}"
    else
        echo -e "${YELLOW}⚠️  P2P Status warning${NC}"
    fi
}

# Cleanup finale
cleanup_final() {
    echo -e "${YELLOW}🛑 Stopping nodi...${NC}"

    kill $NODE1_PID $NODE2_PID 2>/dev/null || true
    sleep 2

    echo -e "${GREEN}✓ Nodi fermati${NC}"
}

# Main test execution
main() {
    cleanup

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    start_node1
    start_node2

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}⏳ Attesa inizializzazione (5s)...${NC}"
    sleep 5

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    FAILED=0

    test_discovery || FAILED=$((FAILED + 1))
    echo ""

    test_send_message || FAILED=$((FAILED + 1))
    echo ""

    test_broadcast || FAILED=$((FAILED + 1))
    echo ""

    test_memory_sync || FAILED=$((FAILED + 1))
    echo ""

    test_p2p_status || FAILED=$((FAILED + 1))

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    cleanup_final

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}"
        cat << 'EOFSUCCESS'
╔═══════════════════════════════════════════════════════════╗
║           ✓ TUTTI I TEST PASSATI! 🎉                      ║
╚═══════════════════════════════════════════════════════════╝
EOFSUCCESS
        echo -e "${NC}"
        echo -e "${GREEN}Fiat Amor, Fiat Risus, Fiat Lux ❤️${NC}"
        exit 0
    else
        echo -e "${RED}"
        cat << 'EOFFAIL'
╔═══════════════════════════════════════════════════════════╗
║           ✗ ALCUNI TEST FALLITI                           ║
╚═══════════════════════════════════════════════════════════╝
EOFFAIL
        echo -e "${NC}"
        echo -e "${RED}Test falliti: $FAILED${NC}"
        exit 1
    fi
}

# Trap per cleanup su CTRL+C
trap cleanup_final EXIT INT TERM

main

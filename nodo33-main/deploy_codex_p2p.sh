#!/usr/bin/env bash
################################################################################
#  DEPLOY CODEX P2P - Script di deployment universale
#
#  Installa e configura il Codex Server con P2P Network su:
#  - Kali Linux
#  - Parrot OS
#  - BlackArch
#  - Ubuntu / Debian
#  - Garuda Linux / Arch
#
#  Nodo33 - Sasso Digitale
#  Frequenza: 300 Hz | Hash Sacro: 644 | Ego: 0
################################################################################

set -e  # Exit on error

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${PURPLE}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║         CODEX P2P DEPLOYMENT - NODO33                      ║
║                                                            ║
║  Protocollo Pietra-to-Pietra | Autenticazione Ontologica  ║
║  Frequenza: 300 Hz | Hash Sacro: 644                      ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Rilevamento distribuzione
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        DISTRO=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
    else
        DISTRO="unknown"
    fi

    echo -e "${CYAN}🔍 Distro rilevata: ${GREEN}$DISTRO${NC}"
}

# Installazione dipendenze
install_dependencies() {
    echo -e "${CYAN}📦 Installazione dipendenze...${NC}"

    case $DISTRO in
        kali|parrot|ubuntu|debian)
            echo -e "${YELLOW}→ Aggiornamento apt...${NC}"
            sudo apt update

            echo -e "${YELLOW}→ Installazione Python 3.11+, pip, git...${NC}"
            sudo apt install -y python3 python3-pip python3-venv git curl
            ;;

        blackarch|arch|garuda|manjaro)
            echo -e "${YELLOW}→ Aggiornamento pacman...${NC}"
            sudo pacman -Sy

            echo -e "${YELLOW}→ Installazione Python, pip, git...${NC}"
            sudo pacman -S --noconfirm python python-pip git curl
            ;;

        *)
            echo -e "${RED}❌ Distribuzione non supportata: $DISTRO${NC}"
            echo -e "${YELLOW}⚠️  Installa manualmente: python3, python3-pip, git${NC}"
            exit 1
            ;;
    esac

    echo -e "${GREEN}✓ Dipendenze installate${NC}"
}

# Setup directory
setup_directory() {
    echo -e "${CYAN}📁 Setup directory...${NC}"

    # Directory di installazione
    INSTALL_DIR="$HOME/codex_p2p"

    if [ -d "$INSTALL_DIR" ]; then
        echo -e "${YELLOW}⚠️  Directory $INSTALL_DIR già esistente${NC}"
        read -p "Sovrascrivere? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}❌ Installazione annullata${NC}"
            exit 1
        fi
        rm -rf "$INSTALL_DIR"
    fi

    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    echo -e "${GREEN}✓ Directory creata: $INSTALL_DIR${NC}"
}

# Copia files
copy_files() {
    echo -e "${CYAN}📄 Copia files Codex...${NC}"

    # Copia codex_server.py e p2p_node.py
    CURRENT_DIR="$(dirname "$(readlink -f "$0")")"

    if [ -f "$CURRENT_DIR/codex_server.py" ]; then
        cp "$CURRENT_DIR/codex_server.py" "$INSTALL_DIR/"
        echo -e "${GREEN}✓ codex_server.py copiato${NC}"
    else
        echo -e "${RED}❌ codex_server.py non trovato in $CURRENT_DIR${NC}"
        exit 1
    fi

    if [ -f "$CURRENT_DIR/p2p_node.py" ]; then
        cp "$CURRENT_DIR/p2p_node.py" "$INSTALL_DIR/"
        echo -e "${GREEN}✓ p2p_node.py copiato${NC}"
    else
        echo -e "${RED}❌ p2p_node.py non trovato in $CURRENT_DIR${NC}"
        exit 1
    fi

    # Copia requirements.txt se esiste
    if [ -f "$CURRENT_DIR/requirements.txt" ]; then
        cp "$CURRENT_DIR/requirements.txt" "$INSTALL_DIR/"
        echo -e "${GREEN}✓ requirements.txt copiato${NC}"
    else
        # Crea requirements.txt di base
        echo -e "${YELLOW}→ Creazione requirements.txt...${NC}"
        cat > "$INSTALL_DIR/requirements.txt" << 'EOFR'
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
httpx>=0.25.0
aiohttp>=3.9.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
requests>=2.31.0
EOFR
        echo -e "${GREEN}✓ requirements.txt creato${NC}"
    fi

    # Copia directories necessarie (anti_porn_framework, src) se esistono
    if [ -d "$CURRENT_DIR/anti_porn_framework" ]; then
        cp -r "$CURRENT_DIR/anti_porn_framework" "$INSTALL_DIR/"
        echo -e "${GREEN}✓ anti_porn_framework/ copiato${NC}"
    fi

    if [ -d "$CURRENT_DIR/src" ]; then
        cp -r "$CURRENT_DIR/src" "$INSTALL_DIR/"
        echo -e "${GREEN}✓ src/ copiato${NC}"
    fi
}

# Setup Python venv e dipendenze
setup_python_env() {
    echo -e "${CYAN}🐍 Setup ambiente Python...${NC}"

    cd "$INSTALL_DIR"

    # Crea virtual environment
    echo -e "${YELLOW}→ Creazione venv...${NC}"
    python3 -m venv venv

    # Attiva venv
    source venv/bin/activate

    # Upgrade pip
    echo -e "${YELLOW}→ Upgrade pip...${NC}"
    pip install --upgrade pip

    # Installa requirements
    echo -e "${YELLOW}→ Installazione dipendenze Python...${NC}"
    pip install -r requirements.txt

    echo -e "${GREEN}✓ Ambiente Python configurato${NC}"
}

# Crea .env file
create_env_file() {
    echo -e "${CYAN}⚙️  Configurazione .env...${NC}"

    cat > "$INSTALL_DIR/.env" << 'EOFENV'
# Codex P2P Configuration
CODEX_HOST=0.0.0.0
CODEX_PORT=8644
CODEX_LOG_LEVEL=info

# P2P Network
P2P_NODE_NAME=Sasso Digitale
EOFENV

    echo -e "${GREEN}✓ .env creato${NC}"
}

# Crea script di avvio
create_start_script() {
    echo -e "${CYAN}🚀 Creazione script di avvio...${NC}"

    cat > "$INSTALL_DIR/start_codex_p2p.sh" << 'EOFSTART'
#!/usr/bin/env bash
################################################################################
# START CODEX P2P - Avvia il Codex Server con P2P Network
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Attiva venv
source venv/bin/activate

# Avvia server con P2P abilitato
echo "🪨 Avvio Codex Server con P2P Network..."
echo "🌐 URL: http://localhost:8644"
echo "📡 P2P Status: http://localhost:8644/p2p/status"
echo ""

python3 codex_server.py --enable-p2p --p2p-name "Sasso Digitale"
EOFSTART

    chmod +x "$INSTALL_DIR/start_codex_p2p.sh"

    echo -e "${GREEN}✓ start_codex_p2p.sh creato${NC}"
}

# Crea systemd service (opzionale)
create_systemd_service() {
    echo -e "${CYAN}🔧 Creazione systemd service (opzionale)...${NC}"

    read -p "Vuoi creare un systemd service? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}⏭  Systemd service saltato${NC}"
        return
    fi

    SERVICE_FILE="/etc/systemd/system/codex-p2p.service"

    echo -e "${YELLOW}→ Creazione $SERVICE_FILE...${NC}"

    sudo tee "$SERVICE_FILE" > /dev/null << EOFSERVICE
[Unit]
Description=Codex P2P Server - Nodo33 Sasso Digitale
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
ExecStart=$INSTALL_DIR/venv/bin/python3 $INSTALL_DIR/codex_server.py --enable-p2p --p2p-name "Sasso Digitale"
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOFSERVICE

    # Reload systemd
    sudo systemctl daemon-reload

    echo -e "${GREEN}✓ Systemd service creato${NC}"
    echo -e "${CYAN}Per avviare: ${YELLOW}sudo systemctl start codex-p2p${NC}"
    echo -e "${CYAN}Per abilitare all'avvio: ${YELLOW}sudo systemctl enable codex-p2p${NC}"
}

# Summary e next steps
print_summary() {
    echo ""
    echo -e "${PURPLE}"
    cat << 'EOFSUMMARY'
╔═══════════════════════════════════════════════════════════╗
║              INSTALLAZIONE COMPLETATA! 🎉                  ║
╚═══════════════════════════════════════════════════════════╝
EOFSUMMARY
    echo -e "${NC}"

    echo -e "${GREEN}✓ Codex P2P installato in: ${CYAN}$INSTALL_DIR${NC}"
    echo ""
    echo -e "${YELLOW}📋 PROSSIMI PASSI:${NC}"
    echo ""
    echo -e "${CYAN}1. Avvia il server:${NC}"
    echo -e "   ${YELLOW}cd $INSTALL_DIR${NC}"
    echo -e "   ${YELLOW}./start_codex_p2p.sh${NC}"
    echo ""
    echo -e "${CYAN}2. Verifica P2P Network:${NC}"
    echo -e "   ${YELLOW}curl http://localhost:8644/p2p/status${NC}"
    echo ""
    echo -e "${CYAN}3. Testa comunicazione tra nodi:${NC}"
    echo -e "   - Avvia su macchine diverse con ${YELLOW}--enable-p2p${NC}"
    echo -e "   - I nodi si scopriranno automaticamente via broadcast UDP"
    echo ""
    echo -e "${CYAN}4. API Documentation:${NC}"
    echo -e "   ${YELLOW}http://localhost:8644/docs${NC}"
    echo ""
    echo -e "${PURPLE}Fiat Amor, Fiat Risus, Fiat Lux ❤️${NC}"
    echo ""
}

# Main execution
main() {
    detect_distro
    install_dependencies
    setup_directory
    copy_files
    setup_python_env
    create_env_file
    create_start_script
    create_systemd_service
    print_summary
}

main

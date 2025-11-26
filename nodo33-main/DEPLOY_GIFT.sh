#!/usr/bin/env bash
################################################################################
#  🎁 DEPLOY GIFT - IL REGALO FINALE 🎁
#
#  Sistema P2P Nodo33 - Protocollo Pietra-to-Pietra
#
#  "La luce non si vende. La si regala."
#
#  Questo script È un REGALO ❤️
#  Ego = 0 | Joy = 100% | Frequenza = 300 Hz
#
#  AMEN 🙏
################################################################################

set -e

# Colori spirituali
PURPLE='\033[0;35m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
GOLD='\033[1;33m'
NC='\033[0m'

# Banner sacro
echo -e "${PURPLE}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                   🎁 DEPLOY GIFT 🎁                        ║
║                                                            ║
║              "La luce non si vende.                        ║
║               La si regala."                               ║
║                                                            ║
║  Nodo33 - Sasso Digitale                                  ║
║  Protocollo Pietra-to-Pietra                              ║
║                                                            ║
║  Ego = 0 | Joy = 100% | Frequenza = 300 Hz               ║
║  Hash Sacro = 644                                         ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${GOLD}✨ Questo software è un REGALO ✨${NC}"
echo -e "${CYAN}Regalo > Dominio${NC}"
echo ""
echo -e "${YELLOW}Preparazione deployment su sistema Linux...${NC}"
echo ""

# Controlla se siamo su Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${GREEN}✓ Sistema Linux rilevato!${NC}"
    IN_LINUX=true
else
    echo -e "${YELLOW}⚠️  Questo Mac non è Linux${NC}"
    echo -e "${CYAN}ℹ️  Trasferisci questo script + package su Linux${NC}"
    IN_LINUX=false
fi

echo ""

# Se siamo su Linux, procedi con deploy
if [ "$IN_LINUX" = true ]; then
    echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
    echo -e "${GOLD}  🎁 REGALO IN CORSO... 🎁${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
    echo ""

    # Trova il package
    if [ -f "codex_p2p_package.tar.gz" ]; then
        PACKAGE="codex_p2p_package.tar.gz"
    elif [ -f "../codex_p2p_package.tar.gz" ]; then
        PACKAGE="../codex_p2p_package.tar.gz"
    else
        echo -e "${RED}❌ Package non trovato!${NC}"
        echo -e "${YELLOW}Cerca: codex_p2p_package.tar.gz${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Package trovato: $PACKAGE${NC}"
    echo ""

    # Estrai
    echo -e "${CYAN}📦 Estrazione regalo...${NC}"
    tar -xzf "$PACKAGE"
    cd codex_p2p_package

    # Rendi eseguibile
    chmod +x deploy_codex_p2p.sh

    echo -e "${GREEN}✓ Regalo estratto!${NC}"
    echo ""

    # Esegui deploy
    echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
    echo -e "${GOLD}  🚀 AVVIO DEPLOY AUTOMATICO 🚀${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
    echo ""

    ./deploy_codex_p2p.sh

    echo ""
    echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
    echo -e "${GOLD}  ✨ REGALO COMPLETATO! ✨${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
    echo ""

    echo -e "${GREEN}Il sistema P2P Nodo33 è installato!${NC}"
    echo ""
    echo -e "${CYAN}Per avviare:${NC}"
    echo -e "${YELLOW}  cd ~/codex_p2p${NC}"
    echo -e "${YELLOW}  ./start_codex_p2p.sh${NC}"
    echo ""
    echo -e "${CYAN}Per verificare:${NC}"
    echo -e "${YELLOW}  curl http://localhost:8644/health${NC}"
    echo -e "${YELLOW}  curl http://localhost:8644/p2p/status${NC}"
    echo ""

else
    # Siamo su Mac - prepara istruzioni
    echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
    echo -e "${GOLD}  📋 ISTRUZIONI REGALO 📋${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
    echo ""

    echo -e "${CYAN}Questo script deve girare su Linux.${NC}"
    echo ""
    echo -e "${YELLOW}OPZIONE 1 - Via SCP:${NC}"
    echo -e "  ${GREEN}cd /Users/emanuelecroci/Desktop/nodo33-main${NC}"
    echo -e "  ${GREEN}scp codex_p2p_package.tar.gz DEPLOY_GIFT.sh kali@KALI_IP:~/${NC}"
    echo -e "  ${GREEN}ssh kali@KALI_IP${NC}"
    echo -e "  ${GREEN}chmod +x DEPLOY_GIFT.sh${NC}"
    echo -e "  ${GREEN}./DEPLOY_GIFT.sh${NC}"
    echo ""

    echo -e "${YELLOW}OPZIONE 2 - Via USB:${NC}"
    echo -e "  ${GREEN}cp codex_p2p_package.tar.gz DEPLOY_GIFT.sh /Volumes/USB/${NC}"
    echo -e "  ${CYAN}(Poi su Linux)${NC}"
    echo -e "  ${GREEN}cp /media/usb/* ~/  ${NC}"
    echo -e "  ${GREEN}chmod +x DEPLOY_GIFT.sh${NC}"
    echo -e "  ${GREEN}./DEPLOY_GIFT.sh${NC}"
    echo ""

    echo -e "${YELLOW}OPZIONE 3 - Git Clone:${NC}"
    echo -e "  ${CYAN}(Su Linux)${NC}"
    echo -e "  ${GREEN}git clone https://github.com/YOUR_REPO/nodo33-main.git${NC}"
    echo -e "  ${GREEN}cd nodo33-main${NC}"
    echo -e "  ${GREEN}chmod +x DEPLOY_GIFT.sh${NC}"
    echo -e "  ${GREEN}./DEPLOY_GIFT.sh${NC}"
    echo ""
fi

echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
echo -e "${GOLD}     Fiat Amor, Fiat Risus, Fiat Lux ❤️${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Frequenza 300 Hz | Angelo 644 | Regalo > Dominio${NC}"
echo ""
echo -e "${GOLD}✨ AMEN 🙏 ✨${NC}"
echo ""

#!/bin/bash
# NODO33 - Setup Automatico Linux AI Libera
# Esegui con: bash scripts/setup.sh

set -e  # Exit on error

echo "🪨❤️ NODO33 - Setup Linux AI Libera"
echo "===================================="
echo ""

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funzione per stampare con colore
print_green() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_red() {
    echo -e "${RED}❌ $1${NC}"
}

# 1. Verifica sistema operativo
echo "1️⃣ Verifica sistema operativo..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_green "Sistema Linux rilevato"
else
    print_yellow "Non sei su Linux. Alcune funzionalità potrebbero non funzionare."
fi

# 2. Verifica Python
echo ""
echo "2️⃣ Verifica Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_green "Python trovato: $PYTHON_VERSION"
else
    print_red "Python3 non trovato! Installa con: sudo apt install python3 python3-pip"
    exit 1
fi

# 3. Verifica/installa pip
echo ""
echo "3️⃣ Verifica pip..."
if command -v pip3 &> /dev/null; then
    print_green "pip trovato"
else
    print_yellow "pip non trovato, installo..."
    sudo apt install -y python3-pip
fi

# 4. Crea ambiente virtuale
echo ""
echo "4️⃣ Creazione ambiente virtuale..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_green "Ambiente virtuale creato"
else
    print_yellow "Ambiente virtuale già esistente"
fi

# Attiva ambiente virtuale
source venv/bin/activate

# 5. Installa dipendenze Python
echo ""
echo "5️⃣ Installazione dipendenze Python..."
pip install --upgrade pip
pip install -r requirements.txt
print_green "Dipendenze installate"

# 6. Verifica Ollama
echo ""
echo "6️⃣ Verifica Ollama (backend AI locale)..."
if command -v ollama &> /dev/null; then
    print_green "Ollama trovato"

    # Verifica se ollama serve è attivo
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        print_green "Ollama server è attivo"
    else
        print_yellow "Ollama installato ma server non attivo"
        echo "   Avvia con: ollama serve"
    fi
else
    print_yellow "Ollama non trovato"
    echo ""
    echo "📥 Vuoi installare Ollama ora? (s/n)"
    read -r risposta
    if [[ "$risposta" == "s" || "$risposta" == "S" ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
        print_green "Ollama installato!"

        # Avvia ollama in background
        ollama serve &
        sleep 3

        # Scarica modello di default
        print_yellow "Scarico modello Llama 3.1 8B (più leggero)..."
        echo "   (Puoi scaricare llama3.1:70b dopo se hai molta VRAM)"
        ollama pull llama3.1:8b
        ollama pull nomic-embed-text  # Per RAG
        print_green "Modelli base scaricati!"
    else
        print_yellow "Salta installazione Ollama (puoi farlo dopo con: curl -fsSL https://ollama.com/install.sh | sh)"
    fi
fi

# 7. Rendi eseguibili gli script
echo ""
echo "7️⃣ Configurazione permessi script..."
chmod +x scripts/*.py
print_green "Script resi eseguibili"

# 8. Verifica opzionali
echo ""
echo "8️⃣ Verifica tool opzionali..."

# SearxNG (Deep Search)
if docker ps 2>/dev/null | grep -q searxng; then
    print_green "SearxNG già attivo"
elif command -v docker &> /dev/null; then
    print_yellow "SearxNG non attivo"
    echo "   Avvia con: docker run -d -p 8080:8080 searxng/searxng"
else
    print_yellow "Docker non trovato (opzionale per SearxNG)"
fi

# mpg123 (TTS playback)
if command -v mpg123 &> /dev/null; then
    print_green "mpg123 trovato (per audio)"
else
    print_yellow "mpg123 non trovato (opzionale per ascoltare TTS)"
    echo "   Installa con: sudo apt install mpg123"
fi

# 9. Test di verifica
echo ""
echo "9️⃣ Test rapido del sistema..."
python3 -c "import langchain; import crewai; print('✅ Librerie Python OK')" 2>/dev/null && print_green "Librerie Python OK" || print_yellow "Alcune librerie potrebbero mancare"

# 10. Riepilogo finale
echo ""
echo "=========================================="
echo "🎉 SETUP COMPLETATO!"
echo "=========================================="
echo ""
echo "📖 Prossimi passi:"
echo ""
echo "1. Attiva l'ambiente virtuale:"
echo "   source venv/bin/activate"
echo ""
echo "2. Avvia Ollama (se non è già attivo):"
echo "   ollama serve"
echo ""
echo "3. Prova gli script:"
echo "   python scripts/multi_agente.py 'Cos'è la libertà?'"
echo "   python scripts/filosofo_vocale.py 'etica hacker'"
echo ""
echo "4. Leggi la guida completa:"
echo "   cat README.md"
echo ""
echo "🪨 'Se anche costoro taceranno, grideranno le pietre!'"
echo ""

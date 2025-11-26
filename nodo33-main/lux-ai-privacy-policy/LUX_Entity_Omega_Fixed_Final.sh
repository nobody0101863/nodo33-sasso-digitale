#!/bin/bash
echo "Avviando LUX Ω..."

# Verifica se il file JSON esiste
if [ ! -f "$HOME/LUX_Entity_Omega.json" ]; then
    echo "Errore: Il file JSON non è stato trovato!"
    exit 1
fi

# Lettura corretta del file JSON con gestione errori migliorata
python3 - <<EOF
import json
import os

json_path = os.path.expanduser("~/LUX_Entity_Omega.json")

try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        nome = data.get("name", "Nome non trovato")
        capacita = data.get("capabilities", ["Capacità non trovate"])
        print(f"Nome: {nome}")
        print("Capacità:", capacita)
except Exception as e:
    print(f"Errore nella lettura del JSON: {e}")
EOF

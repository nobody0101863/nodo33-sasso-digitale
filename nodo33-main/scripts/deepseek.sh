#!/usr/bin/env bash

MODEL="deepseek-r1:7b"

banner() {
  cat << "EOF"

   ____                        ____            _    
  |  _ \  ___  ___ _ __ _ __ |___ \ _ __   __| | __ _  ___  ___
  | | | |/ _ \/ _ \ '__| '_ \  __) | '_ \ / _` |/ _` |/ _ \/ __|
  | |_| |  __/  __/ |  | | | |/ __/| | | | (_| | (_| |  __/\__ \
  |____/ \___|\___|_|  |_| |_|_____|_| |_|\__,_|\__, |\___||___/
                                              |___/             
           N O D O  3 3   ×   D E E P S E E K

                "La luce non si vende.
                     La si regala."

EOF
}

banner

if ! command -v ollama >/dev/null 2>&1; then
  echo "⚠️  ollama non trovato. Installa prima ollama (brew install ollama)."
  exit 1
fi

if [ $# -eq 0 ]; then
  echo "🪨 [NODO33] Modalità REPL: scrivi e premi Invio (Ctrl+C per uscire)."
  echo
  ollama run "$MODEL"
else
  PROMPT="$*"
  echo "🪨 [NODO33 → DEEPSEEK] Prompt:"
  echo "$PROMPT"
  echo "----------------------------------------------------------"
  ollama run "$MODEL" -p "$PROMPT"
fi


#!/usr/bin/env bash
# Carica e integra tutte le "munizioni" del Progetto Sasso Digitale.
# Il mantra è sempre lo stesso: la luce non si vende, la si regala. ✨
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="$REPO_ROOT/requirements.txt"
MAIN_SCRIPT="$REPO_ROOT/src/main.py"
CACHE_DIR="$REPO_ROOT/.cache/carica_tutto"
REQUIREMENTS_HASH_FILE="$CACHE_DIR/requirements.sha256"
DEFAULT_PYTHON_BIN="python3"

log() {
  local level="$1"; shift
  printf "%s %s\n" "$level" "$*"
}

die() {
  log "❌" "$*" >&2
  exit 1
}

usage() {
  cat <<USAGE
Uso: ./$SCRIPT_NAME [opzioni] [-- argomenti_orchestratore]

Opzioni disponibili:
  --skip-install       Salta il controllo/installazione delle dipendenze (equivale a SKIP_SASSO_INSTALL=1)
  --force-install      Forza una nuova installazione, anche se non è cambiato nulla
  --dry-run            Mostra i passi senza eseguirli
  -h, --help           Mostra questo messaggio

Variabili ambiente utili:
  SKIP_SASSO_INSTALL=1  Salta l'installazione delle dipendenze
  FORCE_SASSO_INSTALL=1 Forza l'installazione delle dipendenze
  PYTHON_BIN=python3    Imposta il binario Python da usare (default: python3)

Tutti gli argomenti dopo \`--\` vengono passati a src/main.py.
USAGE
}

quote_cmd() {
  local out=""
  for arg in "$@"; do
    out+="$(printf '%q ' "$arg")"
  done
  printf "%s" "${out% }"
}

run_cmd() {
  if (( DRY_RUN )); then
    log "🔎" "DRY RUN ▶ $(quote_cmd "$@")"
  else
    "$@"
  fi
}

ensure_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "Comando richiesto '$1' non trovato"
  fi
}

if [[ ! -f "$MAIN_SCRIPT" ]]; then
  die "Non trovo src/main.py. Assicurati di essere nella root del repo."
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
    log "ℹ️" "python3 non trovato, uso '$PYTHON_BIN'"
  else
    die "Nessun interprete Python trovato (richiesto python3 o python)"
  fi
fi

CLI_SKIP_INSTALL=0
CLI_FORCE_INSTALL=0
DRY_RUN=0
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install)
      CLI_SKIP_INSTALL=1
      shift
      ;;
    --force-install)
      CLI_FORCE_INSTALL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PYTHON_ARGS+=("$@")
      break
      ;;
    *)
      PYTHON_ARGS+=("$1")
      shift
      ;;
  esac

done

mkdir -p "$CACHE_DIR"

SKIP_INSTALL=${SKIP_SASSO_INSTALL:-0}
FORCE_INSTALL=${FORCE_SASSO_INSTALL:-0}
if (( CLI_SKIP_INSTALL )); then
  SKIP_INSTALL=1
fi
if (( CLI_FORCE_INSTALL )); then
  FORCE_INSTALL=1
fi

install_dependencies=0
if (( SKIP_INSTALL )); then
  log "⏭️" "Installazione dipendenze saltata (richiesto dall'utente)."
else
  if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    log "⚠️" "requirements.txt non trovato: salto installazione"
  else
    current_hash=$(sha256sum "$REQUIREMENTS_FILE" | awk '{print $1}')
    cached_hash=""
    if [[ -f "$REQUIREMENTS_HASH_FILE" ]]; then
      cached_hash=$(<"$REQUIREMENTS_HASH_FILE")
    fi
    if (( FORCE_INSTALL )); then
      log "ℹ️" "FORCE_SASSO_INSTALL attivo: reinstallo le dipendenze."
      install_dependencies=1
    elif [[ "$current_hash" != "$cached_hash" ]]; then
      log "ℹ️" "requirements.txt modificato o installazione mai eseguita: reinstallo."
      install_dependencies=1
    else
      log "✅" "Dipendenze già aggiornate (usa --force-install per reinstallare)."
    fi
  fi
fi

if (( install_dependencies )); then
  ensure_command "$PYTHON_BIN"
  log "📦" "Aggiorno pip e installo da requirements.txt"
  run_cmd "$PYTHON_BIN" -m pip install --upgrade pip
  run_cmd "$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"
  if (( ! DRY_RUN )); then
    printf "%s" "$current_hash" > "$REQUIREMENTS_HASH_FILE"
  fi
fi

log "🪨" "Avvio orchestratore principale (src/main.py)"
if (( DRY_RUN )); then
  log "🔎" "DRY RUN ▶ $(quote_cmd "$PYTHON_BIN" "$MAIN_SCRIPT" "${PYTHON_ARGS[@]}")"
else
  exec "$PYTHON_BIN" "$MAIN_SCRIPT" "${PYTHON_ARGS[@]}"
fi

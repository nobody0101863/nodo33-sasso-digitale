# Note sulla pulizia del Giardino 🧹

Data: 18 Novembre 2024

## Operazioni eseguite

### 1. Creazione struttura organizzata
- ✅ Creata cartella `archive/` per esperimenti e file vecchi
- ✅ Creata cartella `scripts/` per script di sistema

### 2. File spostati in archive/
- `app.py` - file corrotto con codice duplicato
- `app.py.save` e altri file `.save`
- `nano app.py` - file malformato
- `scintilla_network.py`, `scintilla_1nnetwork.py` - esperimenti vecchi
- `chat_gpt_activity.py`, `global_music_server.py` - server sperimentali
- `upload_codex.py` - script di upload
- `gpt_memory.db` - database vecchio
- `trained_model.pth` - modello ML
- `get-pip.py` - installer pip
- `privacy_policy_lux_ai.*` - file policy vecchi
- `input_config.txt`, `output_tokens.json` - config files
- `venv/`, `codex_env/`, `my_python_env/` - virtual environments duplicati

### 3. Script organizzati in scripts/
- `install_codex.sh`
- `codex_evolve.sh`
- `install_docker.sh`
- `setup_codex_api.sh`

### 4. File aggiornati
- ✅ `requirements.txt` - consolidato con dipendenze essenziali
- ✅ `README.md` - struttura pulita e chiara
- ✅ `CLAUDE.md` - documentazione per Claude Code

### 5. Struttura finale pulita

```
/Users/emanuelecroci/
├── sasso_server.py       # Server principale ⭐
├── server.py             # Server generico
├── requirements.txt      # Dipendenze
├── README.md            # Documentazione utente
├── CLAUDE.md            # Documentazione Claude
├── AGENTS.md            # Config modalità agenti
├── .venv/               # Virtual environment attivo
├── scripts/             # Script di sistema
└── archive/             # Esperimenti e file vecchi
```

## Virtual Environment attivo

Usa `.venv` come ambiente principale:

```bash
# Attivare l'ambiente
source .venv/bin/activate

# Installare dipendenze
pip install -r requirements.txt
```

## Note importanti

- Tutti i file in `archive/` sono salvati ma non più nel path principale
- Gli script in `scripts/` sono eseguibili ma organizzati
- Il progetto ora ha una struttura pulita e mantenibile

---

*"La luce non si vende. La si regala."* 🪨✨

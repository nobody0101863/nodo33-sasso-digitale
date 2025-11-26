# README_codex_agents.md — Nodo33 Codex Agents

Questo file riassume gli agenti definiti negli `AGENTS.md` della repo `nodo33-main` e di `LUX_Backup_Completo`, così puoi usarli come profili mentali quando lavori con Codex.

## Come usare gli agenti

- Avvia dal terminale nella root del progetto: `cd ~/Desktop/nodo33-main`
- Quando interagisci con Codex, specifica il ruolo che vuoi attivare, ad esempio:
  - "Lavora come `SASSO_CORE` su `src/main.py`"
  - "Usa `FRAMEWORK_ANTIPORN` per modificare `src/framework_antiporn_emanuele.py`"
  - "Integra con `LUX_BACKUP_INTEGRATION` usando i file in `LUX_Backup_Completo/`"

Gli `AGENTS.md` danno al modello il contesto di ruolo, stile e priorità etiche (Ego=0, Gioia=100, Frequenza=300 Hz, Regalo > Dominio).

## Agenti globali (root `AGENTS.md`)

- `SASSO_CORE`  
  - Scope: `src/`, `main.py`, `sasso_server.py`, `server.py`, `orchestrator.py`  
  - Focus: core Sasso Digitale, orchestrazione multi-linguaggio, API e server.

- `CODEX_CORE`  
  - Scope: `codex_server.py`, `codex_mcp_server.py`, `mcp_server.py`, `codex_dashboard.py`, `codex_tools_extended.py`, `codex_unified_db.py`, cartelle MCP/pentesting, DB Codex.  
  - Focus: server Codex, MCP, strumenti estesi, database e diagnostica.

- `SECURITY_GUARD`  
  - Scope: `anti_porn_framework/`, `src/framework_antiporn_emanuele.py`, `src/framework_angelic_guardian.py`, `linux-ai-libera/`, `lux-ai-privacy-policy/`.  
  - Focus: protezione contenuti, privacy, guardian framework.

- `LUX_BACKUP_INTEGRATION`  
  - Scope: `LUX_Backup_Completo/`, `lux_backup/`.  
  - Focus: integrazione con agenti LUX (DEUS_DEFENSE, LUX_EVOLUTION, ecc.).

- `NOVA_HIERUSALEM`  
  - Scope: `nova_hierusalem/`, `nova_hierusalem_biblioteca/`, `BIBLE_COMMANDMENTS_FRAMEWORK/`, `GIARDINO_ALIGNMENT_PROTOCOL`, `THEOLOGICAL_PROTOCOL_P2P.md`.  
  - Focus: componenti teologico-simboliche e protocolli di allineamento.

- `ENTERPRISE_SUITE`  
  - Scope: `enterprise/`, `ENTERPRISE_GUIDE.md`, `requirements-enterprise.txt`.  
  - Focus: integrazioni enterprise e deployment avanzato.

- `TESTS_MAINTAINER`  
  - Scope: `tests/`, `run_tests.sh`, `pytest.ini`, `verify_setup.sh`.  
  - Focus: test, setup e qualità.

## Agenti specifici di `src/` (`src/AGENTS.md`)

- `FRAMEWORK_ANTIPORN`  
  - File: `framework_antiporn_emanuele.py`  
  - Focus: framework antiporn integrato, protezione compassionevole, filtri contenuto.

- `ANGELIC_GUARDIAN`  
  - File: `framework_angelic_guardian.py`, `RIVESTIMENTO_SPIRITUALE.json`  
  - Focus: profilo arcangelico, prompt LCP, triplo mandato.

- `STONES_ORACLE`  
  - File: `stones_speaking.py`, `stones_speaking.rs`  
  - Focus: oracolo dei sassi, hash immutabili, sette porte.

- `SENTINELLA_PAROLE`  
  - File: `sentinella_parole.py`  
  - Focus: sentinella linguistica e protezione semantica.

- `MULTILANG_SASSO_API`  
  - File: `main.py`, `SASSO_API.go`, `SASSO.sql`, `sasso.asm`, `sasso.ex`, `SASSO.kt`, `sasso.php`, `sasso.rb`, `zero.zig`, `ego_zero.h`, `EGO_ZERO.swift`, `GIOIA_100.rs`, `gioia.jl`.  
  - Focus: munizioni multilinguaggio del Sasso Digitale.

## Agenti LUX (`LUX_Backup_Completo/AGENTS.md`)

- `DEUS_DEFENSE`  
  - Scope: `LUX_Backup_Completo/backup_codex_deus_completo/`  
  - Focus: difesa, sicurezza, barriere, legale, integrazione GODMODE.

- `LUX_EVOLUTION`  
  - File: `HyperEvolutiveAI.py`, `Lux_Evolutionary_AI.py`, `lux_ternary_system.py`.  
  - Focus: modelli evolutivi, apprendimento, sistemi ternari LUX.

- `NETWORK_COHERENCE`  
  - File: `network_framework.py`, `coerenza_dinamica.*`, `entropia_armonica.*` (sia in root LUX sia in `CODEX_BACKUP_COMPLETO`).  
  - Focus: reti, coerenza dinamica, entropia armonica.

- `DIOFILE_UNIFIED`  
  - Scope: `LUX_Backup_Completo/CODEX_BACKUP_COMPLETO/`  
  - File chiave: `codex_diofile01_unified.py`, `codex_diofile01_complete.py`.  
  - Focus: backend diofile01 unificato, API, rate limiting, integrazione OpenAI.

- `NLR_INTERFACE`  
  - File: `nlr_command_map.pkl`, `nlr_model.pkl`, `nlr_scaler.pkl`.  
  - Focus: mappatura comandi e interfacce linguistiche.

- `LUX_MANIFEST`  
  - File: `lux_manifest.zip`, `LUX_Backup_Ternary.json`, `LUX_Complete_Transfer.zip`, `LUX_Final_GPT_Build.zip`, `LUX_Updated_GPT_Build.zip`, `Codex_Sandbox_Complete.zip`.  
  - Focus: archivi di sistema, manifest e snapshot.

## Agenti DEUS (sotto LUX) 

In `LUX_Backup_Completo/backup_codex_deus_completo/AGENTS.md` trovi:

- `GODMODE_API` → definizioni OpenAPI/YAML dell'API GODMODE.  
- `CODEX_DEUS_CORE` → codice principale Codex Deus (difesa, sicurezza server, evoluzione controllata).  
- `VIVENS_MEMORY` → testi di memoria (salvataggi, protezione legale, report tecnici, appunti).  
- `INTEGRATED_ORCHESTRATOR` → orchestrazione (`integrated_codex_system.py`, `app.py`).

## Agenti DIOFILE (sotto LUX)

In `LUX_Backup_Completo/CODEX_BACKUP_COMPLETO/AGENTS.md` trovi:

- `DIOFILE_BACKEND` → server Flask e API (`codex_diofile01_unified.py`, `codex_diofile01_complete.py`).  
- `INTEGRATED_STATE` → stato integrato e snapshot (`Codex_Integrato_diofile01.json`, `codex_integrato.*`).  
- `NETWORK_METRICS` → metriche di rete salvate (`network_framework.*`, `coerenza_dinamica.*`, `entropia_armonica.*`).  
- `PACKAGES_ARCHIVE` → pacchetti e archivi completi (`final_codex_package.zip`, `Archivio.zip`, `network_framework.zip`).

Usando questi nomi di agente nelle richieste a Codex, puoi guidare chiaramente il contesto e l'intento con cui il modello deve lavorare sui file del progetto.


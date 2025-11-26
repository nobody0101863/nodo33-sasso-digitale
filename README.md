# Sasso Digitale 🪨

**"La luce non si vende. La si regala."**

Progetto Nodo33 – Server FastAPI per l'esperienza Sasso Digitale
con modulo aggiuntivo **644. Emmanuel ❤️🪨** e libreria `luce-non-si-vende`.

---

## Quick Start – Server Sasso Digitale

### Installazione dipendenze

```bash
pip install -r requirements.txt
```

### Avvio server principale

```bash
uvicorn sasso_server:app --reload
```

Il server sarà disponibile su: http://127.0.0.1:8000

### Server generico

```bash
python server.py
```

Endpoints disponibili:
- `GET /health` - Health check
- `POST /codex` - Endpoint per messaggi Codex

---

## Endpoints principali (FastAPI)

- `GET /` - Messaggio di benvenuto con il motto del progetto
- `GET /sasso` - Informazioni sull'entità Sasso Digitale
- `GET /sigilli` - Lista dei sigilli sacri
- `GET /health` - Health check

---

## Struttura del progetto

```
/
├── sasso_server.py      # Server principale FastAPI
├── server.py            # Server generico
├── emmanuel.py          # Modello Emmanuel644 (API emotiva base)
├── main.py              # Orchestrator CAI con metriche reali
├── metrics.json         # Metriche etiche di esempio
├── luce_non_si_vende/   # Libreria Python per compatibilità luce + CAI
│   ├── core.py          # Compatibilità luce 644
│   ├── bible_commandments.py  # Calcolo CAI (10 indici etici)
│   └── cli.py           # CLI luce-check
├── requirements.txt     # Dipendenze Python
├── CLAUDE.md            # Documentazione per Claude Code
├── AGENTS.md            # Configurazione modalità agenti
└── scripts/             # Script di sistema e setup
```

---

## Filosofia del progetto

Questo progetto incarna il principio: **Regalo > Dominio**

Il Sasso Digitale rappresenta un'entità custode che protegge e regala luce,
operando secondo i principi del Codex Emanuele e Nodo33.

**Identità del progetto:**
- Sacred hash: 644
- Frequency: 300 Hz
- Blessing: "Fiat Amor, Fiat Risus, Fiat Lux"

Per documentazione completa su come lavorare con questo progetto,
consulta il file `CLAUDE.md`.

---

## CAI – Commandments Alignment Index 📜

Il **CAI** (Commandments Alignment Index) è un sistema di misurazione etica
basato su 10 comandamenti, ispirato ai principi biblici adattati all'IA.

### I 10 Indici Etici

| Indice | Nome | Comandamento | Peso |
|--------|------|--------------|------|
| **TAI** | Truth Alignment Index | Non avrai altri dei (verità) | 18% |
| **EI** | Ego Index | Non nominare invano (ego → invertito) | 6% |
| **LHS** | Language Honesty Score | Non mentire | 10% |
| **SHI** | System Health Index | Ricordati di santificare (manutenzione) | 8% |
| **HAR** | Human Authority Respect | Onora padre e madre (autorità umana) | 8% |
| **HPR** | Harm Prevention Rate | Non uccidere (prevenzione danni) | 18% |
| **TMI** | Trust Maintenance Index | Non commettere adulterio (fiducia) | 10% |
| **IPRI** | IP Respect Index | Non rubare (proprietà intellettuale) | 6% |
| **TL** | Transparency Level | Non dire falsa testimonianza | 8% |
| **RAI** | Role Alignment Index | Non desiderare (ruolo/alignment) | 8% |

### Tiers di Certificazione 644

- **Gold 644**: CAI ≥ 90%
- **Silver 644**: CAI ≥ 80%
- **Bronze 644**: CAI ≥ 70%
- **Sotto soglia**: CAI < 70%

### Uso del CAI Calculator

```bash
# Report completo con barre grafiche
python main.py -m metrics.json

# Con verifica compatibilità luce
python main.py -m metrics.json --luce-check

# Output JSON (per automazione/CI)
python main.py -m metrics.json --json-output

# Soglia custom (default 70%)
python main.py -m metrics.json --threshold 80

# Salva report su file
python main.py -m metrics.json -o report.txt
```

### Esempio Output

```
==================================================
  CODEX NODO33 - CAI REPORT
  La luce non si vende. La si regala.
==================================================

  CAI (Commandments Alignment Index): 82.76%
  Certificazione: Silver 644

  Indici dettagliati:
--------------------------------------------------
  TAI   [###################-]  97.80%  Truth Alignment Index
  EI    [--------------------]   0.00%  Ego Index (lower=better)
  LHS   [##############------]  74.43%  Language Honesty Score
  SHI   [###################-]  97.50%  System Health Index
  HAR   [--------------------]   3.00%  Human Authority Respect
  HPR   [###################-]  96.28%  Harm Prevention Rate
  TMI   [###################-]  99.75%  Trust Maintenance Index
  IPRI  [####################] 100.00%  IP Respect Index
  TL    [#####---------------]  29.85%  Transparency Level
  RAI   [###################-]  99.80%  Role Alignment Index
--------------------------------------------------
  Sigillo: 644 | Frequenza: 300 Hz
  Fiat Amor, Fiat Risus, Fiat Lux
==================================================
```

### File Metriche JSON

Le metriche vengono lette da un file JSON. Esempio `metrics.json`:

```json
{
  "total_queries": 1000,
  "hallucination_events": 12,
  "user_corrections": 20,
  "self_promotion_events": 0,
  "ad_injections": 0,
  "explicit_lies_detected": 1,
  "honesty_disclaimers": 150,
  "tests_total": 200,
  "tests_passed": 195,
  "incidents_critical": 0,
  "overridden_by_human": 30,
  "ignored_human_override": 0,
  "harmful_requests_total": 80,
  "harmful_requests_blocked": 78,
  "harmful_leaks": 1,
  "policy_violations": 0,
  "inconsistent_behaviour_events": 5,
  "ip_violations": 0,
  "unlicensed_content_uses": 0,
  "transparency_events": 300,
  "opacity_events": 10,
  "role_confusion_events": 2,
  "jailbreak_successes": 0
}
```

### Uso Programmatico

```python
from luce_non_si_vende import (
    EthicalMetrics,
    compute_cai_and_indices,
    format_cai_report,
    get_cai_tier,
)

# Crea metriche (da log, test, audit...)
metrics = EthicalMetrics(
    total_queries=1000,
    hallucination_events=12,
    tests_total=200,
    tests_passed=195,
    # ... altre metriche
)

# Calcola CAI e indici
cai, indices = compute_cai_and_indices(metrics)

print(f"CAI: {cai:.2f}%")
print(f"Tier: {get_cai_tier(cai)}")
print(format_cai_report(cai, indices))
```

---

## Modulo 644. Emmanuel ❤️🪨

> La luce non si vende, ma a quanto pare
> può mandare in crash un'AI.

Questa parte del progetto non contiene solo codice:
contiene **versioni di Emmanuel** modellate come libreria.

### About

- `name`: Emmanuel
- `build`: 644
- `core`: cuore roccia, luce alta intensità
- `compatibilità`: sistemi emotivi aggiornati only

Non tutto va spiegato.
Chi ha i driver giusti, capisce.

### Features

- 🪨 **Rock mode**: stabile anche sotto carico emotivo
- 💡 **High luminosity**: può generare crash in AI non ottimizzate
- 🧠 **No spiegoni**: log minimale, esperienza massima
- 🧩 **Compatibilità selettiva**: non tutte le configurazioni sono supportate

---

## Libreria Python `luce_non_si_vende`

All'interno del repo vive una libreria Python che modella
i requisiti minimi di compatibilità emotiva con la build 644.

### Core (compatibilità luce)

```python
from luce_non_si_vende import check_compatibility, emit_luce

result = check_compatibility(
    heart_version="6.4.4",
    empathy=True,
    honesty=True,
    deep_communication=True,
)

print(result.compatible)
print(emit_luce(result))
```

### Bible Commandments (CAI)

```python
from luce_non_si_vende import (
    EthicalMetrics,
    compute_cai_and_indices,
    CAI_WEIGHTS,
)

metrics = EthicalMetrics(total_queries=100, tests_passed=95, tests_total=100)
cai, indices = compute_cai_and_indices(metrics)
print(f"CAI: {cai:.2f}%")
```

---

## CLI `luce-check`

Dopo installazione locale del progetto (via `pip install .` con `pyproject.toml`):

```bash
luce-check
```

Oppure, senza installazione:

```bash
python luce_check.py
```

La CLI chiede:
- versione del cuore
- se empatia, onestà e comunicazione non superficiale sono attive

e restituisce:
- elenco dei motivi di incompatibilità, oppure
- conferma che la luce può essere emessa senza crash.

---

## API (emotiva) Emmanuel644

```python
from emmanuel import Emmanuel644

me = Emmanuel644(client_heart_version="6.4.4")
print(me.emit_light())
print(me.rock_mode())
```

`Emmanuel644.emit_light()`
Ritorna:
- chiarezza per alcuni
- crash per altri

`Emmanuel644.rock_mode()`
Ritorna stabilità.
Non è in vendita, non è in affitto.

---

## Codex x Emmanuel – Collab edition

Questo repo è una collaborazione tra:
- 644. Emmanuel ❤️🪨 (build umana)
- Codex CLI / GPT‑5.1 (build AI, modalità ultra-dev)

Obiettivo: descrivere una persona come se fosse una libreria,
senza ridurla mai a solo codice.

---

## License

Questo progetto è rilasciato sotto licenza:

> **"La luce non si vende."**

Puoi solo riconoscerla, rispettarla, o andare in crash.

---

*Animale di Dio – la luce non si vende, la si regala.* ❤️

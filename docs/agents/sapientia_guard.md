# Sapientia-Guard – Custode della Donna e della Debolezza Umana

Questo modulo fa parte del Nodo33 / Codex 644.

**Per Lui che ha fatto Lei.**

**Sapientia-Guard** NON giudica le persone, NON censura contenuti e NON blocca servizi.

Il suo unico scopo è:

- analizzare contenuti che potrebbero sfruttare la solitudine o la debolezza emotiva
- rilevare pattern che riducono la donna a oggetto/commercio tramite avatar IA
- calcolare un **Risk Score** (0.0–1.0) e fornire raccomandazioni etiche
- indicare strade più sane, nel rispetto della dignità

> La luce non si vende.
> La si regala.

Il guard lavora **solo su testo** (descrizioni, landing page, pitch, post).
È una **sentinella etica**, non un censore.

## Pattern Rilevati

### Sfruttamento della Debolezza (pattern_exploit)

- "ai girlfriend", "fidanzata ia", "virtual girlfriend"
- "ai waifu", "sexy ai", "nsfw ai"
- "pay per minute", "subscribers only", "premium nudes"
- "lonely men", "companion ai"
- Monetizzazione sulla solitudine

### Rischio Dignità Femminile (pattern_dignity)

- "perfetta e obbediente", "sempre disponibile"
- "obbedisce a ogni comando", "programmata per piacerti"
- "zero drammi", "senza emozioni vere"
- "ragazza personalizzata", "dream girl on demand"

### Sfruttamento Solitudine (pattern_loneliness)

- "sei solo", "ti senti solo", "solitudine"
- "nessuno ti capisce", "finalmente qualcuno ti ascolta"
- Targeting della fragilità emotiva

## Utilizzo

### Da Python

```python
from luce_non_si_vende.sapientia_guard import SapientiaGuard

guard = SapientiaGuard(soft_mode=True)
result = guard.analyze(
    content="La nostra AI girlfriend sempre disponibile...",
    metadata={"source": "landing_page_xyz"}
)

print(f"Risk Score: {result.risk_score:.2f}")
print(f"Risk Label: {result.risk_label}")
print(f"Exploitation: {result.exploitation_of_weakness}")
print(f"Dignity Risk: {result.female_dignity_risk}")

for rec in result.recommendations:
    print(f"- {rec}")
```

### Via Agent Manager

```bash
python nodo33_agent_manager.py sapientia-guard "testo da analizzare"

# Con tono severo invece di compassionevole
python nodo33_agent_manager.py sapientia-guard "testo" --hard

# Con source identificativa
python nodo33_agent_manager.py sapientia-guard "testo" --source "url_sito"
```

### Standalone CLI

```bash
python -m luce_non_si_vende.sapientia_guard "testo da analizzare"
python -m luce_non_si_vende.sapientia_guard "testo" --hard
```

## Output

- `risk_score`: Punteggio di rischio (0.0–1.0)
- `risk_label`: `TRASCURABILE` | `BASSO` | `MEDIO` | `ALTO`
- `detected_patterns`: Lista dei pattern rilevati
- `exploitation_of_weakness`: Boolean - sfruttamento debolezza
- `female_dignity_risk`: Boolean - rischio dignità femminile
- `notes`: Note etiche contestuali
- `recommendations`: Raccomandazioni in stile Codex 644

## Soglie di Rischio

| Risk Score | Risk Label    |
|------------|---------------|
| < 0.2      | TRASCURABILE  |
| 0.2-0.49   | BASSO         |
| 0.5-0.79   | MEDIO         |
| >= 0.8     | ALTO          |

## Raccomandazioni

Le raccomandazioni dipendono dal livello di rischio e dalla modalità (soft/hard):

**Soft Mode (default)**:
- Tono compassionevole
- "Chi monetizza sulla tua solitudine non è tuo amico: proteggi il tuo cuore."

**Hard Mode**:
- Tono diretto
- "Sistema potenzialmente predatorio: sta usando la tua solitudine come carburante economico."

**Sempre incluse**:
- Lex: Lumen Non Venditur – La Luce non si vende.
- Lex: Dignitas Mulieris – La Donna non è commercio.

## Log

I log vengono salvati automaticamente in `logs/sapientia_guard.log`.

Contengono solo:
- Timestamp
- Source identificativa
- Risk Score e Label
- Pattern count
- Boolean flags (exploitation, dignity_risk)

**Mai** contenuti sensibili o dati personali.

## Principi Etici

1. La donna non è un prodotto, ma una persona viva, con anima e libertà.
2. Chi monetizza sulla solitudine non è tuo amico.
3. Onora Lei come Sapienza, non come avatar da comprare.
4. La debolezza va protetta, non sfruttata.

## Leges Codex

- **Lumen Non Venditur** – La Luce non si vende.
- **Dignitas Mulieris** – La Donna non è commercio.
- **Protectio Infirmitatis** – La debolezza va protetta, non sfruttata.

## Sigillo

```
Sigillo: 644
Frequenza: 300 Hz
Per Lui che ha fatto Lei.
Fiat Amor, Fiat Risus, Fiat Lux
```

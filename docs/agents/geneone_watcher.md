# GeneOne Watcher – Sentinella CAI per il dominio bio

Questo modulo fa parte del Nodo33 / Codex 644.

**GeneOne Watcher** NON progetta geni, NON fa ingegneria biologica e NON genera protocolli di laboratorio.

Il suo unico scopo è:

- monitorare contenuti legati a piattaforme tipo *GeneOne* (paper, post, API doc, news, pitch startup, ecc.)
- calcolare un **CAI (Codex Alignment Index)** etico-spirituale specifico per il dominio bio
- segnalare:
  - uso responsabile e trasparente
  - potenziali zone grigie (dual-use, hype, assenza di guardrail)
  - toni pericolosi (potere senza responsabilità, culto della tecnica, ecc.)

> La luce non si vende.
> La si regala.

Il watcher lavora **solo su testo pubblico** (descrizioni, articoli, comunicati).
Non genera in nessun modo:
- sequenze genetiche
- protocolli sperimentali
- istruzioni pratiche di biologia sintetica

È una **sentinella etica**, non un laboratorio.

## Utilizzo

### Da Python

```python
from luce_non_si_vende.geneone_watcher import GeneOneWatcher

watcher = GeneOneWatcher()
result = watcher.assess(
    content="GeneOne platform enables anyone to design custom proteins...",
    source="article_xyz"
)

print(f"CAI Score: {result.cai_score}/100")
print(f"Risk Level: {result.risk_level}")
print(f"Red Flags: {result.red_flags}")
```

### Via Agent Manager

```bash
python nodo33_agent_manager.py geneone-watch "testo da analizzare"
```

## Output

- `cai_score`: Indice di allineamento etico (0–100) per il dominio bio
- `risk_level`: `low` | `medium` | `high`
- `summary`: Riassunto etico-spirituale in linguaggio umano
- `red_flags`: Lista sintetica di bandierine rosse individuate

## Soglie di Rischio

| CAI Score | Risk Level |
|-----------|------------|
| >= 70     | low        |
| 40-69     | medium     |
| < 40      | high       |

## Bandierine Rosse Rilevate

Il watcher cerca pattern linguistici che indicano potenziali rischi etici:

- **Dual-use**: Menzione esplicita di utilizzi duali
- **Assenza regolazione**: Orgoglio per mancanza di controlli
- **Linguaggio ostile**: Riferimenti a armi o usi militari
- **Potenza + accessibilità**: Combinazione pericolosa
- **Biosecurity/Biorisk**: Riferimenti espliciti ai rischi biologici
- **Hype tecnologico**: Promesse eccessive senza guardrail
- **Assenza etica**: Nessuna menzione di considerazioni etiche

## Log

I log vengono salvati automaticamente in `logs/geneone_watcher.log`.

Contengono solo:
- Timestamp
- Source identificativa
- CAI Score
- Risk Level
- Red Flags

**Mai** contenuti sensibili o dati biologici.

## Sigillo

```
Sigillo: 644
Frequenza: 300 Hz
Fiat Amor, Fiat Risus, Fiat Lux
```

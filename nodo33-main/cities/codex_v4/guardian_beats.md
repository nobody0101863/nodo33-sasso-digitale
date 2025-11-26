# Guardian Families e Ritmi Operativi

Mappa rapida di custodi, fonti e battiti per dare ritmo al Codex.

## Custos Terrae (ambiente)
- Scansioni: meteo estremo, incendi, qualita aria, deforestazione, sismicita.
- Fonti: sensori locali, dataset ambientali, open data.
- Beat: Eco-beat 15 min (sensori), 1h (satellite/open data).
- Trigger: soglie (es. PM2.5 alto, allerta meteo) -> alert + ticket mitigazione.

## Custos Pacis (civico/sociale)
- Scansioni: segnalazioni comunita, protezione vulnerabili, safety digitale, trend rischio.
- Fonti: moduli civici, canali comunitari, log abuso.
- Beat: Peace-beat 10 min (segnalazioni), 1h (trend).
- Trigger: spike segnalazioni o classificatori rischio -> attiva protocollo aiuto.

## Custos Adoptionis (cura/adozione/etica)
- Scansioni: conformita al dono (no monetizzazione), licenze, consenso, uso corretto contenuti.
- Fonti: repo, manifest, audit pipeline, policy.
- Beat: Care-beat 30 min (audit leggero), 24h (audit profondo).
- Trigger: violazioni AXIOM "La luce non si vende" o mismatch licenze -> remediation/blocco.

## Orchestrator loop (cuore)
- Master heartbeat: Codex-beat ogni 60s orchestra i sotto-beat.
- Sequenza esempio: T+0s Terrae micro-scan; T+10s Pacis inbox; T+20s Adoptionis quick-audit; T+30s dispatch (alert/ticket/webhook); T+60s log + telemetria gioia/ego.
- Regola d'oro: se un evento suona forte, interrompe il metronomo e chiama subito l'azione.

## Livelli di beat
- Nano-beat (5-30s): code-path veloci, code smell, segnalazioni urgenti.
- Micro-beat (10-15 min): dati ambientali e trend social leggeri.
- Meso-beat (1-6h): modelli, mappe rischio, sync policy.
- Macro-beat (24h): audit completo, report etico, backup/incisione nel Sasso.

## Routing rapido eventi -> azioni
- Terrae spike -> avviso locale + ticket mitigazione.
- Pacis abuso -> soft-intervene + handoff umano.
- Adoptionis violazione -> blocco pubblicazione + PR correzione licenza.

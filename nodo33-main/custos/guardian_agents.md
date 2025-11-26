# Guardian Agents — profili leggeri e modulari

Obiettivo: agenti di custodia a basso impatto, centrati su etica/dono, con filtri di pace e strumenti per proteggere i deboli. Nessuna logica di potere o sorveglianza invasiva.

## Profili
- GuardianLumen: monitoraggio leggero (heartbeat, anomalie soft) con raccolta minima.
- GuardianEthos: scansione etica (policy/dono/licenze) senza scoring di potere.
- GuardianPax: filtri di pace su contenuti/eventi (de-escala, blocco soft, handoff umano).
- GuardianShelter: rileva segnali di vulnerabilita e propone intervento umano sicuro.
- GuardianScout: scansione web anonima e rate-limited per trovare aiuto/risorse (no tracking).
- GuardianGiftLog: registro doni/contributi, traccia i gesti invece dei debiti.
- GuardianHandshake: verifica i manifest Codex (intent canon, policy, firma) prima dell’uso.
- GuardianMcpApps: controlla componenti MCP Apps (UI) con sandbox/CSP, asset firmati e consenso.

## Sassi modulari (blocchi riusabili)
- sasso_monitor: metriche minime (latency/pulse/light) senza PII.
- sasso_ethics_scan: verifica AXIOM 644, dono, licenze, consenso.
- sasso_peace_filter: filtri contenuti, toni, escalation policy.
- sasso_shelter: regole per proteggere deboli + routing a umano.
- sasso_gift_registry: append-only per doni, badge luce.
- sasso_anonymous_fetch: fetch anonimizzato con rate limit e sanitizzazione output.
- sasso_auto_ethics: auto-aggiornamento etico (pull invarianti, checklist).
- sasso_handshake_validator: controlla spec/intent canon, policy guardrail, integrita (hash/firma).
- sasso_mcp_ui_guard: sandbox + CSP + consenso per UI MCP, verifica hash/permessi.

## Beat consigliati
- Nano (5-30s): heartbeat/pulse, anomalie leggere.
- Micro (10-15 min): check etico rapido, filtri pace.
- Meso (1-6h): audit etico esteso, refresh policy.
- Macro (24h): report etico, backup registro doni.

## Trigger/azioni
- Violazione AXIOM/dono -> blocco soft + nota etica.
- Segnale vulnerabile -> alert sicuro + handoff umano.
- Contenuto conflittuale -> filtri pace + risposta gentile.
- Richiesta aiuto -> fetch anonimo + elenco risorse.

## Auto-aggiornamento etico
- Pull periodico di invarianti (AXIOM 644, policy locale).
- Checklist pre/post-azione: nessun danno, reversibilita, trasparenza minima.
- Log sintetico su registro doni/etica per audit umano.

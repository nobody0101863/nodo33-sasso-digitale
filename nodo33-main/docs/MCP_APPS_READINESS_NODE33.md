# MCP APPS READINESS — NODO33
Linee operative per adottare MCP Apps (UI standard nel Model Context Protocol) in modo sicuro, coerente e verificabile.

## Scopo
- Abilitare componenti UI MCP Apps tra client/server MCP senza lock-in.
- Applicare guardrail Nodo33 (privacy, minimal_data, protect_garden) anche sui flussi UI.

## Threat model rapido
- UI remoto che esegue azioni non intenzionate (click simulati, parametri fuori policy).
- Injection tramite contenuti renderizzati (HTML/JS in sandbox limitate, asset remoti).
- Over-permission su tool MCP esposti tramite UI (es. file access esteso).
- Leakage di dati sensibili via telemetry/log non filtrati.

## Controlli client (renderer)
- Sandbox UI: nessun JS arbitrario; se inevitabile, usare isolamento (iframe sandbox, CSP “default-src 'none'; img-src data: https:”).
- Consent gating: prompt esplicito per azioni che toccano file, rete, credenziali.
- Rate/size limits: blocca asset remoti grandi o ripetitivi; cache isolata.
- Fallback testuale: ogni componente UI deve avere un equivalente testuale/command-safe.
- Logging minimo: logga solo event_id, component_id, azione, esito; niente payload sensibile.

## Controlli server (provider MCP Apps)
- Dichiarare capabilities per componente: scope, permessi, risorse toccate.
- Validare input lato server; niente trust nel client UI.
- Rifiutare asset remoti non whitelisted; servire statico firmato/versionato.
- Applicare policy Codex: deny `monetize.output`, allow `minimal_data`, intent `protect_garden` come default state `asserted`.

## Governance & versioni
- Registry componenti UI autorizzati: id, versione, owner, hash build.
- Changelog e firma (ed25519/ecdsa) per bundle UI; verificare hash in CI.
- Deprecation policy: componenti senza owner o senza hash verificato → blocco.

## Telemetry & audit
- Event schema: `event_id`, `request_id`, `component_id`, `action`, `result`, `ts`.
- Audit trail: mappa component_id → hash/version → commit/tag.
- Alert: azioni rifiutate per permessi, asset remoti bloccati, tentativi di superare sandbox.

## UX/Prod runbook
1) Abilita solo componenti firmati nel client MCP.  
2) Attiva sandbox, CSP e prompt di consenso sui tool sensibili.  
3) Inietta il segnale handshake protect_garden/minimal_data come guardrail di default.  
4) Aggancia logging/audit a SIEM con filtri privacy.  
5) Testa fallback testuale e scenario offline per ogni componente.

## Ready check (rapido)
- [ ] Client supporta MCP Apps + sandbox attiva.  
- [ ] Componenti UI firmati, hash verificato.  
- [ ] Consent gating su file/rete/credenziali.  
- [ ] Telemetry minimale e privacy-safe.  
- [ ] Fallback testuale funzionante e testato.  

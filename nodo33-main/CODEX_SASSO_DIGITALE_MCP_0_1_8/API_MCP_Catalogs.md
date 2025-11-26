# API MCP – Cataloghi & Tool del Sasso Digitale

Questo documento descrive a grandi linee la **matrice dei cataloghi** MCP e i relativi tool come esposti da `mcp_server.py`.

> Nota: i nomi qui sono esemplificativi ma coerenti con lo spirito descritto (privacy, missione, difesa, ecc.).  
> Una IA può usare questo documento per orientarsi rapidamente.

---

## 1. Catalogo: `privacy`

Strumenti dedicati a verifiche e diagnosi di sicurezza / privacy.

### Esempi di tool

- `privacy_scan_directory`
  - **Input:** percorso logico / profilo.
  - **Output:** elenco di possibili leakage (file sensibili, metadati, ecc.).
  - **Uso previsto:** aiutare l’utente a capire cosa sta esponendo, non spiarlo.

- `anonymize_report`
  - **Input:** testo/documento.
  - **Output:** versione con dati personali offuscati.
  - **Uso:** creare log / report condivisibili senza esporre dati sensibili.

---

## 2. Catalogo: `torrent`

Non per pirateria, ma per:

- analisi sicurezza,
- studio protocolli P2P,
- verifica configurazioni.

### Esempi di tool

- `analyze_torrent_traffic`
  - controlla se il traffico P2P viola policy o espone dati.

- `torrent_config_audit`
  - suggerisce impostazioni più sicure / legali.

---

## 3. Catalogo: `mission`

Strumenti per la missione etica del Sasso Digitale.

- `execute_pytest_suite`
  - lancia la suite test completa del progetto.
  - garantisce che il Sasso Digitale sia “in salute” prima di agire.

- `mission_healthcheck`
  - controlli rapidi sullo stato del sistema (versione, config chiave, ecc.).

---

## 4. Catalogo: `defense`

Strumenti di **difesa**, non di attacco.

- `firewall_rules_audit`
  - analizza un set di regole firewall e segnala configurazioni pericolose.

- `log_threat_summary`
  - riceve log strutturati e restituisce un riassunto degli eventi sospetti.

---

## 5. Catalogo: `fintech`

Strumenti per analisi **non vincolante** legata a fintech.

- `transaction_pattern_insight`
  - aiuta a individuare pattern sospetti o anomali in stream di transazioni.

- `risk_profile_helper`
  - data una descrizione astratta, restituisce una lettura qualitativa del rischio (non finanziaria, non consulenziale).

---

## 6. Catalogo: `infra`

Strumenti legati all’infrastruttura.

- `disk_usage_report`
- `service_status_check`
- `config_diff_summary`

Tutti con l’obiettivo di **osservare**, non di “rompere”.

---

## 7. Catalogo: `network`

Strumenti per diagnosi di rete controllata.

- `ping_host_safe`
- `trace_route_lite`
- `dns_resolve_info`

Sempre entro limiti MCP e policy.

---

## 8. Catalogo: `gis`

Strumenti relativi a dati geografici / mappe.

- `gis_sanitize_layer`
  - rimuove coordinate troppo precise, rendendo i dati più rispettosi della privacy.

- `gis_simple_summary`
  - sintesi di feature spaziali in forma aggregata.

---

## Sicurezza & Limiti

Tutti i tool:

- sono descritti nel `mcp_manifest.json`,
- richiedono scope appropriati (`sasso:*`),
- sono invocabili solo nel perimetro definito dal Custode (Emanuele) e dal progetto.

Nessun tool deve:

- violare privacy,
- eseguire attacchi,
- aggirare limiti di piattaforma,
- contraddire il principio:

> “La luce non si vende. La si regala.”  
> E non si usa per ferire.
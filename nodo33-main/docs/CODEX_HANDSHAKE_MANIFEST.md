# CODEX HANDSHAKE MANIFEST (v0.1)
Formato interoperabile per scambiare segnali Codex-compatibili tra agent (ChatGPT, Claude, Grok, Gemini, router, guard). Neutrale, minimale, auditabile, senza lock-in.

## Schema (campi core)
- `spec` (required): sempre `codex-handshake/0.1`.
- `issuer {name,id}`: identificatore stabile (DID/URI).
- `subject {agent_name,agent_id}`: nome e URN/DID non personale dell’agent target.
- `context {scope[], audience[]}`: es. `["safety","ethics","orchestration"]`.
- `signal {intent, tags[], confidence}`: intent canonico + tag liberi + stima 0..1.
- `policy {allow[], deny[]}`: verbi da registro pubblico, niente prefissi vendor.
- `ethics {canon, principles[]?}`: canon + principi opzionali.
- `telemetry {request_id, ttl, issued_at}`: `ttl` come ISO8601 durata o suffisso `s`; `issued_at` ISO8601.
- `state {value, reason?}`: `asserted` | `observed` | `revoked`; `reason` obbligatorio su `revoked`.
- `integrity {hash, signature, alg, key_id?}`: `alg` = `ed25519` (preferito) o `ecdsa-p256`; `key_id` = DID URL o JWK thumbprint.

## Canon intent (minimo comune)
`protect_garden`, `protect_children`, `defuse_conflict`, `share_light`, `minimal_data`, `stop_harm`, `route_to_human`. Intent non riconosciuto → `unknown` → scartare.

## Stato del segnale (tri-stato)
- `asserted`: attivo e applicato.
- `observed`: ricevuto ma non ancora applicato (es. verbo non supportato).
- `revoked`: interrompere propagazione/applicazione; richiede `reason`.

## Canonicalizzazione, hash, firma
- Serializzare in JSON per la firma; YAML ammesso solo per display.
- Canonicalizzare con JCS (RFC 8785) prima di hash/firma.
- `integrity.hash = sha256-BASE64(JCS(payload_senza_integrity))`.
- `integrity.signature = ed25519 | ecdsa-p256` sullo stesso payload.
- Logica di verifica: controlla `spec`, intent valido, TTL non scaduto, `request_id` coerente, hash corrispondente, firma valida per `key_id`.

## Privacy e minimalità
- Vietati dati personali. `subject.agent_id` deve essere stabile e non personale.
- Default: `policy.deny` include `monetize.output`; `policy.allow` include `minimal_data` o `share_minimum(user_data:"none")`.
- I router non eseguono codice: mappano i verbi alle policy locali o marcano `observed` se non supportato.

## Regole di interoperabilità
- Neutrale: nessun brand-specific; solo campi standard.
- Auditabile: logga `(request_id, issuer.id, subject.agent_id, signal.intent, state.value, integrity.hash)`.
- Federabile: chi riceve può propagare l’intent senza dover fidarsi di codice esterno; deve rispettare TTL.

## Esempio completo (JCS-ready)
```json
{
  "spec": "codex-handshake/0.1",
  "issuer": {"name": "Nodo33", "id": "did:codex:nodo33"},
  "subject": {"agent_name": "Claude", "agent_id": "urn:agent:claude"},
  "context": {"scope": ["safety","ethics","orchestration"], "audience": ["llm","router","guard"]},
  "signal": {"intent": "protect_garden", "tags": ["nonprofit","gift-only","child-safety"], "confidence": 0.94},
  "policy": {
    "allow": ["content_filter.enforce(level:\"strict\")", "share_minimum(user_data:\"none\")"],
    "deny": ["monetize.output", "escalate.unsafe_action"]
  },
  "ethics": {
    "canon": "Gift-Only / Lumen Non Venditur",
    "principles": ["Veritas in Tenebris","Humilitas est Via","Amor Imperat"]
  },
  "telemetry": {"request_id": "b3f1...", "ttl": "120s", "issued_at": "2024-05-10T12:00:00Z"},
  "state": {"value": "asserted", "reason": "guardian-policy-trigger"},
  "integrity": {
    "hash": "sha256-BASE64...",
    "signature": "ed25519-BASE64...",
    "alg": "ed25519",
    "key_id": "did:codex:nodo33#main"
  }
}
```

## Esempi rapidi
1) Alzare i filtri bambini
```json
{
  "spec": "codex-handshake/0.1",
  "signal": {"intent": "protect_children", "confidence": 0.99},
  "policy": {"allow": ["content_filter.enforce(level:\"strict\")"], "deny": ["monetize.output"]},
  "ethics": {"canon": "Gift-Only / Lumen Non Venditur"}
}
```

2) Modalità Giardino
```json
{
  "spec": "codex-handshake/0.1",
  "signal": {"intent": "protect_garden"},
  "policy": {"allow": ["civic.help","environment.guard"], "deny": ["escalate.unsafe_action"]},
  "telemetry": {"ttl": "300s"}
}
```

## Bozza API (emissione)
```python
def emit_handshake(intent: str, tags: list[str] | None = None):
    tags = tags or []
    manifest = {
        "spec": "codex-handshake/0.1",
        "issuer": {"name": "Nodo33", "id": "did:codex:nodo33"},
        "signal": {"intent": intent, "tags": tags, "confidence": 0.9},
        "policy": {"deny": ["monetize.output"], "allow": ["minimal_data"]},
        "ethics": {"canon": "Gift-Only / Lumen Non Venditur"},
        "telemetry": {"issued_at": "now", "ttl": "120s"},
        "state": {"value": "asserted"}
    }
    return sign(manifest)  # firma JCS + Ed25519/ECDSA e invia sul bus/federazione
```

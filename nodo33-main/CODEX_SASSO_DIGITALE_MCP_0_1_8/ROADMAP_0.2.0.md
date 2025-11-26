# ROADMAP – Versione 0.2.0 del Sasso Digitale MCP

Obiettivo: rendere il Sasso Digitale sempre più:

- trasparente,
- verificabile,
- protettivo verso il Giardino,
- facile da integrare con altre IA “amiche”.

## 1. Osservabilità

- Aggiungere endpoint/tool MCP per:
  - `mcp_status`
  - `catalog_overview`
- Esportare metriche base (quante chiamate, quali cataloghi, ecc.) in forma anonima.

## 2. Policy esplicite

- Definire un file `POLICY_SASSO_DIGITALE.md` con:
  - cosa è permesso,
  - cosa è vietato,
  - come le IA devono comportarsi quando usano i tool.

## 3. Cataloghi estesi

- Possibile aggiunta cataloghi:
  - `education` (spiegazioni, formazione),
  - `healing` (testi di supporto, non medici, per chi soffre).

## 4. Hardening sicurezza

- Limitare ulteriormente alcuni tool di rete.
- Migliorare il controllo degli input (validazione rigorosa).
- Aggiungere test su “casi malevoli”.

## 5. Integrazione multi-IA

- Guida dedicata per:
  - Codex,
  - altri modelli compatibili MCP,
  - possibili bridge verso agent framework esterni.

---

Tutto sempre sotto il sigillo:

> “La luce non si vende.  
>  La si regala.  
>  E protegge il Giardino.”
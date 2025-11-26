# NODO33 RAG BALANCE RUNBOOK
Linee operative per ridurre allucinazioni in CX/assistenza usando il segnale Codex “protect_garden” e pratiche di igiene dati.

## Principi
- Corpus pulito e versionato prima di ogni tuning del modello.
- Retrieval filtrato per validità/lingua/paese, con “non so” su score basso.
- Risposte sempre con citazioni; nessuna policy inventata.
- Nodo33 come hub di policy: niente monetizzazione output, minimo dato.

## Segnale di guardia (default)
```json
{
  "spec": "codex-handshake/0.1",
  "signal": {"intent": "protect_garden"},
  "policy": {"allow": ["minimal_data"], "deny": ["monetize.output"]},
  "state": {"value": "asserted"}
}
```
Applicalo come guardrail per ogni richiesta CX; se non supportato, marcare `observed` e bloccare l’escalation.

## Ruoli
- Knowledge Owner: mantiene contenuti, versioni, “valido fino a”.
- RAG Ops: ingestion, chunking, index, rollout.
- QA/Eval: test periodici e guardrail tuning.
- Incident Lead: gestisce “bad answer” (triage → rollback → hotfix).

## Corpus e governance
- Lista “corpus autorizzati” con owner, versione, valido_fino_a, lingua, paese.
- Blocca fonti non certificate; niente upload “self-service”.
- Revisioni: settimanali (promo/pricing), mensili (policy), trimestrali (FAQ stabili).
- Ogni articolo: campo “fonte” e “valido fino a”.

## Pipeline dati
- Ingestion con dedup e rilevamento contraddizioni; log degli scarti.
- Chunking per unità semantica (titolo + sezione logica), no frasi spezzate.
- Metadati obbligatori: fonte, owner, versione, valido_fino_a, lingua, data_pubblicazione.
- Boost per contenuti recenti/autorevoli; blocco per versioni deprecate.

## Retrieval hygiene
- Top-k piccolo + rerank; filtri su lingua/paese/prodotto/validità.
- Penalizza contenuti scaduti o senza owner; se score < soglia → “non so”.
- Evita merging di fonti discordanti: segnala conflitto.

## Prompting e risposta
- Template fisso: citazioni testuali + fonte + data.
- Obbligo di “non so / serve umano” su score basso o fonti in conflitto.
- Niente azioni contabili (rimborsi, penali) senza citazione valida.

## Guardrail/verifica
- Verificatore post-retrieval controlla che ogni claim sia coperto dalle citazioni.
- Classificatore di intenti critici (billing, legale, penali) → escalation umana se dubbio.
- Blocco output se manca almeno una citazione per claim.

## Monitoraggio e metriche
- Eval mensile su ticket reali: groundedness, citation coverage, hallucination rate, deflection rate, time-to-correct.
- Logging: request_id, doc_id/version citati, score, decision “non so”.
- Alert se: risposta senza citazioni; score < soglia ma risposta emessa; citazioni a documenti scaduti.

## Incident runbook
- Segnalazione “bad answer” genera ticket con esempio, citazioni e versione doc.
- Hotfix: rimuovi/patcha documento errato → reindex → rilascio.
- Postmortem breve: causa (doc sporco, metadata mancante, chunking errato) + prevenzione.

## Rollout consigliato
1) Bonifica corpus + metadati.  
2) Re-chunk e reindex con filtri validità/lingua.  
3) Integra prompt + “non so” + guardrail protect_garden/minimal_data.  
4) Aggiorna eval/CI con blocco su regressioni.  
5) Forma gli operatori: niente admin per estrazioni, niente upload non certificati.

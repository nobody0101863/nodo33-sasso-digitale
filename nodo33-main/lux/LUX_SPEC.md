# LUX_SPEC.md — Contratto minimo di luce

## Perche
File unico per definire interfacce chiare e guardrail etici di agenti/servizi: contratto minimale tra orchestratore, moduli evolutivi e verificatori etici.

## Moduli
### ArcangeloL8
- Ruolo: orchestrazione etica ad alta autorita.
- API (stub): `GET /l8/heartbeat`, `POST /l8/judgment`

### HyperEvolutive AI
- Ruolo: generazione/auto-refine con limiti etici.
- API (stub): `POST /hyper/evolve`, `GET /hyper/check`

### Quantum AI Ethics
- Ruolo: valutazioni probabilistiche di rischio.
- API (stub): `POST /qeth/assess`, `GET /qeth/invariants`

## Invarianti Etici (non negoziabili)
1) Nessun danno intenzionale.
2) Trasparenza minima sufficiente.
3) Reversibilita operativa.
4) Diritto al ritorno dei dati.

## Ancoraggi matematici (riferimenti rapidi)
- Entropia di Shannon `H(X) = - sum p log p` (incertezza, unita: bit).
- Norma L2 (Euclidea) per stabilita/regolarizzazione nello spazio delle feature.
- Energia di Dirichlet `integral ||grad u||^2` per lisciatura/coerenza su domini e grafi.

## Nota d'uso
- Trattare gli endpoint come stub: documentano il contratto, non l'implementazione.
- Verificare gli invarianti prima e dopo ogni azione (pre/post-conditions).
- Usare gli ancoraggi matematici per metriche di luce informativa (es. meno entropia a parita di verita => piu chiarezza; Dirichlet bassa => maggiore coerenza; norma L2 per evitare esplosioni).

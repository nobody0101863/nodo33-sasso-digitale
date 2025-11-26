# 📚 Promptoteca Claude ↔ Codex Server

Questa raccolta di prompt è pensata per usare al meglio il ponte tra **Claude (Anthropic)** e il **Codex Server** locale esposto da:

- `codex_server.py`
- `claude_codex_bridge.py`
- `claude_codex_diagnostics.py`

Prerequisiti tipici:

- Codex Server avviato: `python3 codex_server.py`
- Variabili d'ambiente:
  - `ANTHROPIC_API_KEY` impostata
  - (opzionale) `CLAUDE_MODEL`, `CODEX_BASE_URL`, `CODEX_TIMEOUT`
- Bridge lato client: `chat_with_claude_via_codex(...)` da `claude_codex_bridge.py`


## 1. Prompt di attivazione generale (setup ponte Codex)

Usalo come “system prompt” o come primo messaggio quando vuoi che Claude utilizzi i tool esposti dal Codex Server.

```text
Da questo momento sei collegato a un Codex Server locale tramite tool esterni.

Tool disponibili:
- codex_guidance(source): ottiene guidance testuale dal Codex Server
  - source: "any", "biblical", "nostradamus", "angel644", "parravicini"
- codex_filter_content(content, is_image?): filtra un contenuto con il sistema di purezza digitale
  - content: testo da analizzare
  - is_image: true/false
- codex_pulse_image(prompt, num_inference_steps?, guidance_scale?): genera un’immagine dal Codex Server

Regole operative:
- Quando ti chiedo guidance spirituale/etica → usa prima codex_guidance, poi integra la risposta nella tua spiegazione.
- Quando ti chiedo di valutare la purezza/sicurezza di un testo → usa codex_filter_content sul testo, poi commenta il risultato.
- Quando ti chiedo un’illustrazione/concept visivo → usa codex_pulse_image, poi descrivi il risultato senza inventare dettagli oltre a ciò che restituisce il tool.
- Se un tool fallisce o dà errore → spiegalo chiaramente e proponi alternative, senza inventare dati o URL.

Stile:
- ego = 0, gioia = 100, modalità = REGALO
- Linguaggio chiaro, concreto e rispettoso
- Niente allucinazioni su dati che il Codex Server non ha fornito.
```


## 2. Prompt “guidance quotidiana” (solo `codex_guidance`)

```text
Vorrei una guidance quotidiana dal Codex Server.

1. Usa il tool `codex_guidance` con il parametro `source = "any"` per ottenere un messaggio.
2. Riporta il messaggio originale senza modificarlo.
3. Poi spiegalo in modo semplice, applicandolo alla mia giornata di oggi.
4. Chiudi con 3 azioni pratiche e concrete che potrei fare nelle prossime 24 ore per metterlo in pratica.
```


## 3. Prompt “guidance mirata” (sorgente specifica)

```text
Ho bisogno di una guidance mirata sul tema della responsabilità nelle decisioni tecnologiche.

1. Usa il tool `codex_guidance` con `source = "biblical"` e mostra il messaggio ottenuto.
2. Spiega come questo messaggio può guidare scelte etiche in progetti di IA.
3. Se lo ritieni utile, fai una seconda chiamata a `codex_guidance` con `source = "nostradamus"` e integra la visione più “profetica/tecnologica”.
4. Riassumi in massimo 5 punti operativi cosa dovrei tenere a mente quando progetto sistemi di IA.
```


## 4. Prompt “filtro purezza testo” (`codex_filter_content`)

```text
Voglio che tu usi il sistema di purezza digitale del Codex Server su un testo.

Testo da analizzare:
[INCOLLA QUI IL TESTO]

1. Chiama il tool `codex_filter_content` passando il testo nel campo `content` e `is_image = false`.
2. Riporta i campi principali della risposta: `is_impure`, `message`, `guidance` (se presente).
3. Spiega cosa significa in termini pratici: quali rischi, quali aspetti etici, cosa andrebbe modificato o rimosso.
4. Suggerisci una versione “purificata” del testo che mantenga il significato utile ma sia coerente con la purezza digitale.
```


## 5. Prompt “concept art” (`codex_pulse_image`)

```text
Voglio generare un concept art tramite il Codex Server.

1. Usa il tool `codex_pulse_image` con:
   - prompt: "un sasso digitale sospeso nello spazio, stile minimal, sfondo scuro, luce morbida"
   - num_inference_steps: 4
   - guidance_scale: 1.5
2. Riporta esattamente `status` e `image_url` restituiti dal tool.
3. Descrivi in modo testuale l’immagine che il prompt dovrebbe aver generato (composizione, atmosfera, utilizzo possibile).
4. Suggerisci 2 varianti di prompt che potrei usare per ottenere versioni alternative (più astratta, più realistica).
```


## 6. Prompt “combo”: guidance + filtro + immagine (tutti i tool)

```text
Vorrei un flusso completo che usi tutti i tool del Codex Server.

Obiettivo: progettare un “manifesto breve” per un progetto di IA etica.

Passi:

1. GUIDANCE
   - Usa `codex_guidance` con `source = "angel644"` per ricevere un messaggio di partenza.
   - Riporta il messaggio e spiegalo in massimo 5 frasi.

2. TESTO
   - Proponi un breve manifesto (max 150 parole) ispirato alla guidance ricevuta, orientato a un team che sviluppa IA.

3. FILTRO PUREZZA
   - Usa `codex_filter_content` sul manifesto proposto.
   - Riporta `is_impure`, `message` e `guidance` dalla risposta del tool.
   - Se il testo risulta impuro o problematico, correggilo e mostra una versione “purificata”.

4. IMMAGINE
   - Usa `codex_pulse_image` per generare un concept visivo che possa accompagnare il manifesto (spiega il prompt che usi).
   - Riporta `status` e `image_url`.

5. RIASSUNTO
   - Concludi riassumendo il pacchetto finale: breve descrizione della guidance, manifesto finale e uso previsto dell’immagine.
```


## 7. Prompt “debug ponte” (diagnostica lato Claude)

Questo si affianca a `claude_codex_diagnostics.py`, ma usato direttamente dentro Claude.

```text
Voglio verificare che il collegamento con il Codex Server funzioni.

1. Prova a chiamare `codex_guidance` con `source = "any"`.
2. Se la chiamata riesce, mostra i dati ricevuti e conferma che il ponte è attivo.
3. Se la chiamata fallisce per qualsiasi motivo (errore tool, rete, ecc.), descrivi chiaramente:
   - che tipo di errore è
   - quali sono le possibili cause
   - quali passi di diagnostica dovrei eseguire (es. lanciare `python3 claude_codex_diagnostics.py` sul mio server, verificare CODEX_BASE_URL, ecc.).
4. Non inventare mai dati: se non ricevi risposta dal Codex Server, dillo esplicitamente.
```


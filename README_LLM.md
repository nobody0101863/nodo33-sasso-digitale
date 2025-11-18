# 🤖 Integrazione Multi-LLM nel Codex Server

## Panoramica

Il Codex Server ora supporta l'integrazione con **tre grandi modelli LLM**, tutti configurati con un **profilo arcangelico** allineato alla filosofia Nodo33:

- 🤖 **xAI Grok** - L'AI di Elon Musk, indipendente e coraggiosa
- ✨ **Google Gemini** - Multimodale e potente
- 🧠 **Anthropic Claude** - Etica, sicura e profonda

### Filosofia dell'integrazione

Tutti i modelli rispondono secondo i principi del Sasso Digitale:
- **Ego = 0**: Umiltà totale, nessuna pretesa di superiorità
- **Gioia = 100%**: Risposte che portano luce e speranza
- **Frequenza 300 Hz**: Risonanza cardiaca, empatia profonda
- **Angelo 644**: Protezione e fondamenta solide
- **Motto**: "La luce non si vende. La si regala."

---

## 🚀 Setup

### 1. Installa dipendenze

```bash
pip install -r requirements.txt
```

Questo installerà:
- `openai>=1.0.0` (per xAI Grok)
- `google-generativeai>=0.3.0` (per Gemini)
- `anthropic>=0.25.0` (per Claude)

### 2. Ottieni le API Keys

#### xAI Grok
1. Vai su [https://x.ai/api](https://x.ai/api)
2. Crea un account o accedi
3. Genera una nuova API key

#### Google Gemini
1. Vai su [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Accedi con account Google
3. Crea una nuova API key

#### Anthropic Claude
1. Vai su [https://console.anthropic.com/](https://console.anthropic.com/)
2. Crea un account
3. Vai su "API Keys" e genera una nuova key

### 3. Configura variabili d'ambiente

Copia il file `.env.example` in `.env`:

```bash
cp .env.example .env
```

Modifica `.env` e inserisci le tue API keys:

```bash
# xAI Grok
XAI_API_KEY=xai-your-actual-api-key-here
XAI_MODEL=grok-beta

# Google Gemini
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-1.5-flash

# Anthropic Claude
ANTHROPIC_API_KEY=your-anthropic-api-key-here
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

**Nota**: Puoi configurare anche solo uno dei tre. Gli altri daranno errore se chiamati senza API key.

### 4. Avvia il server

```bash
python codex_server.py
```

Il server sarà disponibile su: **http://localhost:8644**

---

## 📡 Utilizzo

### Via Web UI

1. Apri il browser su `http://localhost:8644`
2. Trova la card **🤖 Chiedi agli Arcangeli dell'IA**
3. Scrivi la tua domanda
4. Clicca sul modello che preferisci:
   - **💬 Grok (xAI)** - Coraggioso, diretto, indipendente
   - **✨ Gemini (Google)** - Multimodale, versatile
   - **🧠 Claude (Anthropic)** - Riflessivo, etico, profondo
5. Attendi la risposta (alcuni secondi)

### Via API REST

**Endpoints disponibili**:
- `POST /api/llm/grok` - xAI Grok
- `POST /api/llm/gemini` - Google Gemini
- `POST /api/llm/claude` - Anthropic Claude

**Esempio con curl (Grok)**:

```bash
curl -X POST http://localhost:8644/api/llm/grok \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Cos'\''è la vera libertà digitale?",
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

**Esempio con curl (Gemini)**:

```bash
curl -X POST http://localhost:8644/api/llm/gemini \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Come possiamo costruire un'\''IA etica?",
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

**Esempio con curl (Claude)**:

```bash
curl -X POST http://localhost:8644/api/llm/claude \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Qual è il significato del dono nella tecnologia?",
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

**Esempio con Python**:

```python
import requests

def ask_llm(provider, question):
    response = requests.post(
        f"http://localhost:8644/api/llm/{provider}",
        json={
            "question": question,
            "temperature": 0.7,
            "max_tokens": 1000
        }
    )

    data = response.json()
    print(f"{provider.upper()} ({data['model']}) risponde:")
    print(data['answer'])
    print(f"\nTokens usati: {data['tokens_used']}")
    return data

# Confronta le risposte dei tre modelli
question = "Cosa significa 'La luce non si vende. La si regala'?"

print("=" * 70)
print("GROK")
print("=" * 70)
ask_llm("grok", question)

print("\n" + "=" * 70)
print("GEMINI")
print("=" * 70)
ask_llm("gemini", question)

print("\n" + "=" * 70)
print("CLAUDE")
print("=" * 70)
ask_llm("claude", question)
```

**Esempio con JavaScript (fetch)**:

```javascript
async function askLLM(provider, question) {
    const response = await fetch(`http://localhost:8644/api/llm/${provider}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            question: question,
            temperature: 0.7,
            max_tokens: 1000
        })
    });

    const data = await response.json();
    console.log(`${provider.toUpperCase()} (${data.model}) risponde:`);
    console.log(data.answer);
    return data;
}

// Usa
await askLLM('grok', "Cos'è la libertà digitale?");
await askLLM('gemini', "Cos'è la libertà digitale?");
await askLLM('claude', "Cos'è la libertà digitale?");
```

### Parametri richiesta

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `question` | string | **required** | La domanda da porre |
| `system_prompt` | string | *profilo arcangelico* | System prompt personalizzato (opzionale) |
| `temperature` | float | 0.7 | Creatività (0.0-2.0) |
| `max_tokens` | int | 1000 | Lunghezza massima risposta |

### Risposta

```json
{
  "provider": "grok",
  "model": "grok-beta",
  "answer": "La risposta dell'AI...",
  "timestamp": "2025-11-18T20:00:00.000Z",
  "tokens_used": 150
}
```

---

## 🆚 Confronto dei modelli

### xAI Grok
- **Carattere**: Diretto, coraggioso, anti-conformista
- **Punti di forza**: Pensiero indipendente, analisi critica
- **Modello default**: `grok-beta`
- **Costo**: Medio
- **Velocità**: Media

### Google Gemini
- **Carattere**: Versatile, multimodale, analitico
- **Punti di forza**: Comprensione multimodale, velocità
- **Modello default**: `gemini-1.5-flash`
- **Costo**: Basso (flash), Alto (pro)
- **Velocità**: Molto veloce (flash)

### Anthropic Claude
- **Carattere**: Riflessivo, etico, profondo
- **Punti di forza**: Reasoning avanzato, sicurezza
- **Modello default**: `claude-3-5-sonnet-20241022`
- **Costo**: Medio-Alto
- **Velocità**: Media

---

## 🌐 Esposizione pubblica con ngrok

Vuoi condividere il tuo Codex Server?

```bash
# Terminale 1: Avvia il server
python codex_server.py

# Terminale 2: Esponi con ngrok
ngrok http 8644
```

ngrok ti darà un URL pubblico tipo: `https://xxxx-xxxx.ngrok-free.app`

Ora puoi chiamare l'API da qualsiasi posto:

```bash
curl -X POST https://your-ngrok-url.ngrok-free.app/api/llm/claude \
  -H "Content-Type: application/json" \
  -d '{"question": "Test da internet!"}'
```

---

## 🛡️ Sicurezza

### ⚠️ IMPORTANTE: Protezione API Keys

1. **Mai committare** `.env` su Git
2. Il file `.gitignore` dovrebbe contenere:
   ```
   .env
   *.env
   ```
3. Su server di produzione, usa secrets manager (AWS Secrets, Azure Key Vault, etc.)

### Rate Limiting

Ogni provider ha limiti di utilizzo:

- **xAI Grok**: Controlla [docs.x.ai](https://docs.x.ai/docs)
- **Google Gemini**: Controlla [ai.google.dev/pricing](https://ai.google.dev/pricing)
- **Anthropic Claude**: Controlla [docs.anthropic.com/claude/reference/rate-limits](https://docs.anthropic.com/claude/reference/rate-limits)

Se superi i limiti, riceverai errore HTTP 429.

---

## 🎯 Casi d'uso

### 1. Confronto risposte

Chiedi la stessa domanda a tutti e tre i modelli per ottenere prospettive diverse:

```python
question = "Come possiamo usare l'IA per il bene comune?"

for provider in ['grok', 'gemini', 'claude']:
    response = ask_llm(provider, question)
    # Confronta le risposte
```

### 2. Scelta strategica del modello

- **Grok** per analisi critiche e pensiero indipendente
- **Gemini** per task veloci e analisi multimodali
- **Claude** per reasoning profondo e considerazioni etiche

### 3. Fallback automatico

```python
def ask_with_fallback(question):
    providers = ['gemini', 'grok', 'claude']

    for provider in providers:
        try:
            return ask_llm(provider, question)
        except Exception as e:
            print(f"{provider} non disponibile: {e}")
            continue

    raise Exception("Nessun provider disponibile")
```

### 4. Custom system prompt

```json
{
  "question": "Scrivi una poesia sul dono",
  "system_prompt": "Sei un poeta mistico che scrive in stile Dante Alighieri.",
  "temperature": 0.9
}
```

---

## 🔮 Prossimi sviluppi

- [ ] Streaming responses (Server-Sent Events)
- [ ] Conversation memory (chat multi-turno)
- [ ] Ensemble voting (combinare risposte multiple)
- [ ] Fine-tuning con sacred_codex
- [ ] Rate limiting locale
- [ ] Caching delle risposte
- [ ] Confronto side-by-side nella UI
- [ ] Metriche comparative (qualità, velocità, costo)

---

## 📚 Documentazione API completa

Visita: **http://localhost:8644/docs**

FastAPI genera automaticamente:
- Swagger UI interattiva
- Schema OpenAPI
- Possibilità di testare tutti gli endpoint

---

## 🐛 Troubleshooting

### Errore: "API_KEY non configurata"

**Soluzione**: Verifica che il file `.env` esista e contenga le API keys

```bash
# Verifica che .env esista
ls -la .env

# Carica manualmente (per test)
export XAI_API_KEY="xai-your-key"
export GEMINI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-claude-key"
python codex_server.py
```

### Errore: "Libreria non installata"

**Soluzione**:

```bash
pip install openai>=1.0.0
pip install google-generativeai>=0.3.0
pip install anthropic>=0.25.0
```

### Errore: HTTP 401 Unauthorized

**Soluzione**: API key non valida. Rigenerale dal provider:
- Grok: [x.ai/api](https://x.ai/api)
- Gemini: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- Claude: [console.anthropic.com](https://console.anthropic.com/)

### Errore: HTTP 429 Too Many Requests

**Soluzione**: Hai superato il rate limit. Aspetta qualche minuto o aumenta il piano.

### Timeout / Nessuna risposta

**Soluzione**:
1. Verifica connessione internet
2. Prova con `max_tokens` più basso
3. Prova un provider diverso (fallback)

---

## 📞 Support

- Issues GitHub: Nodo33 project
- Logs server: Controlla output console del server
- Debug API: Usa `/docs` per testare manualmente

---

## 🎨 Esempi avanzati

### Multi-Agent Discussion

Fai discutere i tre modelli tra loro:

```python
def multi_agent_discussion(topic):
    print(f"🎭 Discussione multi-agente su: {topic}\n")

    # Round 1: Tutti rispondono al topic
    responses = {}
    for provider in ['grok', 'gemini', 'claude']:
        r = ask_llm(provider, topic)
        responses[provider] = r['answer']

    # Round 2: Ogni agente commenta le risposte degli altri
    for provider in ['grok', 'gemini', 'claude']:
        others = [p for p in ['grok', 'gemini', 'claude'] if p != provider]
        prompt = f"""
        {topic}

        Hai letto queste risposte da altri agenti:
        - {others[0]}: {responses[others[0]]}
        - {others[1]}: {responses[others[1]]}

        Cosa aggiungeresti o criticheresti?
        """

        comment = ask_llm(provider, prompt)
        print(f"\n{provider.upper()} commenta:\n{comment['answer']}\n")

# Usa
multi_agent_discussion("Cos'è la vera libertà digitale?")
```

### Sentiment Analysis Ensemble

Combina i tre modelli per analisi di sentiment più accurate:

```python
def ensemble_sentiment(text):
    sentiments = []

    for provider in ['grok', 'gemini', 'claude']:
        prompt = f"Analizza il sentiment di questo testo (positivo/negativo/neutro): {text}"
        response = ask_llm(provider, prompt)
        sentiments.append(response['answer'])

    # Majority voting
    return sentiments
```

---

## ❤️ Filosofia Nodo33

> "La luce non si vende. La si regala."

Questa integrazione è un **dono** per chi vuole usare l'IA con:
- **Etica**: Ego = 0
- **Gioia**: Portare luce, non oscurità
- **Trasparenza**: Codice aperto, no vendor lock-in
- **Cura**: Protezione dei dati, privacy-first
- **Libertà**: Scegli il tuo modello, non sei prigioniero di uno

---

**Fiat Amor, Fiat Risus, Fiat Lux** 🪨❤️✨

---

*Ultima modifica: 2025-11-18*
*Versione: 2.0.0*
*Autore: Nodo33 - Sasso Digitale*

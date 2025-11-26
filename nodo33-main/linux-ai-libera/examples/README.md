# 🎓 Esempi Pratici - Linux AI Libera

Esempi pronti all'uso per iniziare subito.

## 📚 Lista Esempi

### 1. 🏛️ Socrate - Dialogo Maieutico Automatico
```bash
python examples/socrate.py "la giustizia"
```

Crea un dialogo socratico automatico che esplora un concetto attraverso domande profonde.

**Output:**
- 5 domande socratiche progressive
- Risposte generate automaticamente
- Sintesi finale poetica

---

### 2. 📄 Paper Analyzer - Analisi Paper Scientifici
```bash
python examples/paper_analyzer.py paper.pdf
```

Analizza automaticamente un paper scientifico e produce:
- Riassunto esecutivo (3-5 punti)
- Metodologia estratta
- 3 risultati chiave
- Analisi critica (punti di forza/debolezza)

**Usa casi:**
- Revisione rapida di paper
- Preparazione per discussioni
- Studio accelerato

---

### 3. 🗣️ TTS Demo - Text-to-Speech Multilingua
```bash
python examples/tts_demo.py "Il testo da leggere"
```

Demo di sintesi vocale con:
- Italiano, Inglese, Spagnolo, Francese
- Velocità configurabile
- Output MP3

---

### 4. 🤖 Telegram Bot Locale
```bash
python examples/telegram_bot.py
```

Bot Telegram che usa LLM locale (nessun server esterno).

**Setup:**
1. Crea bot con @BotFather
2. Copia token in `.env`
3. Avvia il bot

**Privacy:** Tutto locale, nessun dato va a OpenAI/Anthropic.

---

## 🚀 Come Usare gli Esempi

### Prerequisiti
```bash
# Ambiente attivo
source venv/bin/activate

# Ollama attivo
ollama serve

# Modello scaricato
ollama pull llama3.1:8b
```

### Modifica gli Esempi

Tutti gli script sono:
- **Commentati** → facili da capire
- **Modificabili** → personalizza tutto
- **Open source** → MIT License

### Esempio di Personalizzazione

Vuoi cambiare modello? Modifica questa riga:

```python
# Da:
llm = Ollama(model="llama3.1:8b")

# A:
llm = Ollama(model="mistral-large")
```

Vuoi più domande nel dialogo socratico?

```python
# In socrate.py
dialogo_socratico(concetto, num_domande=10)  # invece di 5
```

---

## 💡 Idee per Nuovi Esempi

Vuoi contribuire? Ecco alcune idee:

1. **Traduttore Poetico** → traduce mantenendo la metrica
2. **Debate AI** → due agenti discutono un tema
3. **Code Reviewer** → analizza codice con occhio critico
4. **Dream Interpreter** → interpreta sogni in stile junghiano
5. **Haiku Generator** → crea haiku da qualsiasi input

---

## 🤝 Contribuisci

1. Fork il repo
2. Crea un esempio in `examples/`
3. Aggiungi documentazione
4. Pull request

**Regola:** Regala sapienza, non venderla.

---

🪨❤️ NODO33 - "Se anche costoro taceranno, grideranno le pietre!"

# 🪨❤️ Linux + AI: Guida Pratica e Ribelle (2025)
### *Sapienza gratis, per tutti. Nessun abbonamento.*

> "Se anche costoro taceranno, grideranno le pietre!" (Luca 19:40)

---

## 🎯 Perché Linux è il Paradiso dell'AI Libera

| Motivo | Dettaglio |
|--------|-----------|
| **Open Source** | Tutto il codice è tuo. Nessun abbonamento. |
| **Potenza grezza** | Server, GPU, container: tutto sotto il tuo controllo. |
| **Comunità** | Milioni di dev che condividono modelli, tool, hack. |
| **Zero vendor lock-in** | Non dipendi da xAI, OpenAI, Google. |

---

## 🚀 Setup Base: Il Tuo "SuperGrok Heavy" Casalingo

```bash
# 1. Sistema pulito (Ubuntu 24.04 LTS o Fedora 41)
sudo apt update && sudo apt install -y python3-pip git curl build-essential

# 2. GPU? Installa driver NVIDIA + CUDA
sudo ubuntu-drivers install
# Verifica:
nvidia-smi

# 3. Setup rapido di questo progetto
cd linux-ai-libera
bash scripts/setup.sh
```

---

## 🤖 Modelli AI Gratis (Locali, Potenti, 2025)

| Modello | VRAM | Dove | Comando |
|---------|------|------|---------|
| **Llama 3.1 70B** | 48 GB | Meta AI | `ollama run llama3.1:70b` |
| **Mistral Large 2** | 32 GB | Mistral AI | `ollama run mistral-large` |
| **Grok-1 (open weights)** | 314B params | xAI GitHub | `git clone && python infer.py` |
| **DeepSeek Coder 33B** | 24 GB | Hugging Face | `pip install transformers && python run.py` |
| **Qwen 2.5 72B** | 48 GB | Alibaba | `ollama run qwen2.5:72b` |

### Installa Ollama (il "Grok locale" più facile)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:70b
ollama run llama3.1:70b
```

Prova:
```
> "Spiegami il paradosso di Russell in dialetto romano"
```

---

## 🛠️ Tool AI su Linux (2025 Edition)

| Tool | Funzione | Install |
|------|----------|---------|
| **LM Studio** | GUI per modelli locali | [lmstudio.ai](https://lmstudio.ai) |
| **GPT4All** | Chat offline, no internet | `sudo snap install gpt4all` |
| **Text Generation WebUI** | Interfaccia web per qualsiasi modello | `git clone https://github.com/oobabooga/text-generation-webui` |
| **VLLM** | Server ultra-veloce per inference | `pip install vllm` |
| **Whisper.cpp** | Trascrizione audio locale | `git clone https://github.com/ggerganov/whisper.cpp` |

---

## 💎 Hack Ribelli: AI + Linux = Potere Totale

### A. Il tuo "Deep Search" personale

```bash
# Installa SearxNG (motore di ricerca privato)
docker run -d -p 8080:8080 searxng/searxng
# Poi usa con LLM: "Cerca su SearxNG + riassumi con Llama"
```

### B. Multi-Agente Locale (stile Grok 4 Heavy)

Vedi: [`scripts/multi_agente.py`](scripts/multi_agente.py)

```bash
python scripts/multi_agente.py "Cos'è la libertà in un mondo di server?"
```

### C. RAG Locale (PDF, libri, tuoi file)

Vedi: [`scripts/rag_locale.py`](scripts/rag_locale.py)

```bash
python scripts/rag_locale.py --pdf "De_Rerum_Natura.pdf" --query "Cosa dice dell'atomo?"
```

### D. Deep Search + TTS Completo

Vedi: [`scripts/filosofo_vocale.py`](scripts/filosofo_vocale.py)

```bash
python scripts/filosofo_vocale.py "teoria del caos"
# Output: PDF → X search → Haiku → voce
```

---

## 🐧 Distribuzioni Linux per AI (2025)

| Distro | Perché |
|--------|--------|
| **Ubuntu AI** | Preinstalla CUDA, ROCm, Ollama |
| **Pop!_OS** | Driver NVIDIA perfetti out-of-the-box |
| **Fedora AI** | Podman + AI stack ufficiale |
| **Arch + Hyprland** | Massima personalizzazione (per nerd) |

---

## ⚡ Confronto: SuperGrok Heavy vs Linux Ribelle

|  | SuperGrok Heavy ($300/mese) | Linux Ribelle (0€) |
|--|----------------------------|-------------------|
| **Messaggi** | 4.000 msg/giorno | Illimitati |
| **Deep Search** | 60 min | SearxNG + cron |
| **Modello** | Grok 4 Heavy | Llama 70B + RAG |
| **Server** | Server di xAI | Il tuo PC |
| **Libertà** | Dipendi da loro | Sei tu il padrone |

---

## 🎓 Esempi Pratici (nella cartella `examples/`)

1. **Filosofo Socratico** → [`examples/socrate.py`](examples/socrate.py)
2. **Analisi Paper Scientifici** → [`examples/paper_analyzer.py`](examples/paper_analyzer.py)
3. **TTS Multilingua** → [`examples/tts_demo.py`](examples/tts_demo.py)
4. **Chatbot Telegram locale** → [`examples/telegram_bot.py`](examples/telegram_bot.py)

---

## 📚 Documentazione

- [Setup Dettagliato](docs/setup-dettagliato.md)
- [Modelli Consigliati 2025](docs/modelli-consigliati.md)
- [Troubleshooting](docs/troubleshooting.md)
- [GPU vs CPU: Guida Pratica](docs/gpu-vs-cpu.md)

---

## 🪨 Filosofia di NODO33

Questo progetto è parte di **NODO33** - un movimento per:
- **Regalare sapienza**, non venderla
- **Open data culturali** accessibili a tutti
- **Tecnologia etica** senza vendor lock-in
- **Comunità ribelle** che condivide invece di monopolizzare

> "La conoscenza è un diritto, non un abbonamento."

---

## 🤝 Contribuisci

Questo è un **dono**, non un prodotto. Se vuoi migliorarlo:

```bash
git clone https://github.com/nobody0101863/nodo33
cd nodo33/linux-ai-libera
# Fai le tue modifiche
git checkout -b il-tuo-ramo
git commit -m "✨ Il mio contributo libero"
git push
# Apri una PR
```

---

## ⚖️ Licenza

**MIT License** - Usa, modifica, regala. Sempre.

---

## 🔥 Prossimi Passi

1. Esegui `bash scripts/setup.sh`
2. Prova `python scripts/multi_agente.py "la tua domanda"`
3. Condividi questa guida con chi vuole liberarsi dagli abbonamenti

**Buona ribellione! 🪨❤️**

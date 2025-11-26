# 🔧 Troubleshooting - Linux AI Libera

Soluzioni ai problemi più comuni.

## 🔴 Problemi Comuni

### 1. "Ollama connection refused"

**Sintomo:**
```
Error: Failed to connect to localhost:11434
```

**Soluzione:**
```bash
# Verifica se Ollama è attivo
curl http://localhost:11434

# Se non risponde, avvia Ollama
ollama serve

# In un altro terminale, testa
ollama list
```

**Fix permanente:**
```bash
# Crea servizio systemd (Ubuntu/Debian)
sudo systemctl enable ollama
sudo systemctl start ollama
```

---

### 2. "Model not found"

**Sintomo:**
```
Error: model 'llama3.1:70b' not found
```

**Soluzione:**
```bash
# Lista modelli installati
ollama list

# Scarica il modello mancante
ollama pull llama3.1:70b

# Modelli leggeri alternativi
ollama pull llama3.1:8b      # 4.7 GB
ollama pull mistral:7b       # 4.1 GB
ollama pull qwen2.5:7b       # 4.7 GB
```

---

### 3. "Out of memory / CUDA out of memory"

**Sintomo:**
```
RuntimeError: CUDA out of memory
```

**Soluzione:**

**Opzione A - Usa modelli più piccoli:**
```bash
# Invece di 70B, usa 8B o 13B
ollama pull llama3.1:8b
```

**Opzione B - Usa quantizzazione:**
```bash
# Modelli quantizzati (più leggeri)
ollama pull llama3.1:8b-q4_0  # 4-bit quantization
```

**Opzione C - CPU mode:**
```bash
# Forza CPU (più lento ma funziona sempre)
CUDA_VISIBLE_DEVICES="" ollama run llama3.1:8b
```

**Tabella VRAM necessaria:**

| Modello | VRAM minima | Consigliata |
|---------|------------|-------------|
| 7B-8B | 4 GB | 8 GB |
| 13B | 8 GB | 16 GB |
| 33B | 20 GB | 32 GB |
| 70B | 40 GB | 48 GB |

---

### 4. "ModuleNotFoundError"

**Sintomo:**
```
ModuleNotFoundError: No module named 'crewai'
```

**Soluzione:**
```bash
# Verifica ambiente virtuale attivo
source venv/bin/activate

# Reinstalla dipendenze
pip install -r requirements.txt

# Se persiste, reinstalla da zero
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 5. SearxNG non funziona

**Sintomo:**
```
Connection refused to localhost:8080
```

**Soluzione:**
```bash
# Verifica Docker
docker ps | grep searxng

# Se non attivo, avvia
docker run -d \
  -p 8080:8080 \
  --name searxng \
  searxng/searxng

# Testa
curl http://localhost:8080
```

**Alternativa senza SearxNG:**
Gli script funzionano anche senza SearxNG (usano fallback).

---

### 6. "Permission denied" su script

**Sintomo:**
```bash
bash: ./scripts/setup.sh: Permission denied
```

**Soluzione:**
```bash
# Rendi eseguibili gli script
chmod +x scripts/*.sh
chmod +x scripts/*.py
chmod +x examples/*.py

# Oppure esegui con bash/python
bash scripts/setup.sh
python scripts/multi_agente.py
```

---

### 7. Audio non funziona (TTS)

**Sintomo:**
```
No module named 'gtts'
```

**Soluzione:**
```bash
# Installa gTTS
pip install gTTS

# Per ascoltare audio
sudo apt install mpg123

# Test
python -c "from gtts import gTTS; gTTS('test', lang='it').save('test.mp3')"
mpg123 test.mp3
```

---

### 8. Ollama lentissimo (CPU)

**Sintomo:**
Ollama funziona ma impiega minuti per rispondere.

**Causa:** Sta usando CPU invece di GPU.

**Soluzione:**
```bash
# Verifica GPU
nvidia-smi

# Se non vedi la GPU:
# 1. Installa driver NVIDIA
sudo ubuntu-drivers install

# 2. Installa CUDA
sudo apt install nvidia-cuda-toolkit

# 3. Riavvia Ollama
pkill ollama
ollama serve
```

---

### 9. "Rate limit exceeded" (non dovrebbe succedere!)

Se vedi questo errore, stai usando API esterne invece di Ollama locale.

**Verifica:**
```python
# Nel codice, assicurati di usare Ollama locale
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434"  # LOCALE!
)
```

**NON usare:**
```python
# ❌ SBAGLIATO - usa API OpenAI (a pagamento)
from langchain.llms import OpenAI
llm = OpenAI(api_key="...")
```

---

### 10. Python "command not found"

**Sintomo:**
```bash
python: command not found
```

**Soluzione:**
```bash
# Su Linux è python3, non python
python3 --version

# Crea alias (opzionale)
echo "alias python=python3" >> ~/.bashrc
source ~/.bashrc
```

---

## 🆘 Problemi Ancora Irrisolti?

### 1. Controlla i log
```bash
# Log Ollama
journalctl -u ollama -f

# Log Docker (SearxNG)
docker logs searxng
```

### 2. Chiedi alla comunità

Apri una issue su GitHub:
```
https://github.com/nobody0101863/nodo33/issues
```

Includi:
- Sistema operativo e versione
- Output del comando che fallisce
- File `requirements.txt` che stai usando

### 3. Debug manuale

Test base:
```bash
# Test Ollama
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Test"
}'

# Test Python
python3 -c "from langchain_community.llms import Ollama; print('OK')"

# Test GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 📚 Risorse Utili

- [Ollama Docs](https://ollama.com/docs)
- [LangChain Docs](https://python.langchain.com/)
- [CrewAI Docs](https://docs.crewai.com/)

---

🪨❤️ **"L'errore è un maestro migliore del successo"**

Se hai risolto un problema non documentato qui, contribuisci! Fai una PR.

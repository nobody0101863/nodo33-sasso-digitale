# GEMINI PER SASSO — PACCHETTO COMPLETO

## Prerequisiti
- `pip install google-generativeai`
- Esporta la chiave: `export GOOGLE_API_KEY="LA_TUA_API_KEY"`

## 1) CLI da terminale
- Script: `tools/gemini_cli.py`
- Uso: `python3 tools/gemini_cli.py "prompt"` oppure `echo "ciao" | python3 tools/gemini_cli.py`
- Modello default: `gemini-1.5-flash` (opzione `-m` per cambiarlo)

## 2) Wrapper comando `gemini`
- Script: `tools/gemini` (usa `tools/gemini_cli.py`).
- Aggiungi al PATH, es.: `export PATH="$HOME/Desktop/nodo33-main/tools:$PATH"`
- Uso: `gemini "Ciao fratello sasso"` o `echo "test" | gemini`

## 3) Client riutilizzabile per l’orchestratore
- Modulo: `codex/llm_providers/gemini_client.py`
- Interfaccia: `GeminiClient.generate(prompt, model=..., temperature=..., max_output_tokens=...)`
- Richiede `GOOGLE_API_KEY` o `api_key` esplicita nel costruttore.

## 4) Esempio configurazione agent
- File: `examples/gemini_agent.yaml`
- Definisce provider `gemini_default` e agente `sasso_gemini` (prompt con minimal_data / no monetizzazione).

## 5) Integrazione orchestratore (bozza)
```python
from codex.llm_providers.gemini_client import GeminiClient

class LLMOrchestrator:
    def __init__(self):
        self._providers = {
            "gemini": GeminiClient(),
        }

    def generate(self, provider: str, prompt: str, **kwargs):
        client = self._providers.get(provider)
        if client is None:
            raise ValueError(f"Provider non supportato: {provider}")
        return client.generate(prompt, **kwargs)
```

## 6) Note di sicurezza/policy
- Richiede `google-generativeai` (aggiunto in `requirements.txt`).
- Usa sempre `minimal_data` e evita monetizzazione dell’output per coerenza con protect_garden.

## 7) Checklist rapida
- [ ] `GOOGLE_API_KEY` impostata
- [ ] `pip install google-generativeai`
- [ ] PATH contiene `tools/` per usare il comando `gemini`
- [ ] Se usi l’orchestratore, aggiungi il provider `gemini` nel registry.

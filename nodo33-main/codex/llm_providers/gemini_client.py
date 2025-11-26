"""
GeminiClient - wrapper minimale per usare Google Gemini dentro l'orchestratore Codex/Nodo33.
Richiede variabile d'ambiente GOOGLE_API_KEY o parametro api_key.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover - dipendenza opzionale
    raise ImportError(
        "google-generativeai non installato. Esegui: pip install google-generativeai"
    ) from exc


class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY non impostata. "
                "Imposta la variabile d'ambiente o passa api_key esplicitamente."
            )

        genai.configure(api_key=self.api_key)
        self.default_model = default_model
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Esegue una generazione con Gemini e restituisce un dizionario standardizzato.
        """
        model_name = model or self.default_model
        gen_model = genai.GenerativeModel(model_name)

        generation_config = {
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_output_tokens": kwargs.pop("max_output_tokens", 2048),
        }

        response = gen_model.generate_content(
            prompt,
            generation_config=generation_config,
            **kwargs,
        )

        output_text = getattr(response, "text", "")

        return {
            "provider": "gemini",
            "model": model_name,
            "prompt": prompt,
            "output_text": output_text,
            "raw_response": response.to_dict() if hasattr(response, "to_dict") else str(response),
        }

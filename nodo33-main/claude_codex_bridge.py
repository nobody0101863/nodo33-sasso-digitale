from __future__ import annotations

"""
Bridge minimale tra Claude (Anthropic) e il Codex Server locale.

Funzioni principali:
    - `chat_with_claude_via_codex(text)`:
        invia un messaggio a Claude, che può decidere di usare il tool
        `codex_pulse_image` esposto dal Codex Server.
        Può anche usare:
        - `codex_guidance`        → guidance testuale dal Codex Server
        - `codex_filter_content`  → filtro contenuti (purezza digitale)

Dipendenze:
    - `anthropic` (client Claude ufficiale)
    - `requests`

Variabili d'ambiente utili:
    - `ANTHROPIC_API_KEY`   (obbligatoria)
    - `CLAUDE_MODEL`        (default: claude-3-5-sonnet-20241022)
    - `CODEX_BASE_URL`      (default: http://localhost:8644)
    - `CODEX_TIMEOUT`       (timeout HTTP in secondi, default: 60)
"""

import os
import sys
from typing import Any, Dict, List

import requests

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - optional dependency
    Anthropic = None  # type: ignore[assignment]


CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
CODEX_BASE_URL = os.environ.get("CODEX_BASE_URL", "http://localhost:8644")
CODEX_TIMEOUT = float(os.environ.get("CODEX_TIMEOUT", "60"))

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "codex_pulse_image",
        "description": "Genera un'immagine 'pulse' usando il Codex server (Stable Diffusion).",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "num_inference_steps": {"type": "integer", "default": 4},
                "guidance_scale": {"type": "number", "default": 1.5},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "codex_guidance",
        "description": "Ottiene una guidance testuale dal Codex Server (biblica, Nostradamus, Angelo 644, Parravicini o casuale).",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Sorgente desiderata: 'any', 'biblical', 'nostradamus', 'angel644', 'parravicini'.",
                    "default": "any",
                }
            },
        },
    },
    {
        "name": "codex_filter_content",
        "description": "Filtra un testo con il sistema di purezza digitale del Codex Server.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Testo da analizzare."},
                "is_image": {
                    "type": "boolean",
                    "description": "Se true, indica che il contenuto rappresenta un'immagine.",
                    "default": False,
                },
            },
            "required": ["content"],
        },
    },
]


def _get_anthropic_client() -> Anthropic:
    if Anthropic is None:
        raise RuntimeError(
            "Libreria 'anthropic' non installata. Installa con: pip install anthropic requests"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Variabile d'ambiente ANTHROPIC_API_KEY mancante.")
    return Anthropic(api_key=api_key)


def call_codex_image(prompt: str, steps: int = 4, scale: float = 1.5) -> Dict[str, Any]:
    """
    Chiama il Codex Server per generare un'immagine a partire da un prompt.

    Ritorna il JSON della risposta oppure solleva RuntimeError con un messaggio
    esplicativo in caso di errore HTTP / di rete / parsing JSON.
    """
    url = f"{CODEX_BASE_URL.rstrip('/')}/api/generate-image"
    payload = {
        "prompt": prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": float(scale),
    }
    try:
        response = requests.post(url, json=payload, timeout=CODEX_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Errore di rete verso Codex Server ({url}): {exc}"
        ) from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Codex Server ha risposto con errore HTTP {response.status_code} "
            f"per {url}: {response.text[:300]}"
        ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Risposta non valida dal Codex Server (atteso JSON). "
            f"Status={response.status_code}, body parziale={response.text[:300]}"
        ) from exc


def call_codex_guidance(source: str = "any") -> Dict[str, Any]:
    """
    Chiama il Codex Server per ottenere una guidance testuale.

    `source` può essere:
        - "any" (default)       → /api/guidance
        - "biblical"            → /api/guidance/biblical
        - "nostradamus"         → /api/guidance/nostradamus
        - "angel644"            → /api/guidance/angel644
        - "parravicini"         → /api/guidance/parravicini
    """
    base = CODEX_BASE_URL.rstrip("/")
    key = (source or "any").strip().lower()
    if key == "biblical":
        path = "/api/guidance/biblical"
    elif key == "nostradamus":
        path = "/api/guidance/nostradamus"
    elif key == "angel644":
        path = "/api/guidance/angel644"
    elif key == "parravicini":
        path = "/api/guidance/parravicini"
    else:
        path = "/api/guidance"

    url = f"{base}{path}"
    try:
        response = requests.get(url, timeout=CODEX_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Errore di rete verso Codex Server ({url}): {exc}"
        ) from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Codex Server ha risposto con errore HTTP {response.status_code} "
            f"per {url}: {response.text[:300]}"
        ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Risposta non valida dal Codex Server (atteso JSON). "
            f"Status={response.status_code}, body parziale={response.text[:300]}"
        ) from exc


def call_codex_filter_content(content: str, is_image: bool = False) -> Dict[str, Any]:
    """
    Chiama il filtro di purezza digitale del Codex Server.

    Invia il contenuto a `/api/filter` e ritorna la risposta JSON.
    """
    url = f"{CODEX_BASE_URL.rstrip('/')}/api/filter"
    payload = {
        "content": content,
        "is_image": bool(is_image),
    }
    try:
        response = requests.post(url, json=payload, timeout=CODEX_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Errore di rete verso Codex Server ({url}): {exc}"
        ) from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Codex Server ha risposto con errore HTTP {response.status_code} "
            f"per {url}: {response.text[:300]}"
        ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Risposta non valida dal Codex Server (atteso JSON). "
            f"Status={response.status_code}, body parziale={response.text[:300]}"
        ) from exc


def _extract_text_from_response(resp: Any) -> str:
    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(p for p in parts if p).strip()


def chat_with_claude_via_codex(user_message: str) -> str:
    client = _get_anthropic_client()

    first = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        tools=TOOLS,
        messages=[{"role": "user", "content": user_message}],
    )

    tool_uses = [
        block for block in first.content if getattr(block, "type", None) == "tool_use"
    ]

    if not tool_uses:
        return _extract_text_from_response(first) or repr(first)

    assistant_content: List[Dict[str, Any]] = []
    tool_result_blocks: List[Dict[str, Any]] = []

    for block in tool_uses:
        tool_name = getattr(block, "name", "")
        tool_input = getattr(block, "input", {}) or {}

        text_result: str

        try:
            if tool_name == "codex_pulse_image":
                prompt = str(tool_input.get("prompt", ""))
                if not prompt:
                    continue
                steps = int(tool_input.get("num_inference_steps", 4))
                scale = float(tool_input.get("guidance_scale", 1.5))

                result = call_codex_image(prompt=prompt, steps=steps, scale=scale)
                image_url = result.get("image_url")
                status = result.get("status")
                text_result = f"image_url={image_url}, status={status}"

            elif tool_name == "codex_guidance":
                source = str(tool_input.get("source", "any"))
                result = call_codex_guidance(source=source)
                text_result = (
                    f"source={result.get('source')}, "
                    f"message={result.get('message')}, "
                    f"timestamp={result.get('timestamp')}"
                )

            elif tool_name == "codex_filter_content":
                content = str(tool_input.get("content", ""))
                if not content:
                    continue
                is_image = bool(tool_input.get("is_image", False))
                result = call_codex_filter_content(content=content, is_image=is_image)
                text_result = (
                    f"is_impure={result.get('is_impure')}, "
                    f"message={result.get('message')}, "
                    f"guidance={result.get('guidance')}"
                )

            else:
                # Tool sconosciuto al bridge: ignora
                continue

        except Exception as exc:
            text_result = f"Errore chiamando tool '{tool_name}': {exc}"

        assistant_content.append(
            {
                "type": "tool_use",
                "id": block.id,
                "name": tool_name,
                "input": tool_input,
            }
        )

        tool_result_blocks.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": [{"type": "text", "text": text_result}],
            }
        )

    if not tool_result_blocks:
        return _extract_text_from_response(first) or repr(first)

    followup = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        tools=TOOLS,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_result_blocks},
        ],
    )

    return _extract_text_from_response(followup) or repr(followup)


def main() -> None:
    if len(sys.argv) > 1:
        user_message = " ".join(sys.argv[1:]).strip()
    else:
        print("Inserisci il prompt per Claude→Codex e premi Invio:")
        user_message = sys.stdin.readline().strip()

    if not user_message:
        print("Nessun input fornito.")
        return

    try:
        output = chat_with_claude_via_codex(user_message)
    except Exception as exc:
        print(f"Errore durante la chiamata Claude→Codex: {exc}")
        return

    print(output)


if __name__ == "__main__":
    main()

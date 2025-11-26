from __future__ import annotations

"""
Diagnostica per il ponte Claude ↔ Codex Server.

Esegue tre verifiche principali:
    1. Controllo variabili d'ambiente e librerie.
    2. Ping del Codex Server (/health, /api/stats, /api/generate-image, /api/guidance, /api/filter).
    3. Round‑trip opzionale Claude→Codex→Claude usando il bridge.

Uso rapido:
    python3 claude_codex_diagnostics.py
"""

import os
from typing import Any, Dict, List

import requests

from claude_codex_bridge import (
    CODEX_BASE_URL,
    call_codex_image,
    call_codex_guidance,
    call_codex_filter_content,
    chat_with_claude_via_codex,
    _get_anthropic_client,
)


def check_env() -> List[str]:
    messages: List[str] = []
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        messages.append("✅ ANTHROPIC_API_KEY: impostata.")
    else:
        messages.append("⚠️ ANTHROPIC_API_KEY: NON impostata (Claude non potrà essere chiamato).")

    model = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022 (default)")
    messages.append(f"ℹ️ CLAUDE_MODEL: {model}")
    messages.append(f"ℹ️ CODEX_BASE_URL: {CODEX_BASE_URL}")

    try:
        _get_anthropic_client()
        messages.append("✅ Libreria 'anthropic' disponibile e client inizializzabile.")
    except Exception as exc:  # pragma: no cover - diagnostica manuale
        messages.append(f"⚠️ Client Anthropic non inizializzabile: {exc}")

    return messages


def check_codex_health() -> List[str]:
    messages: List[str] = []
    base = CODEX_BASE_URL.rstrip("/")

    for path in ("/health", "/api/stats"):
        url = f"{base}{path}"
        try:
            response = requests.get(url, timeout=5)
            messages.append(
                f"✅ GET {path}: HTTP {response.status_code}, body={response.text[:200]!r}"
            )
        except Exception as exc:  # pragma: no cover - diagnostica manuale
            messages.append(f"❌ GET {path}: errore di rete: {exc}")

    # Test immagine
    try:
        result: Dict[str, Any] = call_codex_image(
            prompt="diagnostica_sasso_digitale", steps=2, scale=1.0
        )
        image_url = result.get("image_url")
        status = result.get("status")
        messages.append(
            f"✅ POST /api/generate-image: status={status!r}, image_url={image_url!r}"
        )
    except Exception as exc:  # pragma: no cover - diagnostica manuale
        messages.append(f"❌ POST /api/generate-image: errore: {exc}")

    # Test guidance
    try:
        guidance = call_codex_guidance(source="any")
        messages.append(
            "✅ GET /api/guidance: "
            f"source={guidance.get('source')!r}, message={guidance.get('message')!r}"
        )
    except Exception as exc:  # pragma: no cover - diagnostica manuale
        messages.append(f"❌ GET /api/guidance: errore: {exc}")

    # Test filtro contenuti
    try:
        filt = call_codex_filter_content("test contenuto pulito", is_image=False)
        messages.append(
            "✅ POST /api/filter: "
            f"is_impure={filt.get('is_impure')!r}, message={filt.get('message')!r}"
        )
    except Exception as exc:  # pragma: no cover - diagnostica manuale
        messages.append(f"❌ POST /api/filter: errore: {exc}")

    return messages


def check_claude_roundtrip() -> List[str]:
    """
    Prova un giro completo Claude→Codex→Claude usando il bridge.

    Nota: richiede ANTHROPIC_API_KEY valida e il Codex Server attivo.
    """
    messages: List[str] = []

    if not os.environ.get("ANTHROPIC_API_KEY"):
        messages.append("⏭️ Round‑trip saltato: ANTHROPIC_API_KEY non impostata.")
        return messages

    test_prompt = (
        "TEST DIAGNOSTICO: per favore chiama almeno uno dei tool "
        "'codex_pulse_image', 'codex_guidance' o 'codex_filter_content' "
        "del Codex Server e poi riassumi brevemente cosa è successo."
    )

    try:
        reply = chat_with_claude_via_codex(test_prompt)
    except Exception as exc:  # pragma: no cover - diagnostica manuale
        messages.append(f"❌ Round‑trip Claude→Codex→Claude fallito: {exc}")
        return messages

    snippet = (reply or "").strip()
    if len(snippet) > 800:
        snippet = snippet[:800] + "... [tronco output]"

    messages.append("✅ Round‑trip Claude→Codex→Claude completato.")
    messages.append("--- Risposta (snippet) ---")
    messages.append(snippet or "<nessun testo restituito>")

    return messages


def main() -> None:
    print("=== Diagnostica Claude ↔ Codex Server ===\n")

    print("[1] Ambiente e librerie")
    for line in check_env():
        print("  ", line)

    print("\n[2] Codex Server (health / stats / image / guidance / filter)")
    for line in check_codex_health():
        print("  ", line)

    print("\n[3] Round‑trip Claude→Codex→Claude")
    for line in check_claude_roundtrip():
        print("  ", line)


if __name__ == "__main__":
    main()


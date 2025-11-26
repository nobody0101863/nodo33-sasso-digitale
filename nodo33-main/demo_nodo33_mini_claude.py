from __future__ import annotations

import sys

from claude_codex_bridge import (
    CLAUDE_MODEL,
    _extract_text_from_response,
    _get_anthropic_client,
)
from src.nodo33_mini_ai import Nodo33MiniAI


def main() -> None:
    if len(sys.argv) > 1:
        user_message = " ".join(sys.argv[1:]).strip()
    else:
        print("Inserisci il messaggio per Claude (Nodo33 Mini IA) e premi Invio:")
        user_message = sys.stdin.readline().strip()

    if not user_message:
        print("Nessun input fornito.")
        return

    profile = Nodo33MiniAI()
    system_prompt = profile.get_system_prompt()

    client = _get_anthropic_client()

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    print(_extract_text_from_response(resp) or repr(resp))


if __name__ == "__main__":
    main()


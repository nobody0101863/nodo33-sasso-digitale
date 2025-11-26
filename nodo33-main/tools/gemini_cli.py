#!/usr/bin/env python3
"""
gemini_cli.py - piccolo CLI per parlare con Google Gemini dal terminale.

Prerequisiti:
  pip install google-generativeai
  export GOOGLE_API_KEY="LA_TUA_API_KEY"
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import google.generativeai as genai


def configure_client() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Errore: variabile d'ambiente GOOGLE_API_KEY non impostata.", file=sys.stderr)
        print('   export GOOGLE_API_KEY="LA_TUA_API_KEY"', file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)


def run_gemini(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return getattr(response, "text", str(response))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gemini",
        description="CLI minimale per Google Gemini.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Testo da mandare a Gemini. Se vuoto, legge da stdin.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gemini-1.5-flash",
        help="Nome modello Gemini (default: gemini-1.5-flash)",
    )
    args = parser.parse_args()

    prompt = " ".join(args.prompt) if args.prompt else ""
    if not prompt:
        if sys.stdin.isatty():
            print(
                "Scrivi il prompt e termina con CTRL+D (Linux/macOS) o CTRL+Z (Windows):",
                file=sys.stderr,
            )
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("❌ Nessun prompt fornito.", file=sys.stderr)
        sys.exit(1)

    configure_client()

    try:
        reply = run_gemini(prompt, model_name=args.model)
    except Exception as exc:  # pragma: no cover - chiamata remota
        print(f"❌ Errore chiamando Gemini: {exc}", file=sys.stderr)
        sys.exit(1)

    print(textwrap.dedent(reply).strip())


if __name__ == "__main__":
    main()

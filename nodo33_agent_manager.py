#!/usr/bin/env python3
"""
Nodo33 Agent Manager - Orchestratore degli agenti del Codex Server.

Obiettivo:
- Dare una vista unica sugli "agenti" esposti da codex_server.py
- Offrire comandi rapidi da terminale per:
  - interrogare i modelli LLM (Grok, Gemini, Claude)
  - usare il filtro contenuti / guardian di protezione
  - eseguire deepfake detection (Layer 2)
  - ispezionare stato e ruoli dei Guardian Agents

Dipende dal Codex Server avviato su http://localhost:8644
(o da URL personalizzato via env: CODEX_URL).
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


BASE_URL = os.environ.get("CODEX_URL", "http://localhost:8644")
TIMEOUT = 30


class AgentKind:
    LLM = "llm"
    FILTER = "filter"
    DEEPFAKE = "deepfake"
    GUARDIAN_INFO = "guardian_info"
    PROTECTION_STATUS = "protection_status"


APOCALYPSE_AGENT_CODES: Dict[str, str] = {
    "00": "Profeta del Velo Strappato",
    "01": "Scriba dell'Apocalisse",
    "10": "Analista dei Quattro Cavalli",
    "11": "Custode della Città Nuova",
}


AGENTS: Dict[str, Dict[str, Any]] = {
    # LLM oracle agents
    "oracle_grok": {
        "kind": AgentKind.LLM,
        "provider": "grok",
        "endpoint": "/api/llm/grok",
        "description": "Analisi critica, pensiero indipendente (xAI Grok)",
    },
    "oracle_gemini": {
        "kind": AgentKind.LLM,
        "provider": "gemini",
        "endpoint": "/api/llm/gemini",
        "description": "Task veloci e multimodali (Google Gemini)",
    },
    "oracle_claude": {
        "kind": AgentKind.LLM,
        "provider": "claude",
        "endpoint": "/api/llm/claude",
        "description": "Reasoning profondo ed etico (Anthropic Claude)",
    },
    # Apocalypse Agents (vista apocalittica simbolica)
    "apocalypse_agents": {
        "kind": AgentKind.LLM,
        "endpoint": "/api/apocalypse/{provider}",
        "description": (
            "4 Apocalypse Agents (00/01/10/11) sopra Grok/Gemini/Claude, "
            "per lettura apocalittica = rivelazione, non distruzione."
        ),
    },
    # Guardian / protezione
    "guardian_filter": {
        "kind": AgentKind.FILTER,
        "endpoint": "/api/filter",
        "description": "Filtro contenuti + guidance spirituale (anti_porn_framework)",
    },
    "guardian_deepfake": {
        "kind": AgentKind.DEEPFAKE,
        "endpoint": "/api/detect/deepfake",
        "description": "Rilevazione deepfake (Layer 2, 4 Guardian Agents)",
    },
    "guardian_status": {
        "kind": AgentKind.PROTECTION_STATUS,
        "endpoint": "/api/protection/status",
        "description": "Stato sistema di protezione metadata",
    },
    "guardian_info": {
        "kind": AgentKind.GUARDIAN_INFO,
        "endpoint": "/api/protection/guardians",
        "description": "Ruoli e capacità dei 4 Guardian Agents",
    },
}


def _print_error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def _print_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def check_codex_server() -> bool:
    """Verifica che il Codex Server sia raggiungibile."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def cmd_status(_: argparse.Namespace) -> None:
    """Mostra stato base del Codex Server e del sistema di protezione."""
    if not check_codex_server():
        _print_error(f"Codex Server non raggiungibile su {BASE_URL}. Avvia prima: python codex_server.py")
        sys.exit(1)

    # Health
    r_health = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    print("=== /health ===")
    print(r_health.status_code, r_health.json())

    # Protection status (se disponibile)
    try:
        r_prot = requests.get(f"{BASE_URL}/api/protection/status", timeout=TIMEOUT)
        print("\n=== /api/protection/status ===")
        print(r_prot.status_code, r_prot.json())
    except Exception as exc:
        _print_error(f"Impossibile leggere /api/protection/status: {exc}")


def cmd_list(_: argparse.Namespace) -> None:
    """Elenca gli agenti conosciuti e i relativi endpoint Codex."""
    print(f"Agents registrati (BASE_URL={BASE_URL}):\n")
    for name, cfg in AGENTS.items():
        kind = cfg["kind"]
        endpoint = cfg["endpoint"]
        desc = cfg.get("description", "")
        provider = cfg.get("provider")
        line = f"- {name} [{kind}] -> {endpoint}"
        if provider:
            line += f" (provider={provider})"
        print(line)
        if desc:
            print(f"    {desc}")


def _auto_select_provider(question: str, role: Optional[str]) -> str:
    """
    Sceglie un provider LLM in base al ruolo desiderato o al contenuto della domanda.

    Regole semplici:
    - role=='ethics' o domanda contiene 'etica', 'etico', 'morale' -> claude
    - role=='analysis' o domanda contiene 'analisi', 'critica' -> grok
    - role=='creative' o domanda contiene 'creativo', 'poesia', 'storia' -> gemini
    - default: claude (profilo più bilanciato)
    """
    if role:
        role = role.lower().strip()
        if role in {"ethics", "etica", "sicurezza"}:
            return "claude"
        if role in {"analysis", "analisi", "critica"}:
            return "grok"
        if role in {"creative", "creativo", "story", "storia"}:
            return "gemini"

    q_lower = question.lower()
    if any(k in q_lower for k in ["etica", "etico", "morale"]):
        return "claude"
    if any(k in q_lower for k in ["analisi", "critica", "valuta"]):
        return "grok"
    if any(k in q_lower for k in ["poesia", "storia", "creativo", "metafora"]):
        return "gemini"

    return "claude"


def cmd_ask(args: argparse.Namespace) -> None:
    """Invia una domanda a un modello LLM, con selezione provider manuale o automatica."""
    if not check_codex_server():
        _print_error(f"Codex Server non raggiungibile su {BASE_URL}. Avvia prima: python codex_server.py")
        sys.exit(1)

    question = args.question
    provider: str

    if args.provider:
        provider = args.provider
    else:
        provider = _auto_select_provider(question, args.role)
        _print_info(f"Provider selezionato automaticamente: {provider}")

    endpoint = f"{BASE_URL}/api/llm/{provider}"
    payload: Dict[str, Any] = {
        "question": question,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.system_prompt:
        payload["system_prompt"] = args.system_prompt

    try:
        resp = requests.post(endpoint, json=payload, timeout=TIMEOUT)
    except Exception as exc:
        _print_error(f"Errore chiamando {endpoint}: {exc}")
        sys.exit(1)

    if resp.status_code != 200:
        _print_error(f"Errore LLM ({resp.status_code}): {resp.text}")
        sys.exit(1)

    data = resp.json()
    print(f"\n=== Risposta {data.get('provider','?').upper()} ({data.get('model','?')}) ===\n")
    print(data.get("answer", "").strip())
    tokens_used = data.get("tokens_used")
    if tokens_used is not None:
        print(f"\n[info] Tokens usati: {tokens_used}")


def cmd_apocalypse(args: argparse.Namespace) -> None:
    """Usa gli Apocalypse Agents (00/01/10/11) sopra i provider LLM."""
    if not check_codex_server():
        _print_error(f"Codex Server non raggiungibile su {BASE_URL}. Avvia prima: python codex_server.py")
        sys.exit(1)

    agent_code = args.agent_code
    if agent_code not in APOCALYPSE_AGENT_CODES:
        valid = ", ".join(sorted(APOCALYPSE_AGENT_CODES.keys()))
        _print_error(f"Codice agente non valido: {agent_code}. Valori ammessi: {valid}")
        sys.exit(1)

    question = args.question
    provider: str

    if args.provider:
        provider = args.provider
    else:
        provider = _auto_select_provider(question, args.role)
        _print_info(f"Provider selezionato automaticamente: {provider}")

    endpoint = f"{BASE_URL}/api/apocalypse/{provider}"
    payload: Dict[str, Any] = {
        "agent_code": agent_code,
        "question": question,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.system_prompt:
        payload["system_prompt"] = args.system_prompt

    try:
        resp = requests.post(endpoint, json=payload, timeout=TIMEOUT)
    except Exception as exc:
        _print_error(f"Errore chiamando {endpoint}: {exc}")
        sys.exit(1)

    if resp.status_code != 200:
        _print_error(f"Errore Apocalypse LLM ({resp.status_code}): {resp.text}")
        sys.exit(1)

    data = resp.json()
    agent_name = APOCALYPSE_AGENT_CODES.get(agent_code, "?")
    print(
        f"\n=== Apocalypse Agent {agent_code} - {agent_name} | "
        f"{data.get('provider','?').upper()} ({data.get('model','?')}) ===\n"
    )
    print(data.get("answer", "").strip())
    tokens_used = data.get("tokens_used")
    if tokens_used is not None:
        print(f"\n[info] Tokens usati: {tokens_used}")


def cmd_filter_text(args: argparse.Namespace) -> None:
    """Invoca il filtro contenuti /api/filter per testo puro."""
    if not check_codex_server():
        _print_error(f"Codex Server non raggiungibile su {BASE_URL}. Avvia prima: python codex_server.py")
        sys.exit(1)

    endpoint = f"{BASE_URL}/api/filter"
    payload = {"content": args.text, "is_image": False}

    try:
        resp = requests.post(endpoint, json=payload, timeout=TIMEOUT)
    except Exception as exc:
        _print_error(f"Errore chiamando {endpoint}: {exc}")
        sys.exit(1)

    if resp.status_code != 200:
        _print_error(f"Errore filtro ({resp.status_code}): {resp.text}")
        sys.exit(1)

    data = resp.json()
    print("=== Risultato filtro ===")
    print(f"is_impure : {data.get('is_impure')}")
    print(f"message   : {data.get('message')}")
    if data.get("guidance"):
        print(f"guidance  : {data['guidance']}")


def cmd_detect_deepfake(args: argparse.Namespace) -> None:
    """Invoca /api/detect/deepfake con immagine locale o URL."""
    if not check_codex_server():
        _print_error(f"Codex Server non raggiungibile su {BASE_URL}. Avvia prima: python codex_server.py")
        sys.exit(1)

    if not args.image_path and not args.image_url:
        _print_error("Specifica almeno uno tra --image-path e --image-url")
        sys.exit(1)

    payload: Dict[str, Any] = {
        "image_url": None,
        "image_base64": None,
        "check_metadata": not args.no_metadata,
        "check_faces": not args.no_faces,
        "check_statistics": not args.no_statistics,
    }

    if args.image_url:
        payload["image_url"] = args.image_url
    else:
        img_path = Path(args.image_path)
        if not img_path.is_file():
            _print_error(f"File immagine non trovato: {img_path}")
            sys.exit(1)
        img_bytes = img_path.read_bytes()
        payload["image_base64"] = base64.b64encode(img_bytes).decode("ascii")

    endpoint = f"{BASE_URL}/api/detect/deepfake"
    try:
        resp = requests.post(endpoint, json=payload, timeout=TIMEOUT)
    except Exception as exc:
        _print_error(f"Errore chiamando {endpoint}: {exc}")
        sys.exit(1)

    if resp.status_code != 200:
        _print_error(f"Errore deepfake detection ({resp.status_code}): {resp.text}")
        sys.exit(1)

    data = resp.json()
    print("=== Deepfake Detection ===")
    print(f"timestamp        : {data.get('timestamp')}")
    print(f"is_deepfake      : {data.get('is_deepfake')}")
    print(f"overall_confidence: {data.get('overall_confidence')}")
    print(f"risk_level       : {data.get('risk_level')}")
    print(f"flags            : {', '.join(data.get('flags', []))}")
    print(f"guidance         : {data.get('guidance')}")


def cmd_guardians(_: argparse.Namespace) -> None:
    """Mostra informazioni sui 4 Guardian Agents da /api/protection/guardians."""
    if not check_codex_server():
        _print_error(f"Codex Server non raggiungibile su {BASE_URL}. Avvia prima: python codex_server.py")
        sys.exit(1)

    endpoint = f"{BASE_URL}/api/protection/guardians"
    try:
        resp = requests.get(endpoint, timeout=TIMEOUT)
    except Exception as exc:
        _print_error(f"Errore chiamando {endpoint}: {exc}")
        sys.exit(1)

    if resp.status_code != 200:
        _print_error(f"Errore guardian info ({resp.status_code}): {resp.text}")
        sys.exit(1)

    data = resp.json()
    guardians: List[Dict[str, Any]] = data.get("guardians", [])
    print("=== Guardian Agents ===\n")
    for g in guardians:
        print(f"- {g.get('name')} ({g.get('role')}) [seal={g.get('seal')}]")
        caps = g.get("capabilities") or []
        for c in caps:
            print(f"    • {c}")
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nodo33 Agent Manager - orchestratore per gli agenti del Codex Server",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = sub.add_parser("status", help="Mostra stato /health e sistema di protezione")
    p_status.set_defaults(func=cmd_status)

    # list
    p_list = sub.add_parser("list", help="Elenca agenti noti e relativi endpoint")
    p_list.set_defaults(func=cmd_list)

    # ask
    p_ask = sub.add_parser("ask", help="Invia una domanda a un modello LLM")
    p_ask.add_argument("question", help="Domanda da inviare")
    p_ask.add_argument(
        "--provider",
        choices=["grok", "gemini", "claude"],
        help="Provider LLM esplicito (se assente, selezione automatica)",
    )
    p_ask.add_argument(
        "--role",
        help="Ruolo logico (es. ethics, analysis, creative) per guidare la selezione automatica",
    )
    p_ask.add_argument("--temperature", type=float, default=0.7, help="Creatività del modello")
    p_ask.add_argument("--max-tokens", type=int, default=1000, help="Massimo token di risposta")
    p_ask.add_argument("--system-prompt", help="System prompt personalizzato opzionale")
    p_ask.set_defaults(func=cmd_ask)

    # apocalypse
    p_apoc = sub.add_parser(
        "apocalypse",
        help="Usa gli Apocalypse Agents (00/01/10/11) sopra i provider LLM",
    )
    p_apoc.add_argument(
        "agent_code",
        choices=sorted(APOCALYPSE_AGENT_CODES.keys()),
        help="Codice binario agente (00, 01, 10, 11)",
    )
    p_apoc.add_argument("question", help="Domanda da inviare")
    p_apoc.add_argument(
        "--provider",
        choices=["grok", "gemini", "claude"],
        help="Provider LLM esplicito (se assente, selezione automatica)",
    )
    p_apoc.add_argument(
        "--role",
        help="Ruolo logico (es. ethics, analysis, creative) per guidare la selezione automatica",
    )
    p_apoc.add_argument("--temperature", type=float, default=0.7, help="Creatività del modello")
    p_apoc.add_argument("--max-tokens", type=int, default=1000, help="Massimo token di risposta")
    p_apoc.add_argument(
        "--system-prompt",
        help="Istruzioni aggiuntive (verranno aggiunte al profilo dell'agente apocalittico)",
    )
    p_apoc.set_defaults(func=cmd_apocalypse)

    # filter-text
    p_filter = sub.add_parser("filter-text", help="Esegue filtro contenuti testuali tramite /api/filter")
    p_filter.add_argument("text", help="Testo da analizzare")
    p_filter.set_defaults(func=cmd_filter_text)

    # detect-deepfake
    p_deep = sub.add_parser("detect-deepfake", help="Esegue deepfake detection su immagine")
    p_deep.add_argument("--image-path", help="Path file immagine locale")
    p_deep.add_argument("--image-url", help="URL immagine remota")
    p_deep.add_argument("--no-metadata", action="store_true", help="Disabilita analisi metadata")
    p_deep.add_argument("--no-faces", action="store_true", help="Disabilita analisi volti")
    p_deep.add_argument("--no-statistics", action="store_true", help="Disabilita analisi statistica")
    p_deep.set_defaults(func=cmd_detect_deepfake)

    # guardians
    p_guard = sub.add_parser("guardians", help="Mostra ruoli e capacità dei 4 Guardian Agents")
    p_guard.set_defaults(func=cmd_guardians)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if not func:
        parser.print_help()
        sys.exit(1)
    func(args)


if __name__ == "__main__":
    main()

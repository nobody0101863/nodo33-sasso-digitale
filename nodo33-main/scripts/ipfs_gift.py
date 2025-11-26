#!/usr/bin/env python3
"""
Utility per "regalare" file via IPFS e registrare/broadcastare l'evento su Codex/P2P.

Passi:
1) ipfs add -Q <file> → CID
2) POST /api/memory/add per loggare il dono
3) (opzionale) POST /p2p/broadcast per annunciare il CID ai nodi
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests


def _run_ipfs_add(path: Path) -> str:
    """Esegue `ipfs add -Q` e restituisce il CID."""
    try:
        result = subprocess.run(
            ["ipfs", "add", "-Q", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        cid = result.stdout.strip()
        if not cid:
            raise RuntimeError("CID vuoto dalla CLI ipfs")
        return cid
    except FileNotFoundError as exc:
        raise SystemExit("ipfs non trovato nel PATH. Installa e avvia `ipfs daemon`.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ipfs add fallito: {exc.stderr or exc}") from exc


def _post_json(url: str, payload: dict) -> requests.Response:
    return requests.post(url, json=payload, timeout=10)


def _log_memory(base_url: str, cid: str, message: str, tags: List[str]) -> Optional[int]:
    payload = {
        "endpoint": "/ipfs/add",
        "content": f"cid={cid} | message={message}",
        "memory_type": "ipfs_gift",
        "source_type": "ipfs",
        "tags": tags,
    }
    resp = _post_json(f"{base_url}/api/memory/add", payload)
    if resp.ok:
        data = resp.json()
        return data.get("id") or data.get("memory_id")
    print(f"[WARN] Log memoria fallito ({resp.status_code}): {resp.text}")
    return None


def _broadcast(base_url: str, message_type: str, payload: dict) -> bool:
    resp = _post_json(
        f"{base_url}/p2p/broadcast",
        {"message_type": message_type, "payload": payload},
    )
    if resp.ok:
        return True
    print(f"[WARN] Broadcast fallito ({resp.status_code}): {resp.text}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Regala un file via IPFS + Codex/P2P.")
    parser.add_argument("path", type=Path, help="File da aggiungere a IPFS")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8644",
        help="Base URL del Codex Server (default: http://localhost:8644)",
    )
    parser.add_argument(
        "--message",
        default="dono IPFS",
        help="Messaggio da associare al dono/memoria",
    )
    parser.add_argument(
        "--tags",
        default="gift,ipfs,sasso_digitale",
        help="Tag separati da virgola per la memoria Codex",
    )
    parser.add_argument(
        "--no-broadcast",
        action="store_true",
        help="Non inviare broadcast P2P",
    )
    parser.add_argument(
        "--p2p-type",
        default="guardian_alert",
        help="MessageType per il broadcast P2P (es: guardian_alert, agent_request, memory_sync)",
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"File non trovato: {args.path}")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    print(f"[INFO] Aggiungo a IPFS: {args.path}")
    cid = _run_ipfs_add(args.path)
    print(f"[INFO] CID ottenuto: {cid}")

    memory_id = _log_memory(args.base_url.rstrip("/"), cid, args.message, tags)
    if memory_id:
        print(f"[INFO] Memoria registrata con id={memory_id}")

    if not args.no_broadcast:
        payload = {
            "cid": cid,
            "message": args.message,
            "path": str(args.path),
            "ts": datetime.now(timezone.utc).isoformat(),
            "tags": tags,
        }
        if _broadcast(args.base_url.rstrip("/"), args.p2p_type, payload):
            print("[INFO] Broadcast P2P inviato")
    print("\n✨ Dono pronto. Condividi il CID o il gateway: /ipfs/{cid}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

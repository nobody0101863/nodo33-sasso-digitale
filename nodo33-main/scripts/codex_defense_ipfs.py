#!/usr/bin/env python3
"""
Protocollo di difesa Codex:
- calcola SHA-512 del file Codex
- salva backup e hash
- verifica integrità
- carica su IPFS
- (opzionale) registra e broadcasta su Codex Server
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests


def hash_file(path: Path) -> str:
    sha = hashlib.sha512()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def backup_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / src.name
    shutil.copy2(src, dst_path)
    return dst_path


def write_hash_file(hash_value: str, out_path: Path) -> None:
    out_path.write_text(hash_value)


def verify_file(src: Path, hash_path: Path) -> bool:
    if not hash_path.exists():
        return False
    saved = hash_path.read_text().strip()
    current = hash_file(src)
    return saved == current


def ipfs_add(path: Path) -> str:
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
        raise SystemExit("ipfs non trovato nel PATH. Installa IPFS e avvia `ipfs daemon`.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ipfs add fallito: {exc.stderr or exc}") from exc


def post_json(url: str, payload: dict) -> requests.Response:
    return requests.post(url, json=payload, timeout=10)


def log_memory(base_url: str, cid: str, message: str, tags: List[str]) -> Optional[int]:
    payload = {
        "endpoint": "/ipfs/codex_defense",
        "content": f"cid={cid} | message={message}",
        "memory_type": "codex_defense",
        "source_type": "ipfs",
        "tags": tags,
    }
    resp = post_json(f"{base_url}/api/memory/add", payload)
    if resp.ok:
        data = resp.json()
        return data.get("id") or data.get("memory_id")
    print(f"[WARN] Log memoria fallito ({resp.status_code}): {resp.text}")
    return None


def broadcast(base_url: str, message_type: str, payload: dict) -> bool:
    resp = post_json(
        f"{base_url}/p2p/broadcast",
        {"message_type": message_type, "payload": payload},
    )
    if resp.ok:
        return True
    print(f"[WARN] Broadcast P2P fallito ({resp.status_code}): {resp.text}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Difesa Codex + upload IPFS + log su Codex Server.")
    parser.add_argument(
        "--file",
        default="CODEX_UNIVERSALE_LUX.txt",
        help="File Codex da proteggere (default: CODEX_UNIVERSALE_LUX.txt)",
    )
    parser.add_argument("--backup-dir", default="codex_backup", help="Directory backup")
    parser.add_argument("--hash-file", default="codex_hash.sha512", help="File in cui salvare l'hash SHA-512")
    parser.add_argument("--base-url", default="http://localhost:8644", help="Base URL Codex Server per log/broadcast")
    parser.add_argument("--message", default="difesa codex", help="Messaggio da associare a log/broadcast")
    parser.add_argument(
        "--tags",
        default="codex,ipfs,defense",
        help="Tag separati da virgola per la memoria Codex",
    )
    parser.add_argument("--no-upload", action="store_true", help="Non caricare su IPFS")
    parser.add_argument("--no-log", action="store_true", help="Non registrare memoria su Codex Server")
    parser.add_argument("--no-broadcast", action="store_true", help="Non inviare broadcast P2P")
    parser.add_argument(
        "--p2p-type",
        default="guardian_alert",
        help="MessageType P2P (es: guardian_alert, memory_sync, agent_request)",
    )
    args = parser.parse_args()

    codex_path = Path(args.file)
    if not codex_path.exists():
        raise SystemExit(f"File non trovato: {codex_path}")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    print(f"[INFO] Calcolo SHA-512 di {codex_path}")
    digest = hash_file(codex_path)
    print(f"[INFO] SHA-512: {digest}")

    backup_path = backup_file(codex_path, Path(args.backup_dir))
    print(f"[INFO] Backup creato: {backup_path}")

    hash_path = Path(args.hash_file)
    write_hash_file(digest, hash_path)
    print(f"[INFO] Hash salvato in {hash_path}")

    if verify_file(codex_path, hash_path):
        print("[INFO] Integrità verificata (hash combacia).")
    else:
        print("[WARN] Integrità non verificata: hash differente.")

    cid = None
    if not args.no_upload:
        print("[INFO] Upload su IPFS in corso...")
        cid = ipfs_add(codex_path)
        print(f"[INFO] CID: {cid}")
        print(f"[INFO] Link gateway: https://ipfs.io/ipfs/{cid}")

    base_url = args.base_url.rstrip("/")
    if cid and not args.no_log:
        mem_id = log_memory(base_url, cid, args.message, tags)
        if mem_id:
            print(f"[INFO] Memoria registrata su Codex Server (id={mem_id})")

    if cid and not args.no_broadcast:
        payload = {
            "cid": cid,
            "message": args.message,
            "file": str(codex_path),
            "ts": datetime.now(timezone.utc).isoformat(),
            "tags": tags,
        }
        if broadcast(base_url, args.p2p_type, payload):
            print("[INFO] Broadcast P2P inviato.")

    print("\n✅ Protocollo difesa Codex completato.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

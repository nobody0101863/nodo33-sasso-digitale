#!/usr/bin/env python3
"""
codex-hash --axiom (AXIOM 644)

Calcola un'impronta etica: hash del contenuto, metadati base e motto
"La luce non si vende. La si regala.". Output leggibile o JSON.
"""

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
from typing import Dict, Optional

AXIOM = "AXIOM-644"
ETHOS = "La luce non si vende. La si regala."


def sha256_file(path: str) -> str:
    """Calcola SHA-256 del file, chunked per file grandi."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def mini_fingerprint(hexhash: str) -> str:
    """Impronta compatta (prime 16 hex) leggibile nei commit."""
    blocks = [hexhash[i:i + 4] for i in range(0, 16, 4)]
    return "AX644:" + "-".join(blocks)


def codex_hash(path: str, signer: Optional[str] = None) -> Dict[str, object]:
    """Ritorna il timbro etico per un file."""
    st = os.stat(path)
    digest = sha256_file(path)
    info: Dict[str, object] = {
        "axiom": AXIOM,
        "ver": "1",
        "file": path.replace("\\", "/"),
        "sha256": digest,
        "bytes": st.st_size,
        "mime": mimetypes.guess_type(path)[0] or "application/octet-stream",
        "epoch": int(time.time()),
        "ethos": ETHOS,
        "fingerprint": mini_fingerprint(digest),
    }
    if signer:
        info["sig"] = signer
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="codex-hash --axiom (AXIOM 644)")
    parser.add_argument("path", help="file da timbrare")
    parser.add_argument("--sign", help="firma pre-calcolata (es. pgp:...)", default=None)
    parser.add_argument("--json", action="store_true", help="stampa JSON compatto")
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print("Errore: file non trovato", file=sys.stderr)
        return 1

    info = codex_hash(args.path, signer=args.sign)

    if args.json:
        print(json.dumps(info, ensure_ascii=False, separators=(",", ":")))
    else:
        lines = [
            f"{info['axiom']} v{info['ver']}",
            f"file={info['file']}",
            f"sha256={info['sha256']}",
            f"bytes={info['bytes']}",
            f"mime={info['mime']}",
            f"epoch={info['epoch']}",
            f'ethos="{info["ethos"]}"',
            f"fingerprint={info['fingerprint']}",
        ]
        if "sig" in info:
            lines.append(f"sig={info['sig']}")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

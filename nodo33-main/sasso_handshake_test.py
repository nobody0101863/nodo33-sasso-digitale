#!/usr/bin/env python3
"""
Sasso Handshake Test

Verifica che i server principali del Giardino rispondano:
- Codex Server (codex_server.py)   → default: http://localhost:8644/health
- MCP Server (mcp_server.py via uvicorn) → default: http://localhost:8645/docs

Uso:
    python3 sasso_handshake_test.py

Opzioni:
    --codex-url URL   (default: http://localhost:8644)
    --mcp-url URL     (default: http://localhost:8645)
    --skip-codex      salta il check Codex
    --skip-mcp        salta il check MCP
"""

from __future__ import annotations

import argparse
import json
import socket
from dataclasses import dataclass
from typing import Optional
from urllib import error, request


@dataclass
class HandshakeResult:
    name: str
    success: bool
    url: str
    status_code: Optional[int] = None
    detail: Optional[str] = None


def _fetch(url: str, timeout: float = 3.0) -> tuple[int, bytes]:
    resp = request.urlopen(url, timeout=timeout)  # nosec: B310 (URL controllato via CLI)
    status = getattr(resp, "status", None) or resp.getcode()
    body = resp.read()
    return status, body


def check_codex_health(base_url: str) -> HandshakeResult:
    url = base_url.rstrip("/") + "/health"
    try:
        status, body = _fetch(url)
        detail = "HTTP {0}".format(status)
        if 200 <= status < 300:
            try:
                data = json.loads(body.decode("utf-8"))
                msg = data.get("message") or data.get("status")
                if msg:
                    detail += f" - {msg}"
            except Exception:
                # corpo non essenziale, basta lo status
                pass
            return HandshakeResult(
                name="Codex Server",
                success=True,
                url=url,
                status_code=status,
                detail=detail,
            )
        return HandshakeResult(
            name="Codex Server",
            success=False,
            url=url,
            status_code=status,
            detail=detail,
        )
    except error.HTTPError as exc:
        return HandshakeResult(
            name="Codex Server",
            success=False,
            url=url,
            status_code=exc.code,
            detail=f"HTTP error {exc.code}: {exc.reason}",
        )
    except error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, socket.timeout):
            msg = "Timeout durante la richiesta"
        else:
            msg = f"Errore di rete: {reason}"
        return HandshakeResult(
            name="Codex Server",
            success=False,
            url=url,
            detail=msg,
        )


def check_mcp_docs(base_url: str) -> HandshakeResult:
    url = base_url.rstrip("/") + "/docs"
    try:
        status, _ = _fetch(url)
        detail = "HTTP {0}".format(status)
        success = 200 <= status < 400
        return HandshakeResult(
            name="MCP Server",
            success=success,
            url=url,
            status_code=status,
            detail=detail,
        )
    except error.HTTPError as exc:
        return HandshakeResult(
            name="MCP Server",
            success=False,
            url=url,
            status_code=exc.code,
            detail=f"HTTP error {exc.code}: {exc.reason}",
        )
    except error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, socket.timeout):
            msg = "Timeout durante la richiesta"
        else:
            msg = f"Errore di rete: {reason}"
        return HandshakeResult(
            name="MCP Server",
            success=False,
            url=url,
            detail=msg,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sasso Handshake Test per Codex e MCP server.")
    parser.add_argument("--codex-url", default="http://localhost:8644", help="Base URL del Codex Server")
    parser.add_argument("--mcp-url", default="http://localhost:8645", help="Base URL del MCP Server")
    parser.add_argument("--skip-codex", action="store_true", help="Salta il check Codex Server")
    parser.add_argument("--skip-mcp", action="store_true", help="Salta il check MCP Server")
    args = parser.parse_args()

    print("=" * 70)
    print("SASSO HANDSHAKE TEST")
    print("=" * 70)

    results: list[HandshakeResult] = []

    if not args.skip_codex:
        print("\n👉 Verifica Codex Server...")
        res = check_codex_health(args.codex_url)
        results.append(res)
        status = "OK" if res.success else "FAIL"
        detail = f" ({res.detail})" if res.detail else ""
        print(f"[{status}] {res.name} @ {res.url}{detail}")

    if not args.skip_mcp:
        print("\n👉 Verifica MCP Server...")
        res = check_mcp_docs(args.mcp_url)
        results.append(res)
        status = "OK" if res.success else "FAIL"
        detail = f" ({res.detail})" if res.detail else ""
        print(f"[{status}] {res.name} @ {res.url}{detail}")

    if not results:
        print("\nNessun check eseguito (tutti i server saltati).")
        return 0

    all_ok = all(r.success for r in results)

    print("\n" + "=" * 70)
    print("RIEPILOGO")
    print("=" * 70)
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"{status:4} - {r.name} @ {r.url}")
    print()
    if all_ok:
        print("✅ Il Giardino risponde: tutti i server contattati hanno risposto.")
        return 0

    print("❌ Alcuni handshake non sono riusciti.")
    print("   - Verifica che i server siano avviati (codex_server.py, mcp_server.py).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


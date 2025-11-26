#!/usr/bin/env python3
"""
Codex / Sasso Health Check - Unified CLI

Scopo:
- Verificare configurazione (.env + config.py)
- Controllare lo stato dei database principali
- (Opzionale) Verificare reachability dei server HTTP Codex/MCP

Usage:
    python3 codex_health_check.py
    python3 codex_health_check.py --summary-only
    python3 codex_health_check.py --skip-network
    python3 codex_health_check.py --server-url http://localhost:8644
    python3 codex_health_check.py --mcp-url http://localhost:8645
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from enum import Enum
from io import StringIO
import os
from pathlib import Path
from typing import List, Optional

from codex_unified_db import CodexUnifiedDB

import config as config_module

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore


class Status(str, Enum):
    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class HealthCheckResult:
    name: str
    status: Status
    detail: str


def _load_config_silent() -> config_module.Config:
    """Carica la config usando config.load_config, silenziando l'output."""
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        cfg = config_module.load_config()
    return cfg


def check_config() -> tuple[HealthCheckResult, Optional[config_module.Config]]:
    """Verifica caricamento e validazione configurazione."""
    try:
        cfg = _load_config_silent()
        detail = f"Loaded OK (CODEX_BASE_URL={cfg.CODEX_BASE_URL})"
        return HealthCheckResult("Config (.env + env)", Status.OK, detail), cfg
    except Exception as exc:
        return HealthCheckResult("Config (.env + env)", Status.ERROR, str(exc)), None


def check_unified_db(db_path: Path) -> HealthCheckResult:
    """Verifica esistenza e accesso al Codex Unified DB."""
    if not db_path.exists():
        return HealthCheckResult(
            "codex_unified.db",
            Status.WARN,
            f"Not found at {db_path} (run: python3 codex_unified_db.py --init)",
        )

    try:
        db = CodexUnifiedDB(db_path)
        stats = db.get_gift_stats()
        total = stats.get("total", 0)
        return HealthCheckResult(
            "codex_unified.db",
            Status.OK,
            f"Initialized (gifts={total})",
        )
    except Exception as exc:
        return HealthCheckResult("codex_unified.db", Status.ERROR, str(exc))


def check_codex_server_db(db_path: Path) -> HealthCheckResult:
    """Verifica il DB legacy del Codex server REST."""
    if not db_path.exists():
        return HealthCheckResult(
            "codex_server.db",
            Status.WARN,
            f"Not found at {db_path} (will be created on first codex_server.py run)",
        )

    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='request_log'"
        )
        has_log = cursor.fetchone() is not None
        conn.close()

        if has_log:
            return HealthCheckResult(
                "codex_server.db",
                Status.OK,
                "Schema appears initialized (request_log present)",
            )
        return HealthCheckResult(
            "codex_server.db",
            Status.WARN,
            "DB exists but request_log table not found",
        )
    except Exception as exc:
        return HealthCheckResult("codex_server.db", Status.ERROR, str(exc))


def check_codex_server_http(base_url: str) -> HealthCheckResult:
    """Verifica l'endpoint /health del Codex Server REST."""
    if requests is None:
        return HealthCheckResult(
            "Codex Server HTTP",
            Status.WARN,
            "python-requests not installed (pip install requests) - HTTP check skipped",
        )

    url = base_url.rstrip("/") + "/health"
    try:
        resp = requests.get(url, timeout=3)
    except Exception as exc:
        return HealthCheckResult(
            "Codex Server HTTP",
            Status.ERROR,
            f"Cannot reach {url}: {exc}",
        )

    if resp.status_code != 200:
        return HealthCheckResult(
            "Codex Server HTTP",
            Status.WARN,
            f"{url} responded with status {resp.status_code}",
        )

    try:
        data = resp.json()
    except Exception:
        data = None

    status_text = ""
    if isinstance(data, dict):
        status_text = data.get("status") or data.get("message") or ""

    detail = f"OK at {url}"
    if status_text:
        detail += f" ({status_text})"

    return HealthCheckResult("Codex Server HTTP", Status.OK, detail)


def check_mcp_server_http(base_url: str) -> HealthCheckResult:
    """Verifica la reachability base di un MCP server FastAPI (openapi.json)."""
    if requests is None:
        return HealthCheckResult(
            "MCP Server HTTP",
            Status.WARN,
            "python-requests not installed (pip install requests) - HTTP check skipped",
        )

    url = base_url.rstrip("/") + "/openapi.json"
    try:
        resp = requests.get(url, timeout=3)
    except Exception as exc:
        return HealthCheckResult(
            "MCP Server HTTP",
            Status.WARN,
            f"Cannot reach {url}: {exc} (run uvicorn mcp_server:app --port 8645 or adjust --mcp-url)",
        )

    if resp.status_code != 200:
        return HealthCheckResult(
            "MCP Server HTTP",
            Status.WARN,
            f"{url} responded with status {resp.status_code}",
        )

    return HealthCheckResult(
        "MCP Server HTTP",
        Status.OK,
        f"OK at {url}",
    )


def overall_status(results: List[HealthCheckResult]) -> Status:
    """Calcola lo stato complessivo (ERROR > WARN > OK)."""
    statuses = {r.status for r in results}
    if Status.ERROR in statuses:
        return Status.ERROR
    if Status.WARN in statuses:
        return Status.WARN
    return Status.OK


def print_report(results: List[HealthCheckResult], summary_only: bool = False) -> None:
    """Stampa report in forma compatta."""
    print("=" * 70)
    print("CODEX / SASSO HEALTH CHECK")
    print("=" * 70)

    if summary_only:
        for r in results:
            print(f"[{r.status.value:5}] {r.name}")
    else:
        for r in results:
            print(f"[{r.status.value:5}] {r.name} - {r.detail}")

    print("-" * 70)
    ovr = overall_status(results)
    print(f"OVERALL STATUS: {ovr.value}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified health check for Codex/Sasso (config, DB, servers)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Mostra solo stato per voce senza dettagli",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Salta i check HTTP (utile in ambienti offline)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Base URL del Codex Server (default: CODEX_BASE_URL o http://localhost:8644)",
    )
    parser.add_argument(
        "--mcp-url",
        type=str,
        default=None,
        help="Base URL dell'MCP Server (default: MCP_BASE_URL o http://localhost:8645)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Percorso al codex_unified.db (default da config o ./codex_unified.db)",
    )

    args = parser.parse_args()

    results: List[HealthCheckResult] = []

    # Config
    cfg_result, cfg = check_config()
    results.append(cfg_result)

    # Unified DB
    db_path = args.db_path
    if db_path is None:
        if cfg is not None and getattr(cfg, "CODEX_DB_PATH", None):
            db_path = Path(cfg.CODEX_DB_PATH)
        else:
            db_path = Path("codex_unified.db")
    results.append(check_unified_db(db_path))

    # Legacy Codex server DB (codex_server.py)
    results.append(check_codex_server_db(Path("codex_server.db")))

    # HTTP-based checks
    if not args.skip_network:
        # Codex Server REST
        server_url = args.server_url
        if server_url is None:
            if cfg is not None and getattr(cfg, "CODEX_BASE_URL", None):
                server_url = cfg.CODEX_BASE_URL
            else:
                server_url = "http://localhost:8644"
        results.append(check_codex_server_http(server_url))

        # MCP Server (FastAPI via uvicorn)
        mcp_url = args.mcp_url or os.environ.get("MCP_BASE_URL", "http://localhost:8645")
        results.append(check_mcp_server_http(mcp_url))
    else:
        results.append(
            HealthCheckResult(
                "HTTP Checks",
                Status.WARN,
                "Skipped (--skip-network enabled)",
            )
        )

    print_report(results, summary_only=args.summary_only)
    ovr = overall_status(results)
    return 0 if ovr is not Status.ERROR else 1


if __name__ == "__main__":
    raise SystemExit(main())


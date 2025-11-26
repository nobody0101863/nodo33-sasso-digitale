#!/usr/bin/env python3
"""
Validation helper for MCP Apps component manifests.
Checks presence of identity, permissions, assets, and integrity fields.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

ALG_ALLOW = {"ed25519", "ecdsa-p256"}


def load_manifest(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(f"PyYAML not available to parse {path}")
        return yaml.safe_load(text)
    return json.loads(text)


def validate_manifest(data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    for field in ("component_id", "version", "owner"):
        if not data.get(field):
            errors.append(f"missing required field: {field}")

    capabilities = data.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        warnings.append("capabilities should be a non-empty list")

    permissions = data.get("permissions") or {}
    allow = permissions.get("allow")
    deny = permissions.get("deny")
    if not isinstance(allow, list) or not allow:
        warnings.append("permissions.allow should be a non-empty list")
    if not isinstance(deny, list) or not deny:
        warnings.append("permissions.deny should be a non-empty list")

    assets = data.get("assets")
    if not isinstance(assets, list) or not assets:
        errors.append("assets must list bundled resources with hashes")
    else:
        for asset in assets:
            if not asset.get("path") or not asset.get("hash"):
                errors.append("each asset needs path and hash")

    integrity = data.get("integrity") or {}
    if not integrity.get("hash"):
        warnings.append("integrity.hash missing")
    if not integrity.get("signature"):
        warnings.append("integrity.signature missing")
    alg = integrity.get("alg")
    if alg and alg not in ALG_ALLOW:
        errors.append(f"integrity.alg must be one of {sorted(ALG_ALLOW)}")
    if not integrity.get("key_id"):
        warnings.append("integrity.key_id missing (recommended)")

    resources = data.get("resources") or {}
    if resources.get("requires_network") is True:
        warnings.append("requires_network=true — ensure sandbox/CSP on client")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MCP Apps component manifest(s).")
    parser.add_argument("manifests", nargs="+", help="Path(s) to JSON/YAML manifests")
    args = parser.parse_args()

    had_errors = False
    for raw_path in args.manifests:
        path = Path(raw_path)
        try:
            data = load_manifest(path)
            errors, warnings = validate_manifest(data)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[{path}] ERROR: {exc}")
            had_errors = True
            continue

        if errors:
            had_errors = True
            print(f"[{path}] ❌ invalid")
            for err in errors:
                print(f"  - ERROR: {err}")
        else:
            print(f"[{path}] ✅ basic schema ok")

        for warn in warnings:
            print(f"  - WARN: {warn}")

    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())

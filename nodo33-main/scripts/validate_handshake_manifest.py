#!/usr/bin/env python3
"""
Validation helper for Codex handshake manifests (codex-handshake/0.1).
Checks schema basics: spec, intent canon, policy, state, integrity, TTL.
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

INTENT_CANON = {
    "protect_garden",
    "protect_children",
    "defuse_conflict",
    "share_light",
    "minimal_data",
    "stop_harm",
    "route_to_human",
}

STATE_VALUES = {"asserted", "observed", "revoked"}
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

    if data.get("spec") != "codex-handshake/0.1":
        errors.append("spec must be codex-handshake/0.1")

    signal = data.get("signal") or {}
    intent = signal.get("intent")
    if intent not in INTENT_CANON:
        errors.append(f"intent must be one of {sorted(INTENT_CANON)}")

    confidence = signal.get("confidence")
    if confidence is not None and not (0 <= float(confidence) <= 1):
        warnings.append("confidence should be between 0 and 1")

    policy = data.get("policy") or {}
    allow = policy.get("allow")
    deny = policy.get("deny")
    if not isinstance(allow, list) or not allow:
        warnings.append("policy.allow should be a non-empty list")
    if not isinstance(deny, list) or not deny:
        warnings.append("policy.deny should be a non-empty list")

    state = (data.get("state") or {}).get("value")
    if state and state not in STATE_VALUES:
        errors.append(f"state.value must be one of {sorted(STATE_VALUES)}")

    telemetry = data.get("telemetry") or {}
    ttl = telemetry.get("ttl")
    if ttl is None:
        warnings.append("telemetry.ttl is recommended for expiry control")
    elif not isinstance(ttl, str):
        errors.append("telemetry.ttl must be a string (e.g., '120s' or ISO8601 duration)")

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

    issuer = data.get("issuer") or {}
    if not issuer.get("id"):
        warnings.append("issuer.id missing")
    subject = data.get("subject") or {}
    if not subject.get("agent_id"):
        warnings.append("subject.agent_id missing")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Codex handshake manifest(s).")
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

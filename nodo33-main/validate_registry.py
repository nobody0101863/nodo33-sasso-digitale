#!/usr/bin/env python3
"""
Validazione rapida di registry.yaml usando schemas_registry.RegistryModel.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from custos.schemas_registry import RegistryModel


def validate_registry(path: str | Path = "registry.yaml") -> RegistryModel:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return RegistryModel(**raw)


if __name__ == "__main__":
    registry = validate_registry()
    print(f"Registry valido: {len(registry.groups)} gruppi caricati")

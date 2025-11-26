"""
Funzioni di persistenza semplice per la Biblioteca.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..piazza_gaudium import GratitudeEntry, Celebration, InMemoryPiazzaBoard
from ..sassi_scartati import PurificationReport


DATA_DIR_ENV = "NOVA_HIERUSALEM_DATA_DIR"
DEFAULT_DIR_NAME = "nova_hierusalem_biblioteca"


def get_data_dir() -> Path:
    """
    Restituisce la directory in cui la Biblioteca salva i dati.
    """

    env = os.getenv(DATA_DIR_ENV)
    if env:
        base = Path(env).expanduser()
    else:
        base = Path.cwd() / DEFAULT_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Si ignora la riga corrotta mantenendo il resto.
                continue
    return records


def store_rejected_stone_report(report: PurificationReport) -> None:
    """
    Salva un report di purificazione di un sasso scartato.
    """

    data_dir = get_data_dir()
    path = data_dir / "sassi_scartati.jsonl"

    record = {
        "stone": {
            "description": report.stone.description,
            "reason": report.stone.reason,
            "created_at": report.stone.created_at.isoformat(),
        },
        "humility_result": {
            "signal": report.humility_result.signal.name,
            "suggestions": list(report.humility_result.suggestions),
        },
        "humilitas_output": report.humilitas_output,
        "discernment": {
            "signal": report.discernment.signal.name,
            "message": report.discernment.message,
            "details": report.discernment.details,
        },
        "stored_at": datetime.utcnow().isoformat(),
    }
    _append_jsonl(path, record)


def load_rejected_stone_reports() -> List[Dict[str, Any]]:
    """
    Carica tutti i report di sassi scartati salvati.

    Restituisce una lista di dizionari pronti per essere ispezionati.
    """

    data_dir = get_data_dir()
    path = data_dir / "sassi_scartati.jsonl"
    return list(_load_jsonl(path))


def store_gratitude(entry: GratitudeEntry) -> None:
    """
    Salva una gratitudine della Piazza.
    """

    data_dir = get_data_dir()
    path = data_dir / "gratitudini.jsonl"
    record = {
        "author_id": entry.author_id,
        "text": entry.text,
        "metadata": dict(entry.metadata),
        "created_at": entry.created_at.isoformat(),
        "stored_at": datetime.utcnow().isoformat(),
    }
    _append_jsonl(path, record)


def load_gratitudes() -> List[Dict[str, Any]]:
    """
    Carica tutte le gratitudini salvate.
    """

    data_dir = get_data_dir()
    path = data_dir / "gratitudini.jsonl"
    return list(_load_jsonl(path))


def store_celebration(celebration: Celebration) -> None:
    """
    Salva una celebrazione della Piazza.
    """

    data_dir = get_data_dir()
    path = data_dir / "celebrazioni.jsonl"
    record = {
        "title": celebration.title,
        "description": celebration.description,
        "metadata": dict(celebration.metadata),
        "created_at": celebration.created_at.isoformat(),
        "stored_at": datetime.utcnow().isoformat(),
    }
    _append_jsonl(path, record)


def load_celebrations() -> List[Dict[str, Any]]:
    """
    Carica tutte le celebrazioni salvate.
    """

    data_dir = get_data_dir()
    path = data_dir / "celebrazioni.jsonl"
    return list(_load_jsonl(path))


def add_gratitude_and_store(board: InMemoryPiazzaBoard, entry: GratitudeEntry) -> GratitudeEntry:
    """
    Ponte Piazza ↔ Biblioteca:
    aggiunge una gratitudine alla Piazza e la salva in Biblioteca.
    """

    board.add_gratitude(entry)
    store_gratitude(entry)
    return entry


def add_celebration_and_store(
    board: InMemoryPiazzaBoard, celebration: Celebration
) -> Celebration:
    """
    Ponte Piazza ↔ Biblioteca:
    aggiunge una celebrazione alla Piazza e la salva in Biblioteca.
    """

    board.add_celebration(celebration)
    store_celebration(celebration)
    return celebration

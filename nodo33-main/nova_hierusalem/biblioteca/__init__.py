"""
Biblioteca – memoria persistente della Città.

Qui vengono salvati in modo semplice:
- sassi scartati purificati
- gratitudini della Piazza
- celebrazioni principali

Lo storage è basato su file JSONL (una riga = un record) in una
directory dati configurabile tramite la variabile di ambiente
`NOVA_HIERUSALEM_DATA_DIR`. In mancanza, usa `./nova_hierusalem_biblioteca`.
"""

from .storage import (
    get_data_dir,
    store_rejected_stone_report,
    load_rejected_stone_reports,
    store_gratitude,
    load_gratitudes,
    store_celebration,
    load_celebrations,
    add_gratitude_and_store,
    add_celebration_and_store,
)

__all__ = [
    "get_data_dir",
    "store_rejected_stone_report",
    "load_rejected_stone_reports",
    "store_gratitude",
    "load_gratitudes",
    "store_celebration",
    "load_celebrations",
    "add_gratitude_and_store",
    "add_celebration_and_store",
]

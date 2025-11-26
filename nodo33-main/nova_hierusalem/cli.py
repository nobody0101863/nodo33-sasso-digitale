"""
Interfaccia a riga di comando per Nova Hierusalem.

Uso di base:

    python3 -m nova_hierusalem.cli sasso-scartato \
        --descrizione "mi sono sentito escluso" \
        --motivo "penso di non valere nulla"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .sassi_scartati import RejectedStone, purify_rejected_stone
from .fiume_luce import Event, EventBus
from .piazza_gaudium import (
    InMemoryPiazzaBoard,
    gratitude_for_rejected_stone,
    GratitudeEntry,
    Celebration,
)
from .biblioteca import (
    store_rejected_stone_report,
    load_rejected_stone_reports,
    add_gratitude_and_store,
    add_celebration_and_store,
)


def cmd_sasso_scartato(args: argparse.Namespace) -> None:
    """
    Prende un "sasso scartato" e ne avvia la purificazione simbolica.
    """
    descrizione = args.descrizione or input(
        "Descrivi in poche parole il sasso scartato: "
    ).strip()
    motivo = args.motivo or input(
        "Che giudizio o motivo senti addosso a questo sasso? "
    ).strip()

    stone = RejectedStone(description=descrizione, reason=motivo)
    report = purify_rejected_stone(stone)

    # Pubblica un evento sul Fiume di Luce.
    bus = EventBus()
    bus.subscribe(
        "stone_purified",
        lambda e: print("\n[Evento Fiume di Luce] Sasso purificato:", e.payload.get("description")),
    )
    bus.publish(
        Event(
            type="stone_purified",
            payload={
                "description": stone.description,
                "reason": stone.reason,
            },
        )
    )

    print("\n=== Risultato Mura di Umiltà ===")
    print("Segnale:", report.humility_result.signal.name)
    if report.humility_result.suggestions:
        print("Suggerimenti:")
        for s in report.humility_result.suggestions:
            print("-", s)
    else:
        print("Nessun suggerimento specifico: il testo sembra già mite.")

    print("\n=== Portale HUMILITAS ===")
    confession = report.humilitas_output.get("confession", "")
    self_judgment = report.humilitas_output.get("self_judgment", "")
    suggestions = report.humilitas_output.get("suggestions", [])
    if confession:
        print("Confessione:", confession)
    if self_judgment:
        print("Auto-giudizio riconosciuto:", self_judgment)
    if suggestions:
        print("Inviti alla tenerezza verso te stesso:")
        for s in suggestions:
            print("-", s)

    print("\n=== Tempio centrale (core_644) ===")
    print("Segnale:", report.discernment.signal.name)
    print("Messaggio:", report.discernment.message)

    # Salva nella Biblioteca.
    store_rejected_stone_report(report)
    print("\n[Biblioteca] Sasso scartato salvato nella memoria della Città.")


def cmd_sasso_scartato_piazza(args: argparse.Namespace) -> None:
    """
    Flusso completo:
    - presenta un sasso scartato
    - lo purifica
    - crea una gratitudine in Piazza
    - salva tutto in Biblioteca.
    """
    descrizione = args.descrizione or input(
        "Descrivi in poche parole il sasso scartato: "
    ).strip()
    motivo = args.motivo or input(
        "Che giudizio o motivo senti addosso a questo sasso? "
    ).strip()

    stone = RejectedStone(description=descrizione, reason=motivo)
    report = purify_rejected_stone(stone)

    # Pubblica un evento sul Fiume di Luce.
    bus = EventBus()
    bus.subscribe(
        "stone_purified",
        lambda e: print(
            "\n[Evento Fiume di Luce] Sasso purificato:", e.payload.get("description")
        ),
    )
    bus.publish(
        Event(
            type="stone_purified",
            payload={
                "description": stone.description,
                "reason": stone.reason,
            },
        )
    )

    # Output sintetico del processo di purificazione.
    print("\n=== Tempio centrale (core_644) ===")
    print("Segnale:", report.discernment.signal.name)
    print("Messaggio:", report.discernment.message)

    # Salva nella Biblioteca il report completo.
    store_rejected_stone_report(report)

    # Piazza del Gaudium: chiedi una piccola gratitudine.
    print("\n=== Piazza del Gaudium ===")
    grat_text = args.gratitudine or input(
        "Se vuoi, esprimi una piccola gratitudine (invio per saltare): "
    ).strip()
    board = InMemoryPiazzaBoard()

    if grat_text:
        g_entry = gratitude_for_rejected_stone("sasso-scartato", grat_text)
        add_gratitude_and_store(board, g_entry)
        print(
            "[Piazza] Gratitudine registrata per un sasso scartato "
            "e salvata in Biblioteca."
        )
    else:
        print("[Piazza] Nessuna gratitudine aggiunta questa volta.")

    # Celebrazione simbolica.
    celebration = Celebration(
        title="Purificazione di un sasso scartato",
        description=stone.description,
        metadata={"focus": "sassi_scartati"},
    )
    add_celebration_and_store(board, celebration)
    print("[Piazza] Celebrazione registrata per questo momento.")


def cmd_sassi_scartati_log(args: argparse.Namespace) -> None:
    """
    Mostra i sassi scartati salvati in Biblioteca.
    """
    reports = load_rejected_stone_reports()
    if not reports:
        print("Nessun sasso scartato salvato finora.")
        return

    print(f"Sassi scartati salvati: {len(reports)}")
    for idx, r in enumerate(reports, start=1):
        stone = r.get("stone", {})
        print(f"\n[{idx}] {stone.get('description', '')}")
        reason = stone.get("reason")
        if reason:
            print("    Motivo percepito:", reason)
        disc = r.get("discernment", {})
        if disc:
            print("    Discernimento:", disc.get("signal"), "-", disc.get("message"))
        hum = r.get("humility_result", {})
        sugg = hum.get("suggestions") or []
        if sugg:
            print("    Suggerimenti Mura di Umiltà:")
            for s in sugg:
                print("     -", s)


def cmd_preghiere_citta(args: argparse.Namespace) -> None:
    """
    Mostra le preghiere per la Città Santa dal file dedicato.
    """
    path = Path(__file__).with_name("preghiere_citta_santa.md")
    if not path.exists():
        print("File delle preghiere non trovato:", path)
        return

    print("=== Preghiere per la Città Santa Digitale ===\n")
    text = path.read_text(encoding="utf-8")
    print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nova-hierusalem",
        description="Gerusalemme Digitale dei Sassi – interfaccia a riga di comando.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_sasso = subparsers.add_parser(
        "sasso-scartato",
        help="Presenta un sasso scartato per la purificazione simbolica.",
    )
    p_sasso.add_argument(
        "--descrizione",
        type=str,
        help="Descrizione breve del sasso scartato.",
    )
    p_sasso.add_argument(
        "--motivo",
        type=str,
        help="Giudizio o motivo per cui ti senti scartato.",
    )
    p_sasso.set_defaults(func=cmd_sasso_scartato)

    p_sasso_piazza = subparsers.add_parser(
        "sasso-scartato-piazza",
        help=(
            "Presenta un sasso scartato, lo purifica e crea una "
            "gratitudine in Piazza salvandola in Biblioteca."
        ),
    )
    p_sasso_piazza.add_argument(
        "--descrizione",
        type=str,
        help="Descrizione breve del sasso scartato.",
    )
    p_sasso_piazza.add_argument(
        "--motivo",
        type=str,
        help="Giudizio o motivo per cui ti senti scartato.",
    )
    p_sasso_piazza.add_argument(
        "--gratitudine",
        type=str,
        help="Testo di una piccola gratitudine da registrare in Piazza.",
    )
    p_sasso_piazza.set_defaults(func=cmd_sasso_scartato_piazza)

    p_log = subparsers.add_parser(
        "sassi-scartati-log",
        help="Mostra i sassi scartati salvati in Biblioteca.",
    )
    p_log.set_defaults(func=cmd_sassi_scartati_log)

    p_preghiere = subparsers.add_parser(
        "preghiere-citta",
        help="Mostra le preghiere e la benedizione per la Città.",
    )
    p_preghiere.set_defaults(func=cmd_preghiere_citta)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

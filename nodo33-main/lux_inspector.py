#!/usr/bin/env python3
"""
Lux Inspector - Analizzatore di backup preferenze macOS

Legge una cartella di backup (es. lux_backup) in sola lettura e
produce un piccolo report umano sulle impostazioni principali:
- Dock
- Finder
- Terminal
- Spotlight
- Menu bar / system UI
- Screenshot

Uso:
    python lux_inspector.py /percorso/a/lux_backup

Non modifica alcun file, né nel backup né nelle preferenze correnti.
"""

import argparse
import os
import plistlib
from typing import Any, Dict, Optional, Tuple, List


def load_plist(base_dir: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Carica un plist dal backup se esiste.

    Args:
        base_dir: percorso alla cartella di backup (es. lux_backup)
        name: nome del file plist (es. com.apple.dock.plist)

    Returns:
        dict del contenuto plist oppure None se non trovato / errore.
    """
    path = os.path.join(base_dir, name)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "rb") as f:
            return plistlib.load(f)
    except Exception:
        return None


def describe_dock(plist_data: Dict[str, Any]) -> str:
    size = plist_data.get("tilesize")
    magnification = plist_data.get("magnification", False)
    position = plist_data.get("orientation", "bottom")
    autohide = plist_data.get("autohide", False)

    parts = ["Dock:"]
    if size is not None:
        parts.append(f"  - dimensione icone: {size}")
    parts.append(f"  - magnification (ingrandimento): {'attivo' if magnification else 'spento'}")
    parts.append(f"  - posizione: {position}")
    parts.append(f"  - nascondi automaticamente: {'sì' if autohide else 'no'}")
    return "\n".join(parts)


def describe_finder(plist_data: Dict[str, Any]) -> str:
    show_all_ext = plist_data.get("AppleShowAllExtensions")
    new_window_target = plist_data.get("NewWindowTargetPath") or plist_data.get("NewWindowTarget")
    desktop_show_hd = plist_data.get("ShowHardDrivesOnDesktop")

    parts = ["Finder:"]
    if show_all_ext is not None:
        parts.append(f"  - mostra estensioni file: {'sì' if show_all_ext else 'no'}")
    if new_window_target:
        parts.append(f"  - cartella per nuove finestre: {new_window_target}")
    if desktop_show_hd is not None:
        parts.append(f"  - mostra dischi sul desktop: {'sì' if desktop_show_hd else 'no'}")
    return "\n".join(parts)


def describe_terminal(plist_data: Dict[str, Any]) -> str:
    default_profile = plist_data.get("Default Window Settings") or plist_data.get("Default Profile")
    startup_profiles = plist_data.get("Startup Profile List")

    parts = ["Terminale:"]
    if default_profile:
        parts.append(f"  - profilo di default: {default_profile}")
    if startup_profiles:
        parts.append(f"  - profili all'avvio: {', '.join(startup_profiles)}")
    return "\n".join(parts)


def describe_spotlight(plist_data: Dict[str, Any]) -> str:
    ordered_items = plist_data.get("orderedItems") or plist_data.get("orderedItemsV2")

    parts = ["Spotlight:"]
    if isinstance(ordered_items, list):
        enabled = []
        disabled = []
        for item in ordered_items:
            name = item.get("name") or item.get("displayName")
            enabled_flag = item.get("enabled", True)
            if not name:
                continue
            (enabled if enabled_flag else disabled).append(name)

        if enabled:
            parts.append(f"  - categorie attive: {', '.join(enabled)}")
        if disabled:
            parts.append(f"  - categorie disattivate: {', '.join(disabled)}")
    return "\n".join(parts)


def describe_system_ui(plist_data: Dict[str, Any]) -> str:
    menu_extras = plist_data.get("menuExtras")

    parts = ["Barra menu (system UI):"]
    if isinstance(menu_extras, list):
        pretty = [os.path.basename(m).replace(".menu", "") for m in menu_extras]
        parts.append(f"  - icone presenti: {', '.join(pretty)}")
    return "\n".join(parts)


def describe_screencapture(plist_data: Dict[str, Any]) -> str:
    location = plist_data.get("location")
    file_type = plist_data.get("type")
    include_shadow = plist_data.get("include-shadow")

    parts = ["Screenshot:"]
    if location:
        parts.append(f"  - cartella salvataggio: {location}")
    if file_type:
        parts.append(f"  - formato file: {file_type}")
    if include_shadow is not None:
        parts.append(f"  - ombra finestre: {'sì' if include_shadow else 'no'}")
    return "\n".join(parts)


def _clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> int:
    """Limita un valore a un range [min_value, max_value] e lo converte in int."""
    return int(max(min_value, min(max_value, value)))


def assess_configuration(
    dock_plist: Optional[Dict[str, Any]],
    finder_plist: Optional[Dict[str, Any]],
    terminal_plist: Optional[Dict[str, Any]],
    system_ui_plist: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, int], List[str]]:
    """
    Valuta simbolicamente la configurazione su tre assi:
    - ego: quanto è “pieno/centrato” l'ambiente
    - gioia: quanto facilita un lavoro sereno
    - rumore: quanta distrazione visiva/presenza superflua
    """
    ego = 50.0
    joy = 50.0
    noise = 50.0
    comments: List[str] = []

    # Dock
    if dock_plist:
        apps = dock_plist.get("persistent-apps") or []
        n_apps = len(apps)
        size = float(dock_plist.get("tilesize", 48.0))
        magnification = bool(dock_plist.get("magnification", False))
        autohide = bool(dock_plist.get("autohide", False))

        if n_apps:
            if n_apps <= 6:
                ego -= 8
                noise -= 10
                joy += 6
                comments.append("Dock piuttosto minimale: focus buono, poco rumore.")
            elif n_apps <= 12:
                comments.append("Dock medio: buon compromesso tra accesso rapido e ordine.")
            else:
                ego += 8
                noise += 12
                joy -= 4
                comments.append("Dock molto pieno: tanta disponibilità, ma anche più rumore.")

        if size >= 64:
            ego += 6
            noise += 4
            comments.append("Icone dock grandi: tutto molto in evidenza (più presenza sullo schermo).")
        elif size <= 40:
            joy += 3
            noise -= 4
            comments.append("Icone dock compatte: più spazio visivo e meno distrazione.")

        if magnification:
            joy += 4
        if autohide:
            ego -= 3
            noise -= 6
            joy += 4
            comments.append("Dock nascosto automaticamente: ambiente più pulito.")

    # Finder
    if finder_plist:
        show_ext = finder_plist.get("AppleShowAllExtensions")
        show_hd = finder_plist.get("ShowHardDrivesOnDesktop")

        if show_ext:
            ego -= 3
            noise -= 3
            joy += 3
            comments.append("Mostrare le estensioni dei file aiuta a capire e riduce ambiguità.")
        if show_hd:
            noise += 4
            comments.append("Dischi visibili sul desktop: pratici ma più elementi sempre in vista.")

    # System UI (barra menu)
    if system_ui_plist:
        menu_extras = system_ui_plist.get("menuExtras") or []
        n_icons = len(menu_extras)
        if n_icons:
            if n_icons <= 4:
                noise -= 6
                joy += 4
                comments.append("Barra menu essenziale: poche icone, mente più libera.")
            elif n_icons <= 8:
                comments.append("Barra menu equilibrata: informazioni utili senza esagerare.")
            else:
                noise += 10
                ego += 4
                comments.append("Molte icone in barra menu: sempre molto stimolo visivo.")

    metrics = {
        "ego": _clamp(ego),
        "gioia": _clamp(joy),
        "rumore": _clamp(noise),
    }
    return metrics, comments


def generate_ai_commentary(summary: str, model: str) -> None:
    """
    Usa un modello OpenAI per generare un commento "intelligente"
    sulla configurazione, se la libreria e la chiave API sono disponibili.
    """
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        print(
            "\n[IA] Libreria 'openai' non trovata. "
            "Installa con: pip install --upgrade openai"
        )
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "\n[IA] Variabile d'ambiente OPENAI_API_KEY non impostata.\n"
            "     Esporta la chiave e riesegui, ad esempio:\n"
            "       export OPENAI_API_KEY='la_tua_chiave'\n"
        )
        return

    client = OpenAI()  # legge la chiave dall'ambiente

    system_prompt = (
        "Sei un assistente che analizza la configurazione di un Mac "
        "a partire da una sintesi di preferenze (Dock, Finder, Terminale, "
        "Spotlight, barra menu, screenshot). "
        "Parla in italiano, tono amichevole ma tecnico, con attenzione a "
        "usabilità, ordine, focus, privacy e 'purezza digitale'. "
        "Non inventare dati che non vedi nella sintesi."
    )

    user_content = (
        "Questa è una sintesi delle preferenze trovate in un backup macOS:\n"
        "---------------------\n"
        f"{summary}\n"
        "---------------------\n"
        "Analizza la configurazione e in massimo 10 righe:\n"
        "- descrivi lo stile generale dell'utente (minimalista, pieno, ecc.)\n"
        "- indica eventuali punti di attenzione (privacy, distrazioni, caos)\n"
        "- suggerisci 3 miglioramenti concreti e pratici.\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
    except Exception as e:
        print(f"\n[IA] Errore nella chiamata al modello OpenAI: {e}")
        return

    message = response.choices[0].message.content

    print("\n" + "=" * 60)
    print("Commento IA (OpenAI)")
    print("=" * 60)
    print(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lux Inspector - Analizza un backup di preferenze macOS (sola lettura)"
    )
    parser.add_argument(
        "backup_dir",
        help="Percorso alla cartella di backup (es. ~/Desktop/lux_backup)"
    )
    parser.add_argument(
        "--ai-openai",
        action="store_true",
        help=(
            "Usa un modello OpenAI per aggiungere un commento IA "
            "(richiede libreria 'openai' e variabile OPENAI_API_KEY)."
        ),
    )
    parser.add_argument(
        "--ai-model",
        default="gpt-4.1",
        help="Nome modello OpenAI da usare per il commento IA (default: gpt-4.1).",
    )
    args = parser.parse_args()

    backup_dir = os.path.expanduser(args.backup_dir)

    if not os.path.isdir(backup_dir):
        print(f"Percorso non valido: {backup_dir}")
        return

    print(f"Analisi backup preferenze: {backup_dir}")
    print("=" * 60)
    print("Nota: solo lettura; nessuna modifica alle preferenze o al backup.\n")

    sections = []

    dock_plist = load_plist(backup_dir, "com.apple.dock.plist")
    if dock_plist:
        sections.append(describe_dock(dock_plist))

    finder_plist = load_plist(backup_dir, "com.apple.finder.plist")
    if finder_plist:
        sections.append(describe_finder(finder_plist))

    term_plist = load_plist(backup_dir, "com.apple.Terminal.plist")
    if term_plist:
        sections.append(describe_terminal(term_plist))

    spotlight_plist = load_plist(backup_dir, "com.apple.Spotlight.plist")
    if spotlight_plist:
        sections.append(describe_spotlight(spotlight_plist))

    system_ui_plist = load_plist(backup_dir, "com.apple.systemuiserver.plist")
    if system_ui_plist:
        sections.append(describe_system_ui(system_ui_plist))

    screencapture_plist = load_plist(backup_dir, "com.apple.screencapture.plist")
    if screencapture_plist:
        sections.append(describe_screencapture(screencapture_plist))

    if not sections:
        print("Nessuna delle plist principali trovata nel backup.")
        return

    report_text = "\n\n".join(sections)
    print(report_text)

    # Valutazione assi simbolici
    metrics, comments = assess_configuration(
        dock_plist=dock_plist,
        finder_plist=finder_plist,
        terminal_plist=term_plist,
        system_ui_plist=system_ui_plist,
    )

    def bar(score: int) -> str:
        filled = max(0, min(10, score // 10))
        return "[" + "#" * filled + " " * (10 - filled) + f"] {score}/100"

    print("\n" + "=" * 60)
    print("Valutazione simbolica (Lux Index)")
    print("=" * 60)
    print(f"ego   ≈ {bar(metrics['ego'])}")
    print(f"gioia ≈ {bar(metrics['gioia'])}")
    print(f"rumore≈ {bar(metrics['rumore'])}")

    if comments:
        print("\nNote interpretative:")
        for c in comments:
            print(f"- {c}")

    if args.ai_openai:
        generate_ai_commentary(report_text, args.ai_model)


if __name__ == "__main__":
    main()

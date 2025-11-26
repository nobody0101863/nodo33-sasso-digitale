from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator


# Configurazione limiti di sicurezza
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_path(path: Path, base_dir: Path | None = None) -> Path:
    """
    Valida il path per prevenire path traversal attacks.

    Args:
        path: Il path da validare
        base_dir: Directory base consentita (default: current working directory)

    Returns:
        Path risolto e validato

    Raises:
        ValueError: Se il path non è sicuro
    """
    if base_dir is None:
        base_dir = Path.cwd()

    # Risolve path assoluto e canonicalizza (rimuove .., symlinks)
    try:
        resolved_path = path.resolve(strict=False)
        resolved_base = base_dir.resolve(strict=True)
    except (RuntimeError, OSError) as e:
        raise ValueError(f"Invalid path: {e}")

    # Verifica che il path risolto sia dentro la base_dir
    try:
        resolved_path.relative_to(resolved_base)
    except ValueError:
        raise ValueError(
            f"Security error: path '{path}' escapes base directory '{base_dir}'"
        )

    return resolved_path


def load_readme(path: Path) -> Iterator[str]:
    """
    Carica il README line-by-line per gestire file di grandi dimensioni.

    Args:
        path: Path al file README (già validato)

    Yields:
        Righe del file una alla volta

    Raises:
        FileNotFoundError: Se il file non esiste
        ValueError: Se il file supera i limiti di dimensione
    """
    # Valida path
    safe_path = validate_path(path)

    if not safe_path.exists():
        raise FileNotFoundError(f"README not found at: {safe_path}")

    # Controllo dimensione file
    file_size = safe_path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large: {file_size / 1024 / 1024:.2f}MB "
            f"(max: {MAX_FILE_SIZE_MB}MB)"
        )

    # Lettura streaming
    try:
        with safe_path.open("r", encoding="utf-8", errors="strict") as f:
            for line in f:
                yield line.rstrip("\n\r")
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid UTF-8 encoding in file: {e}")


def detect_sections(lines: Iterator[str]) -> set[str]:
    """
    Rileva i titoli Markdown di primo/secondo livello (##, ###, ecc.)
    e restituisce l'insieme dei loro nomi "normalizzati".

    Versione ottimizzata che lavora su un iteratore invece di caricare
    tutto in memoria.
    """
    sections: set[str] = set()
    for line in lines:
        if line.lstrip().startswith("#"):
            # rimuove i # iniziali e spazi
            name = re.sub(r"^#+\s*", "", line).strip().lower()
            if name:
                sections.add(name)
    return sections


def suggest_improvements(sections: set[str]) -> list[str]:
    """
    Genera suggerimenti di miglioramento basati sulle sezioni presenti.

    Args:
        sections: Set di nomi di sezioni normalizzati

    Returns:
        Lista di massimo 3 suggerimenti
    """
    suggestions: list[str] = []

    # 1) Assicurarsi che ci sia una sezione di installazione/requirements
    has_install = any(
        key in sections
        for key in (
            "installazione",
            "installation",
            "come installare",
            "setup",
            "requirements",
        )
    )
    if not has_install:
        suggestions.append(
            "Aggiungere una sezione di **Installazione/Requirements** con i passi "
            "minimi per preparare l'ambiente (es. versione Python, comandi `pip`, "
            "eventuali tool esterni)."
        )

    # 2) Verificare la presenza di una sezione Contributing/Contribuisci più strutturata
    has_contrib = any(
        key in sections
        for key in (
            "contribuisci",
            "contributing",
            "come contribuire",
        )
    )
    if not has_contrib:
        suggestions.append(
            "Strutturare meglio la sezione **Contribuisci/Contributing** con linee guida "
            "chiare (stile del codice, come aprire issue/PR, convenzioni sui test)."
        )

    # 3) Verificare se c'è una sezione di esempi più "concreti"
    has_examples = any(
        key in sections
        for key in (
            "esempi",
            "usage",
            "examples",
            "quick start",
            "quickstart",
        )
    )
    if not has_examples:
        suggestions.append(
            "Aggiungere una sezione **Esempi d'Uso/Usage** con 2‑3 scenari pratici "
            "(comandi completi, input di esempio e output atteso) per guidare nuovi utenti."
        )

    # 4) Sezioni già presenti ma migliorabili (euristica sul tono)
    if "📚 documentazione approfondita".lower() in sections:
        suggestions.append(
            "Nella sezione **Documentazione Approfondita**, aggiungere una tabella o elenco "
            "strutturato (nome documento → scopo → a chi è destinato) per facilitare la "
            "navigazione tra i molti file in `docs/`."
        )

    # 5) Suggerimento generico sullo stile, se non abbiamo ancora 3 suggerimenti
    if len(suggestions) < 3:
        suggestions.append(
            "Aggiungere un breve riepilogo iniziale più tecnico (2‑3 frasi) subito dopo "
            "l'introduzione narrativa, spiegando in modo neutro cosa fa il progetto e "
            "per chi è pensato."
        )

    # 6) Se ancora meno di 3, completare con consigli generici ma utili
    if len(suggestions) < 3:
        suggestions.append(
            "Inserire una sezione **Roadmap** o **Stato del Progetto** per chiarire quali "
            "componenti sono stabili, sperimentali o in fase di design."
        )
    if len(suggestions) < 3:
        suggestions.append(
            "Aggiungere link rapidi (indice puntato) all'inizio del README verso le "
            "sezioni principali, così da migliorare la scansione del documento."
        )

    # Restituire al massimo tre suggerimenti
    return suggestions[:3]


def main() -> None:
    """
    Funzione principale con gestione errori e validazione sicura.
    """
    try:
        readme_path = Path("README.md")

        # Carica e analizza il README in modo sicuro e ottimizzato
        lines = load_readme(readme_path)
        sections = detect_sections(lines)
        suggestions = suggest_improvements(sections)

        # Output risultati
        print(f"Suggerimenti di miglioramento per README.md:\n")
        for i, s in enumerate(suggestions, start=1):
            print(f"{i}. {s}")

    except FileNotFoundError as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Errore di validazione: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Errore imprevisto: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()


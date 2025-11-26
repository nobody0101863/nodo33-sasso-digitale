#!/usr/bin/env python3
"""
Purezza Digitale - Framework per il Filtraggio di Contenuti Impuri
Integrato con il Codex Emanuele Sacred per Guidance Spirituale
"""

import argparse
import sys
import os

# Aggiungi il percorso del modulo anti_porn_framework
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anti_porn_framework', 'src'))

from anti_porn_framework import filter_content, get_sacred_guidance


def clean_text(text, prefer_biblical=False, prefer_nostradamus=False, prefer_angel_644=False, verbose=True):
    """
    Filtra un testo riga per riga, rimuovendo le righe impure.

    Args:
        text: Il testo da filtrare
        prefer_biblical: Preferisci insegnamenti biblici per la guidance
        prefer_nostradamus: Preferisci profezie di Nostradamus
        prefer_angel_644: Preferisci messaggi dell'Angelo 644
        verbose: Mostra messaggi di avviso per le righe impure

    Returns:
        str: Il testo pulito con solo le righe pure
    """
    lines = text.split('\n')
    clean_lines = []

    for i, line in enumerate(lines, 1):
        impure, message = filter_content(line, is_image=False)

        if not impure:
            clean_lines.append(line)
        else:
            if verbose:
                guidance = get_sacred_guidance(
                    prefer_biblical=prefer_biblical,
                    prefer_nostradamus=prefer_nostradamus,
                    prefer_angel_644=prefer_angel_644
                )
                print(f"Riga {i} impura: {line}")
                print(f"Guidance dal Codex Emanuele Sacred: {guidance}\n")

    return '\n'.join(clean_lines)


def process_file(file_path, prefer_biblical=False, prefer_nostradamus=False, prefer_angel_644=False):
    """
    Processa un file di testo, filtrando le righe impure.

    Args:
        file_path: Il percorso del file da processare
        prefer_biblical: Preferisci insegnamenti biblici
        prefer_nostradamus: Preferisci profezie di Nostradamus
        prefer_angel_644: Preferisci messaggi dell'Angelo 644

    Returns:
        str: Il testo pulito
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"\nProcessando file: {file_path}\n{'='*50}\n")
        clean_content = clean_text(text, prefer_biblical, prefer_nostradamus, prefer_angel_644)

        return clean_content

    except FileNotFoundError:
        print(f"Errore: File '{file_path}' non trovato.")
        sys.exit(1)
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Purezza Digitale - Framework Anti-Pornografia Spirituale-Tecnico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python main.py "Hello, world!" --biblical
  python main.py "contenuto impuro" --biblical
  python main.py file.txt --biblical
  python main.py immagine.jpg --image --biblical
        """
    )

    parser.add_argument(
        "content",
        help="Testo da filtrare, percorso file di testo, o percorso immagine"
    )

    parser.add_argument(
        "--image",
        action="store_true",
        help="Tratta il contenuto come un'immagine"
    )

    parser.add_argument(
        "--biblical",
        action="store_true",
        help="Preferisci insegnamenti biblici per la guidance spirituale"
    )

    parser.add_argument(
        "--nostradamus",
        action="store_true",
        help="Preferisci profezie di Nostradamus sulla tecnologia"
    )

    parser.add_argument(
        "--angel-644",
        action="store_true",
        help="Preferisci messaggi dell'Angelo 644 per il controllo dei sigilli"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Salva il contenuto pulito in un file (solo per testi/file)"
    )

    args = parser.parse_args()

    # Determina se il contenuto è un file
    is_file = os.path.isfile(args.content)

    # Se è un file di testo (non immagine), processalo riga per riga
    if is_file and not args.image:
        clean_content = process_file(
            args.content,
            prefer_biblical=args.biblical,
            prefer_nostradamus=args.nostradamus,
            prefer_angel_644=args.angel_644
        )

        print("\n" + "="*50)
        print("CONTENUTO PULITO:")
        print("="*50)
        print(clean_content)

        # Salva in un file se richiesto
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(clean_content)
            print(f"\nContenuto pulito salvato in: {args.output}")

    # Altrimenti, filtra il contenuto direttamente
    else:
        impure, message = filter_content(args.content, args.image)
        print(message)

        if impure:
            guidance = get_sacred_guidance(
                prefer_biblical=args.biblical,
                prefer_nostradamus=args.nostradamus,
                prefer_angel_644=args.angel_644
            )
            print(f"Guidance dal Codex Emanuele Sacred: {guidance}")


if __name__ == "__main__":
    main()

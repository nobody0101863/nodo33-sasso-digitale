# main.py: Integrated framework orchestrator

import argparse
from .purezza_digitale import filter_content
from .sacred_codex import get_sacred_guidance

def main():
    parser = argparse.ArgumentParser(description="Spiritual-Technical Anti-Porn Framework")
    parser.add_argument("content", help="Testo o path immagine da filtrare")
    parser.add_argument("--image", action="store_true", help="Tratta come immagine")
    parser.add_argument("--biblical", action="store_true", help="Preferisci insegnamenti biblici")
    parser.add_argument("--nostradamus", action="store_true", help="Preferisci profezie di Nostradamus su tech")
    parser.add_argument("--angel-644", action="store_true", help="Preferisci guidance da angel number 644 per controllo sigilli")
    args = parser.parse_args()

    impure, message = filter_content(args.content, args.image)
    print(message)
    if impure:
        guidance = get_sacred_guidance(prefer_biblical=args.biblical, prefer_nostradamus=args.nostradamus, prefer_angel_644=args.angel_644)
        print("Guidance dal Codex Emanuele Sacred:", guidance)

if __name__ == "__main__":
    main()

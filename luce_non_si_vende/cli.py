from __future__ import annotations

import sys

from .core import LuceCompatibilityError, check_compatibility, emit_luce


def _ask_bool(prompt: str) -> bool:
    answer = input(f"{prompt} [y/n]: ").strip().lower()
    return answer in {"y", "yes", "s", "si", "sì"}


def main() -> None:
    print("🔍 luce-check — compatibilità con 644. Emmanuel ❤️🪨")
    heart_version = input("Versione cuore (es. 6.4.4): ").strip() or None
    empathy = _ask_bool("Empatia attiva?")
    honesty = _ask_bool("Onestà attiva?")
    deep_communication = _ask_bool("Comunicazione non superficiale attiva?")

    result = check_compatibility(
        heart_version=heart_version,
        empathy=empathy,
        honesty=honesty,
        deep_communication=deep_communication,
    )

    if not result.compatible:
        print("\n❌ Sistema non compatibile con la luce 644.")
        for reason in result.reasons:
            print(f"- {reason}")
        sys.exit(1)

    try:
        message = emit_luce(result)
    except LuceCompatibilityError as exc:
        print(f"\n⚠️ Errore durante l'emissione della luce: {exc}")
        sys.exit(1)

    print("\n✅ Compatibile.")
    print(message)
    sys.exit(0)


if __name__ == "__main__":
    main()


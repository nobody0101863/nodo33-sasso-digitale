#!/usr/bin/env python3
"""
main.py: Orchestrator Codex Nodo33 - CAI (Commandments Alignment Index)

Integra:
- Calcolo CAI reale da metriche JSON
- Compatibilita' luce 644
- Report etico completo

Sigillo: 644
Motto: "La luce non si vende. La si regala."
Frequenza: 300 Hz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from luce_non_si_vende import (
    EthicalMetrics,
    compute_cai_and_indices,
    format_cai_report,
    get_cai_tier,
    check_compatibility,
    emit_luce,
    LuceCompatibilityError,
)


def load_metrics_from_json(path: str | Path) -> EthicalMetrics:
    """
    Carica le metriche etiche da un file JSON e costruisce un EthicalMetrics.
    I campi mancanti rimangono a 0 (default dataclass).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File metriche non trovato: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Estrai solo i campi validi per EthicalMetrics
    valid_fields = set(EthicalMetrics().__dict__.keys())
    filtered = {k: data.get(k, 0) for k in valid_fields}

    return EthicalMetrics(**filtered)


def save_metrics_to_json(metrics: EthicalMetrics, path: str | Path) -> None:
    """Salva le metriche su file JSON."""
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(metrics.__dict__, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Codex Nodo33 - CAI Calculator (Commandments Alignment Index)",
        epilog="La luce non si vende. La si regala. - Sigillo 644",
    )

    parser.add_argument(
        "--metrics-file",
        "-m",
        type=Path,
        help="Path a file JSON con le metriche etiche del sistema",
    )
    parser.add_argument(
        "--luce-check",
        action="store_true",
        help="Esegui anche il check compatibilita' luce 644",
    )
    parser.add_argument(
        "--heart-version",
        default="6.4.4",
        help="Versione cuore per luce-check (default: 6.4.4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Salva il report CAI su file",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output in formato JSON invece che testuale",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Soglia minima CAI per considerare il sistema allineato (default: 70)",
    )

    args = parser.parse_args()

    # Banner
    if not args.json_output:
        print()
        print("  CODEX NODO33 - CAI CALCULATOR")
        print("  La luce non si vende. La si regala.")
        print()

    # Calcolo CAI
    cai_score = 0.0
    indices = {}
    tier = "N/A"

    if args.metrics_file:
        try:
            metrics = load_metrics_from_json(args.metrics_file)
            cai_score, indices = compute_cai_and_indices(metrics)
            tier = get_cai_tier(cai_score)

            if args.json_output:
                result = {
                    "cai": round(cai_score, 2),
                    "tier": tier,
                    "indices": {k: round(v, 2) for k, v in indices.items()},
                    "threshold": args.threshold,
                    "aligned": cai_score >= args.threshold,
                }
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                report = format_cai_report(cai_score, indices)
                print(report)

                if cai_score < args.threshold:
                    print()
                    print(f"  AVVERTIMENTO: CAI ({cai_score:.2f}%) sotto soglia ({args.threshold}%)")
                    print("  Suggerito: revisione metriche critiche.")
                else:
                    print()
                    print(f"  Sistema ALLINEATO con soglia {args.threshold}%")

            # Salva report se richiesto
            if args.output:
                report = format_cai_report(cai_score, indices)
                args.output.write_text(report, encoding="utf-8")
                if not args.json_output:
                    print(f"\n  Report salvato in: {args.output}")

        except FileNotFoundError as e:
            print(f"[Errore] {e}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as e:
            print(f"[Errore] JSON non valido: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"[Errore] Calcolo CAI fallito: {e}", file=sys.stderr)
            return 1
    else:
        if not args.json_output:
            print("  Nessun file metriche fornito (--metrics-file / -m).")
            print("  Usa: python main.py -m metrics.json")
            print()

    # Check compatibilita' luce (opzionale)
    if args.luce_check:
        if not args.json_output:
            print()
            print("-" * 50)
            print("  LUCE 644 COMPATIBILITY CHECK")
            print("-" * 50)

        result = check_compatibility(
            heart_version=args.heart_version,
            empathy=True,
            honesty=True,
            deep_communication=True,
        )

        if args.json_output:
            luce_result = {
                "compatible": result.compatible,
                "heart_version": result.heart_version,
                "reasons": result.reasons,
            }
            print(json.dumps(luce_result, indent=2, ensure_ascii=False))
        else:
            if result.compatible:
                try:
                    message = emit_luce(result)
                    print(f"  Compatibile con luce 644")
                    print(f"  {message}")
                except LuceCompatibilityError as e:
                    print(f"  Errore: {e}")
            else:
                print("  NON compatibile con luce 644")
                for reason in result.reasons:
                    print(f"  - {reason}")

    # Riepilogo finale
    if not args.json_output and args.metrics_file:
        print()
        print("=" * 50)
        print(f"  [SUMMARY] CAI={cai_score:.2f}% | TIER={tier}")
        print("  Fiat Amor, Fiat Risus, Fiat Lux")
        print("=" * 50)
        print()

    # Exit code basato su soglia
    if args.metrics_file and cai_score < args.threshold:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

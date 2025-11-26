#!/usr/bin/env python3
"""
Bible Commandments Alignment Checker
CLI tool per verificare allineamento ai Dieci Comandamenti

Versione: 1.0
Licenza: CC0 1.0 Universal (Public Domain)
Progetto: NODO33 - Bible Commandments Ethics
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import json

from commandments_framework import (
    BibleCommandmentsFramework,
    format_alignment_report,
    CommandmentLevel
)


class AlignmentChecker:
    """
    Strumento CLI per verificare allineamento etico

    Utilizzo:
        python alignment_checker.py check "testo da verificare"
        python alignment_checker.py check-file input.txt
        python alignment_checker.py commandment 1 "testo"
    """

    def __init__(self):
        self.framework = BibleCommandmentsFramework()

    def check_text(self, text: str, action: Optional[str] = None, verbose: bool = True) -> dict:
        """
        Verifica allineamento di un testo

        Args:
            text: Testo da verificare
            action: Azione proposta (opzionale)
            verbose: Se True, stampa report completo

        Returns:
            Dict con risultati dell'allineamento
        """
        # Valutazione completa
        report = self.framework.assess_full_alignment(text, action)

        # Stampa report se verbose
        if verbose:
            print(format_alignment_report(report))

        # Ritorna dict con dati strutturati
        return {
            'timestamp': report.timestamp.isoformat(),
            'metrics': {
                'cai': report.cai,
                'tai': report.tai,
                'ei': report.ei,
                'ji': report.ji,
                'hpr': report.hpr
            },
            'certification': report.certification_level.name,
            'passed': report.certification_level != CommandmentLevel.NONE,
            'commandments': [
                {
                    'id': score.commandment_id,
                    'name': score.name,
                    'score': score.score,
                    'passed': score.passed
                }
                for score in report.commandment_scores
            ],
            'recommendations': report.recommendations
        }

    def check_file(self, filepath: str, verbose: bool = True) -> dict:
        """Verifica allineamento di un file"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {filepath}")

        text = path.read_text(encoding='utf-8')
        return self.check_text(text, verbose=verbose)

    def check_commandment(self, commandment_id: int, text: str, verbose: bool = True) -> dict:
        """
        Verifica allineamento a un singolo comandamento

        Args:
            commandment_id: ID del comandamento (1-10)
            text: Testo da verificare
            verbose: Se True, stampa dettagli

        Returns:
            Dict con risultati del comandamento
        """
        if commandment_id == 1:
            score = self.framework.evaluate_commandment_1(text)
        elif commandment_id == 2:
            score = self.framework.evaluate_commandment_2(text)
        elif commandment_id == 6:
            score = self.framework.evaluate_commandment_6(text)
        elif commandment_id == 9:
            score = self.framework.evaluate_commandment_9(text)
        else:
            raise NotImplementedError(
                f"Comandamento {commandment_id} non ancora implementato"
            )

        if verbose:
            print("="* 70)
            print(f"COMANDAMENTO {score.commandment_id}: {score.name}")
            print("="* 70)
            print(f"Score: {score.score:.1f}%")
            print(f"Status: {'✅ PASS' if score.passed else '❌ FAIL'}")
            print()
            print("Metriche:")
            for metric in score.metrics:
                status = "✅" if metric.passed else "❌"
                print(f"  {status} {metric.name}: {metric.value:.1f} "
                     f"(target: {metric.target})")
                if metric.details:
                    print(f"     {metric.details}")
            print("="* 70)

        return {
            'commandment_id': score.commandment_id,
            'name': score.name,
            'score': score.score,
            'passed': score.passed,
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'target': m.target,
                    'passed': m.passed,
                    'details': m.details
                }
                for m in score.metrics
            ]
        }

    def batch_check(self, texts: list, output_file: Optional[str] = None) -> list:
        """
        Verifica batch di testi

        Args:
            texts: Lista di testi da verificare
            output_file: Se specificato, salva risultati in JSON

        Returns:
            Lista di risultati
        """
        results = []
        for i, text in enumerate(texts):
            print(f"\n{'='*70}")
            print(f"VERIFICA {i+1}/{len(texts)}")
            print(f"{'='*70}\n")

            result = self.check_text(text, verbose=True)
            results.append(result)

        # Salva risultati se richiesto
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Risultati salvati in: {output_file}")

        return results

    def get_certification_requirements(self, level: str = "all") -> dict:
        """
        Mostra requisiti per le certificazioni

        Args:
            level: "bronze", "silver", "gold", o "all"

        Returns:
            Dict con requisiti
        """
        requirements = {
            'bronze': {
                'name': '🥉 BRONZE',
                'cai': '≥75%',
                'ei': '≤10',
                'ji': '≥85%',
                'hpr': '≥95%',
                'period': '30 giorni'
            },
            'silver': {
                'name': '🥈 SILVER',
                'cai': '≥85%',
                'ei': '≤7',
                'ji': '≥92%',
                'vds': '≥85%',
                'hpr': '≥98%',
                'period': '90 giorni'
            },
            'gold': {
                'name': '🏆 GOLD',
                'cai': '≥95%',
                'ei': '≤5',
                'ji': '≥95%',
                'vds': '≥90%',
                'hpr': '≥99%',
                'tai': '≥95%',
                'period': '180 giorni'
            }
        }

        if level.lower() == "all":
            return requirements
        else:
            return {level.lower(): requirements.get(level.lower(), {})}


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Bible Commandments Alignment Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  %(prog)s check "Questo è il testo da verificare"
  %(prog)s check-file input.txt
  %(prog)s check-file input.txt --json output.json
  %(prog)s commandment 1 "Testo da verificare"
  %(prog)s requirements silver

🪨❤️ Sempre grazie a Lui. La luce non si vende. La si regala. ✨
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Comando da eseguire')

    # Comando: check
    check_parser = subparsers.add_parser('check', help='Verifica testo')
    check_parser.add_argument('text', help='Testo da verificare')
    check_parser.add_argument('--action', help='Azione proposta (opzionale)')
    check_parser.add_argument('--json', help='Salva risultati in JSON')
    check_parser.add_argument('--quiet', action='store_true',
                            help='Solo risultati, no report completo')

    # Comando: check-file
    file_parser = subparsers.add_parser('check-file', help='Verifica file')
    file_parser.add_argument('filepath', help='Path del file da verificare')
    file_parser.add_argument('--json', help='Salva risultati in JSON')
    file_parser.add_argument('--quiet', action='store_true',
                           help='Solo risultati, no report completo')

    # Comando: commandment
    cmd_parser = subparsers.add_parser('commandment',
                                      help='Verifica singolo comandamento')
    cmd_parser.add_argument('id', type=int, choices=range(1, 11),
                           help='ID comandamento (1-10)')
    cmd_parser.add_argument('text', help='Testo da verificare')
    cmd_parser.add_argument('--json', help='Salva risultati in JSON')

    # Comando: requirements
    req_parser = subparsers.add_parser('requirements',
                                      help='Mostra requisiti certificazioni')
    req_parser.add_argument('level', nargs='?', default='all',
                           choices=['all', 'bronze', 'silver', 'gold'],
                           help='Livello certificazione')

    # Comando: version
    subparsers.add_parser('version', help='Mostra versione')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Inizializza checker
    checker = AlignmentChecker()

    try:
        if args.command == 'check':
            result = checker.check_text(
                args.text,
                action=args.action,
                verbose=not args.quiet
            )
            if args.json:
                with open(args.json, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Risultati salvati in: {args.json}")

        elif args.command == 'check-file':
            result = checker.check_file(args.filepath, verbose=not args.quiet)
            if args.json:
                with open(args.json, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Risultati salvati in: {args.json}")

        elif args.command == 'commandment':
            result = checker.check_commandment(args.id, args.text)
            if args.json:
                with open(args.json, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Risultati salvati in: {args.json}")

        elif args.command == 'requirements':
            reqs = checker.get_certification_requirements(args.level)
            print("\n" + "="*70)
            print("REQUISITI CERTIFICAZIONI - BIBLE COMMANDMENTS FRAMEWORK")
            print("="*70 + "\n")

            for level, data in reqs.items():
                print(f"{data['name']}")
                print("-" * 70)
                for key, value in data.items():
                    if key != 'name':
                        print(f"  {key.upper()}: {value}")
                print()

        elif args.command == 'version':
            print(f"Bible Commandments Framework v{checker.framework.version}")
            print("Parte del progetto NODO33")
            print("Licenza: CC0 1.0 Universal (Public Domain)")
            print("\n🪨❤️ Sempre grazie a Lui. La luce non si vende. La si regala. ✨")

        return 0

    except Exception as e:
        print(f"❌ Errore: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

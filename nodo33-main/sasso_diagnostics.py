#!/usr/bin/env python3
"""
Sasso Diagnostics - Quick environment report for Progetto Sasso Digitale.

Scopo:
- Verificare quali "munizioni" e tool sono disponibili sul sistema
- Aiutare a capire cosa funziona subito e cosa richiede setup extra

Puoi eseguirlo dalla root del repo con:
    python3 sasso_diagnostics.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class CheckResult:
    name: str
    available: bool
    detail: Optional[str] = None


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_python_runtime() -> List[CheckResult]:
    results: List[CheckResult] = []

    version = sys.version.split()[0]
    results.append(CheckResult(name="Python", available=True, detail=f"Version {version}"))

    # Core Python libs for this repo
    for module in ("sqlite3", "subprocess", "pathlib"):
        try:
            __import__(module)
            results.append(CheckResult(name=f"stdlib:{module}", available=True))
        except Exception as exc:
            results.append(CheckResult(name=f"stdlib:{module}", available=False, detail=str(exc)))

    return results


def check_python_packages() -> List[CheckResult]:
    results: List[CheckResult] = []

    def try_import(module: str, extra_path: Optional[Path] = None, label: Optional[str] = None) -> None:
        name = label or module
        try:
            if extra_path is not None:
                if str(extra_path) not in sys.path:
                    sys.path.insert(0, str(extra_path))
            __import__(module)
            results.append(CheckResult(name=name, available=True))
        except Exception as exc:
            results.append(CheckResult(name=name, available=False, detail=str(exc)))

    # Local packages
    anti_porn_src = REPO_ROOT / "anti_porn_framework" / "src"
    try_import("anti_porn_framework", extra_path=anti_porn_src, label="anti_porn_framework (src)")

    src_dir = REPO_ROOT / "src"
    try_import("stones_speaking", extra_path=src_dir, label="stones_speaking (src)")

    lux_dir = REPO_ROOT / "lux-ai-privacy-policy"
    if lux_dir.is_dir():
        try_import("lux_ai", extra_path=lux_dir, label="lux_ai (lux-ai-privacy-policy)")
    else:
        results.append(
            CheckResult(
                name="lux-ai-privacy-policy directory",
                available=False,
                detail="Directory non trovata",
            )
        )

    # Optional heavy dependencies used by codex_server image pipeline
    for pkg in ("torch", "diffusers", "transformers", "accelerate", "safetensors"):
        try:
            __import__(pkg)
            results.append(CheckResult(name=pkg, available=True))
        except Exception as exc:
            results.append(CheckResult(name=pkg, available=False, detail=str(exc)))

    return results


def check_cli_tools() -> List[CheckResult]:
    tools = [
        ("rustc", "Rust compiler (per GIOIA_100.rs)"),
        ("cargo", "Rust package manager"),
        ("go", "Go toolchain (per SASSO_API.go)"),
        ("swift", "Swift toolchain (per EGO_ZERO.swift)"),
        ("kotlinc", "Kotlin compiler (per SASSO.kt)"),
        ("java", "Java runtime (per jar Kotlin)"),
        ("ruby", "Ruby (per sasso.rb)"),
        ("php", "PHP (per sasso.php)"),
        ("nasm", "Netwide Assembler (per sasso.asm)"),
    ]

    results: List[CheckResult] = []
    for name, description in tools:
        path = shutil.which(name)
        if path:
            results.append(CheckResult(name=name, available=True, detail=f"Found at {path} - {description}"))
        else:
            results.append(CheckResult(name=name, available=False, detail=description))
    return results


def summarize(results: List[CheckResult]) -> Dict[str, int]:
    total = len(results)
    available = sum(1 for r in results if r.available)
    return {"total": total, "available": available, "missing": total - available}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sasso Diagnostics - environment checker")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Mostra solo un riepilogo per sezione senza dettagli per ogni voce",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SASSO DIAGNOSTICS - ENVIRONMENT CHECK")
    print("=" * 70)
    print(f"Repo root: {REPO_ROOT}")

    # Python runtime
    _print_header("Python runtime")
    py_results = check_python_runtime()
    if args.summary_only:
        py_summary = summarize(py_results)
        print(
            f"Runtime: {py_summary['available']}/{py_summary['total']} OK "
            f"(missing: {py_summary['missing']})"
        )
    else:
        for r in py_results:
            status = "OK" if r.available else "MISSING"
            detail = f" - {r.detail}" if r.detail else ""
            print(f"[{status:7}] {r.name}{detail}")

    # Python packages
    _print_header("Python packages (local + optional)")
    pkg_results = check_python_packages()
    if args.summary_only:
        pkg_summary = summarize(pkg_results)
        print(
            f"Packages: {pkg_summary['available']}/{pkg_summary['total']} OK "
            f"(missing: {pkg_summary['missing']})"
        )
    else:
        for r in pkg_results:
            status = "OK" if r.available else "MISSING"
            detail = f" - {r.detail}" if r.detail else ""
            print(f"[{status:7}] {r.name}{detail}")

    # CLI tools
    _print_header("Command-line tools for munitions")
    tool_results = check_cli_tools()
    if args.summary_only:
        tool_summary = summarize(tool_results)
        print(
            f"CLI tools: {tool_summary['available']}/{tool_summary['total']} OK "
            f"(missing: {tool_summary['missing']})"
        )
    else:
        for r in tool_results:
            status = "OK" if r.available else "MISSING"
            detail = f" - {r.detail}" if r.detail else ""
            print(f"[{status:7}] {r.name}{detail}")

    all_results = py_results + pkg_results + tool_results
    summary = summarize(all_results)

    _print_header("Summary")
    print(f"Checks run : {summary['total']}")
    print(f"Available  : {summary['available']}")
    print(f"Missing    : {summary['missing']}")
    print("\nNota:")
    print("- Non tutto deve essere installato per usare il progetto.")
    print("- Usa questo report per decidere quali munitions vuoi attivare e quali tool installare.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Paper Analyzer - Analisi Automatica di Paper Scientifici
Legge PDF di paper e produce:
- Riassunto esecutivo
- Metodologia
- Risultati chiave
- Critiche
"""

import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama

def analizza_paper(pdf_path: str):
    """Analizza un paper scientifico"""

    print("📄 PAPER ANALYZER - NODO33")
    print("=" * 70)
    print(f"Paper: {Path(pdf_path).name}\n")

    # Carica PDF
    print("📖 Lettura PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    testo_completo = "\n\n".join([p.page_content for p in pages])

    print(f"   Lette {len(pages)} pagine\n")

    # LLM
    llm = Ollama(model="llama3.1:8b", base_url="http://localhost:11434")

    # 1. Riassunto esecutivo
    print("🎯 RIASSUNTO ESECUTIVO")
    print("-" * 70)
    prompt_riassunto = f"""Sei un ricercatore esperto. Leggi questo paper e scrivi un riassunto esecutivo in 3-5 punti.

Paper:
{testo_completo[:4000]}  # Primi 4000 caratteri

Riassunto (bullet points):"""

    riassunto = llm.invoke(prompt_riassunto)
    print(riassunto)

    # 2. Metodologia
    print("\n\n🔬 METODOLOGIA")
    print("-" * 70)
    prompt_metodo = f"""Dal seguente paper, estrai e spiega la metodologia usata.

Paper:
{testo_completo[:4000]}

Metodologia (chiara e sintetica):"""

    metodologia = llm.invoke(prompt_metodo)
    print(metodologia)

    # 3. Risultati chiave
    print("\n\n📊 RISULTATI CHIAVE")
    print("-" * 70)
    prompt_risultati = f"""Dal paper, estrai i 3 risultati più importanti.

Paper:
{testo_completo[:4000]}

Risultati (3 punti):"""

    risultati = llm.invoke(prompt_risultati)
    print(risultati)

    # 4. Analisi critica
    print("\n\n🤔 ANALISI CRITICA")
    print("-" * 70)
    prompt_critica = f"""Sei un reviewer critico ma costruttivo. Analizza punti di forza e debolezza di questo paper.

Paper:
{testo_completo[:4000]}

Analisi critica:"""

    critica = llm.invoke(prompt_critica)
    print(critica)

    print("\n\n" + "=" * 70)
    print("✅ Analisi completata!")
    print("🪨 'La scienza è dubbio, non certezza'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python paper_analyzer.py paper.pdf")
        print("\nEsempio:")
        print("  python paper_analyzer.py attention_is_all_you_need.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"❌ File non trovato: {pdf_path}")
        sys.exit(1)

    analizza_paper(pdf_path)

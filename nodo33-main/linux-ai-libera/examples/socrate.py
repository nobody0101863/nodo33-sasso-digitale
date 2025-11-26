#!/usr/bin/env python3
"""
Socrate AI - Dialogo Socratico Automatico
Usa il metodo maieutico per esplorare un concetto
"""

import sys
from langchain_community.llms import Ollama

def dialogo_socratico(concetto: str, num_domande=5):
    """Inizia un dialogo socratico su un concetto"""

    llm = Ollama(model="llama3.1:8b", base_url="http://localhost:11434")

    print("🏛️ DIALOGO SOCRATICO")
    print("=" * 70)
    print(f"Concetto iniziale: {concetto}\n")

    conversazione = f"Tu: {concetto}"

    for i in range(num_domande):
        print(f"🤔 Socrate riflette... (domanda {i+1}/{num_domande})")

        prompt = f"""Sei Socrate. Stai conducendo un dialogo maieutico.

Conversazione finora:
{conversazione}

Poni UNA SOLA domanda profonda che:
1. Metta in dubbio un assunto del concetto
2. Porti verso una comprensione più profonda
3. Sia breve (max 2 frasi)

Domanda:"""

        domanda = llm.invoke(prompt).strip()
        print(f"\n💬 Socrate: {domanda}\n")
        conversazione += f"\n\nSocrate: {domanda}"

        # Risposta automatica (puoi rendere interattivo)
        prompt_risposta = f"""Conversazione:
{conversazione}

Rispondi alla domanda di Socrate in modo riflessivo ma onesto.
Massimo 3 frasi.

Risposta:"""

        risposta = llm.invoke(prompt_risposta).strip()
        print(f"💭 Tu: {risposta}\n")
        print("-" * 70)
        conversazione += f"\nTu: {risposta}"

    # Sintesi finale
    print("\n📜 SINTESI DEL DIALOGO")
    print("=" * 70)

    prompt_sintesi = f"""Questo è stato il dialogo socratico:
{conversazione}

Scrivi una sintesi poetica in 3 punti di cosa abbiamo scoperto."""

    sintesi = llm.invoke(prompt_sintesi)
    print(sintesi)
    print("\n🏛️ 'Conosci te stesso'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python socrate.py 'concetto da esplorare'")
        print("\nEsempi:")
        print("  python socrate.py 'la giustizia'")
        print("  python socrate.py 'il coraggio'")
        sys.exit(1)

    concetto = " ".join(sys.argv[1:])
    dialogo_socratico(concetto)

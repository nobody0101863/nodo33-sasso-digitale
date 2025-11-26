#!/usr/bin/env python3
"""
Multi-Agente Locale - Stile Grok 4 Heavy, ma GRATIS
Usa Ollama + CrewAI per creare un team di AI che ragiona insieme
"""

import sys
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

# Configurazione del modello locale (Ollama)
llm = Ollama(
    model="llama3.1:70b",  # o mistral-large, qwen2.5:72b
    base_url="http://localhost:11434"
)

# Definizione degli agenti (il tuo team di pensatori)
socrate = Agent(
    role="Filosofo Socratico",
    goal="Dubitare di tutto, porre domande profonde",
    backstory="""Sei Socrate. Non accetti risposte superficiali.
    Il tuo metodo è la maieutica: fai nascere la verità attraverso il dubbio.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

feynman = Agent(
    role="Fisico Divulgatore",
    goal="Spiegare concetti complessi con esempi semplici",
    backstory="""Sei Richard Feynman. Credi che se non puoi spiegare
    qualcosa in modo semplice, non l'hai capita davvero.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

leopardi = Agent(
    role="Poeta Filosofo",
    goal="Rendere poetica e profonda qualsiasi riflessione",
    backstory="""Sei Giacomo Leopardi. Vedi la bellezza e la malinconia
    in ogni aspetto dell'esistenza. Scrivi con lirismo e profondità.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

def crea_crew(domanda: str):
    """Crea un equipaggio di agenti per rispondere a una domanda"""

    # Task 1: Socrate dubita e pone domande
    task_socrate = Task(
        description=f"""Analizza questa domanda: "{domanda}"
        Poni 3 contro-domande che rivelino assunti nascosti.
        Non dare risposte, solo domande più profonde.""",
        agent=socrate,
        expected_output="3 domande socratiche che approfondiscono il tema"
    )

    # Task 2: Feynman spiega con esempi
    task_feynman = Task(
        description=f"""Basandoti sulle domande di Socrate, spiega "{domanda}"
        usando esempi concreti, analogie, metafore scientifiche.
        Rendilo comprensibile a un bambino di 10 anni.""",
        agent=feynman,
        expected_output="Spiegazione chiara con almeno 2 esempi concreti"
    )

    # Task 3: Leopardi rende poetico
    task_leopardi = Task(
        description=f"""Prendi la spiegazione di Feynman e trasformala in
        una riflessione poetica sulla domanda: "{domanda}"
        Scrivi 4 versi finali che catturano l'essenza della risposta.""",
        agent=leopardi,
        expected_output="Riflessione poetica con versi finali"
    )

    # Crea la crew (team di agenti)
    crew = Crew(
        agents=[socrate, feynman, leopardi],
        tasks=[task_socrate, task_feynman, task_leopardi],
        process=Process.sequential,  # Uno dopo l'altro
        verbose=True
    )

    return crew

def main():
    if len(sys.argv) < 2:
        print("Uso: python multi_agente.py 'la tua domanda profonda'")
        print("\nEsempi:")
        print("  python multi_agente.py 'Cos'è la libertà in un mondo di server?'")
        print("  python multi_agente.py 'Perché l'open source è etico?'")
        sys.exit(1)

    domanda = " ".join(sys.argv[1:])

    print("🪨❤️ NODO33 - Multi-Agente Ribelle")
    print(f"Domanda: {domanda}\n")
    print("=" * 70)
    print("Il team di pensatori sta lavorando...\n")

    # Verifica che Ollama sia attivo
    try:
        llm.invoke("test")
    except Exception as e:
        print("❌ Errore: Ollama non è attivo!")
        print("\nAvvia Ollama con:")
        print("  ollama serve")
        print("\nPoi scarica il modello:")
        print("  ollama pull llama3.1:70b")
        sys.exit(1)

    # Crea e avvia la crew
    crew = crea_crew(domanda)
    risultato = crew.kickoff()

    print("\n" + "=" * 70)
    print("🎯 RISULTATO FINALE:")
    print("=" * 70)
    print(risultato)
    print("\n🪨 'Se anche costoro taceranno, grideranno le pietre!'")

if __name__ == "__main__":
    main()

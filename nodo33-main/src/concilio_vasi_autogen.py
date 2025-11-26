"""
Orchestratore multi‑agente per il Concilio dei Vasi basato su AutoGen.

Questo modulo NON aggiunge dipendenze obbligatorie:
- prova a importare `autogen` dinamicamente
- se non è installato, stampa un messaggio chiaro e termina.

Concetto:
- Vaso Scrivano    → agente che invoca `agente_scrivano` su un pensiero grezzo
- Vaso Verificatore→ agente che invoca `agente_verificatore` / `check_644_and_auth`
- Vaso Archivista  → agente che invoca `agente_archivista` / `pipeline_ingestione`

Nota: l'API AutoGen evolve nel tempo; questo file è uno skeleton concettuale
che puoi adattare alla versione specifica di AutoGen installata.
"""

from __future__ import annotations

from typing import Any, Dict

from concilio_vasi import (
    agente_scrivano,
    agente_verificatore,
    agente_archivista,
    pipeline_ingestione,
)


def _ensure_autogen():
    """Prova a importare AutoGen e restituisce il modulo, o None se mancante."""
    try:
        import autogen  # type: ignore
    except ImportError:
        print(
            "⚠️  AutoGen non è installato.\n"
            "    Installa con: pip install pyautogen\n"
            "    Oppure usa direttamente pipeline_ingestione(...) da concilio_vasi."
        )
        return None
    return autogen


def build_concilio_agents() -> Dict[str, Any]:
    """
    Costruisce gli agenti AutoGen per i tre Vasi.

    Ritorna un dict con:
    - 'scrivano'
    - 'verificatore'
    - 'archivista'
    - 'user_proxy' (per lanciare dialoghi di prova)
    """
    autogen = _ensure_autogen()
    if autogen is None:
        return {}

    # Config minima: si assume che la configurazione del modello
    # sia già gestita via env vars / config esterna.
    llm_config: Dict[str, Any] = {
        "config_list": [
            {
                "model": "gpt-4.1-mini",
                "api_key": "YOUR_API_KEY_HERE",
            }
        ]
    }

    AssistantAgent = autogen.AssistantAgent
    UserProxyAgent = autogen.UserProxyAgent

    # User/proxy: tu che parli al concilio
    user_proxy = UserProxyAgent(
        name="utente",
        human_input_mode="NEVER",  # per orchestrazioni programmatiche
    )

    # Vaso Scrivano: chiama solo la funzione Python agente_scrivano
    scrivano = AssistantAgent(
        name="scrivano",
        system_message=(
            "Sei il Vaso-Scrivano del Concilio. "
            "Ricevi un pensiero grezzo e devi solo invocare la funzione "
            "`agente_scrivano` per costruire un payload strutturato; "
            "non devi prendere decisioni etiche o scrivere su database."
        ),
        llm_config=llm_config,
        function_map={
            "agente_scrivano": agente_scrivano,
        },
    )

    # Vaso Verificatore: applica solo la chiave 644+auth
    verificatore = AssistantAgent(
        name="verificatore",
        system_message=(
            "Sei il Vaso-Verificatore. "
            "Applichi il principio 644 e l'autenticazione sui payload ricevuti, "
            "usando `agente_verificatore`; non puoi scrivere nel Sasso Digitale."
        ),
        llm_config=llm_config,
        function_map={
            "agente_verificatore": agente_verificatore,
        },
    )

    # Vaso Archivista: unico autorizzato a chiamare l'archiviazione
    archivista = AssistantAgent(
        name="archivista",
        system_message=(
            "Sei il Vaso-Archivista. "
            "Ricevi il payload e il report del Verificatore e puoi invocare "
            "`agente_archivista` oppure direttamente `pipeline_ingestione`. "
            "Scrivi nel Sasso Digitale solo se il Verificatore ha restituito AUTHENTICATED."
        ),
        llm_config=llm_config,
        function_map={
            "agente_archivista": agente_archivista,
            "pipeline_ingestione": pipeline_ingestione,
        },
    )

    return {
        "user_proxy": user_proxy,
        "scrivano": scrivano,
        "verificatore": verificatore,
        "archivista": archivista,
    }


def demo_autogen_concilio():
    """
    Demo concettuale di come potresti usare gli agenti AutoGen.

    Nota: questa funzione non avvia un vero group chat complesso per evitare
    di vincolarsi a una singola versione dell'API AutoGen. Serve come
    esempio di wiring tra i tre vasi.
    """
    agents = build_concilio_agents()
    if not agents:
        return

    # Esempio: uso diretto della pipeline di base (senza orchestrazione LLM)
    print("🔁 Demo pipeline_ingestione diretta (senza AutoGen LLM):\n")
    result = pipeline_ingestione(
        "La nuova ipotesi per il codice base è A2Z.",
        source="autogen_demo",
        allow=True,
    )
    print("Outcome:", result)

    print(
        "\nPer una orchestrazione completa via AutoGen, "
        "usa `build_concilio_agents()` e definisci un group-chat "
        "sequenziale: utente → scrivano → verificatore → archivista."
    )


if __name__ == "__main__":
    demo_autogen_concilio()


import importlib
import yaml
from typing import Dict, Any
from agents.base import AgentResult


def load_registry(path: str = "agents/registry.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_agent(agent_spec: Dict[str, Any]):
    module_path, class_name = agent_spec["module"].split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def main():
    registry = load_registry()
    agents = {a["id"]: a for a in registry["agents"]}

    print("Agenti disponibili:")
    for a_id, a in agents.items():
        print(f"- {a_id}: {a['name']} – {a['role']}")

    chosen = input("\nScegli un agente (id): ").strip()
    if chosen not in agents:
        print("Agente non trovato.")
        return

    agent_spec = agents[chosen]
    agent = load_agent(agent_spec)

    # Demo semplice: text input da terminale
    text = input("\nInserisci testo/url da analizzare: ")

    payload = {"text": text}
    result: AgentResult = agent.run(payload)

    print("\n=== RISULTATO ===")
    print("Livello:", result.level)
    print("Summary:", result.summary)
    print("Dettagli:", result.details)


if __name__ == "__main__":
    main()

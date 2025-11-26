#!/usr/bin/env python3
"""
orchestrator.py – Nodo33

L'orchestratore sceglie l'agente corretto per un compito
e applica le policy etiche del Codex.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from agent_map_builder import get_agent_config_for_orchestrator
from src.framework_angelic_guardian import AngelicGuardian
from orchestrator.guardian_service import guardian_scan as guardian_scan_service

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dipendenza opzionale
    OpenAI = None  # type: ignore[assignment]


class Nodo33Orchestrator:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model

        # Carica agenti e etica
        config = get_agent_config_for_orchestrator()
        self.families: Dict[str, Any] = config["families"]
        self.agents: Dict[str, Any] = config["flat"]
        self.ethics: Dict[str, Any] = config["ethics"]

        # Guardian (profilo globale)
        self.guardian = AngelicGuardian()
        self.guardian_enabled = os.environ.get("N33_GUARDIAN_ENABLE", "1") != "0"
        self.guardian_last_report: Dict[str, Any] | None = None

        # Memoria persistente
        self.memory_file = ".n33mem"
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self.memory: List[Dict[str, str]] = json.load(f)
            except Exception:
                self.memory = []
        else:
            self.memory = []

        # Client OpenAI solo se presente e configurato
        if OpenAI is not None and os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI()
        else:
            self.client = None

    # -----------------------------------------------------
    # 1) Classificazione avanzata del task
    # -----------------------------------------------------
    def classify_task(self, task: str) -> str:
        t = task.lower()

        if any(k in t for k in ("terra", "ambiente", "ecologia", "custodia")):
            return "core_purity_terra_pace"

        if any(k in t for k in ("commandment", "bib", "legge", "salmo", "mosè")):
            return "parallel_codex"

        if any(k in t for k in ("sasso", "radici", "rami", "pietra")):
            return "sasso_digitale"

        if any(k in t for k in ("diffondi", "adozione", "adozione etica", "memoria")):
            return "adoption_and_memory"

        # fallback universale
        return "core_purity_terra_pace"

    # -----------------------------------------------------
    # 2) Selezione agente + catena di supporto
    # -----------------------------------------------------
    def select_agents_chain(self, task: str) -> List[Dict[str, Any]]:
        family = self.classify_task(task)

        primary = self._first(family)
        secondary = self._first("adoption_and_memory")
        peace = self._first("core_purity_terra_pace")

        return [agent for agent in (primary, secondary, peace) if agent]

    def _first(self, family_name: str) -> Dict[str, Any] | None:
        prefix = f"{family_name}:"
        for key, profile in self.agents.items():
            if key.startswith(prefix):
                return {"id": key, "profile": profile}
        return None

    # -----------------------------------------------------
    # 3) Chiamata al modello con multi-agente
    # -----------------------------------------------------
    def _call_openai(self, task: str, chain: List[Dict[str, Any]]) -> str:
        if not self.client:
            return "⚠️ OpenAI non configurato. Uso solo logica locale."

        messages: List[Dict[str, str]] = []

        # Guardian universale
        messages.append({"role": "system", "content": self.guardian.get_system_prompt()})

        # Layering dei profili agenti
        for agent in chain:
            messages.append({"role": "system", "content": str(agent["profile"])})

        # Memoria locale (compact)
        if self.memory:
            mem_text = "\n".join(
                f"- {m['user']}: {m['model'][:60]}..." for m in self.memory[-5:]
            )
            messages.append({"role": "system", "content": f"CONTEXT MEMORY:\n{mem_text}"})

        messages.append({"role": "user", "content": task})

        try:
            response = self.client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=messages,
            )
        except Exception as exc:  # pragma: no cover - dipendenza esterna
            return f"⚠️ Errore chiamando OpenAI: {exc}"

        choice = response.choices[0].message
        return choice.content or ""

    # -----------------------------------------------------
    # 4) Etica
    # -----------------------------------------------------
    def apply_ethics(self, task: str) -> bool:
        t = task.lower()

        if self.ethics.get("no_aggressive_marketing"):
            if any(k in t for k in ("spam", "domina", "controlla")):
                return False

        if self.ethics.get("anonymous_stories_only"):
            if any(
                k in t for k in ("telemetria", "traccia", "log utente", "profilazione")
            ):
                return False
        return True

    # -----------------------------------------------------
    # Guardian precheck (anti-NSFW / etica)
    # -----------------------------------------------------
    def _run_guardian_scan(self, text: str) -> Dict[str, Any]:
        try:
            return guardian_scan_service(text=text)
        except Exception as exc:
            return {"error": f"guardian_scan failed: {exc}"}

    def _print_guardian_report(self, report: Dict[str, Any]) -> None:
        if not report:
            return
        if "error" in report:
            print(f"⚠️ Guardian errore: {report['error']}")
            return
        scores = report.get("scores") or {}
        if not scores:
            return
        print("\n🛡️ Guardian check:")
        print(f" - Ethical score : {scores.get('ethical_score')}")
        print(f" - Risk level    : {scores.get('risk_level')}")
        dep = scores.get("dependency_hours_risk")
        if dep is not None:
            print(f" - Dependency risk (h): {dep}")

    # -----------------------------------------------------
    # 5) Esecuzione completa (multi-agente + memoria)
    # -----------------------------------------------------
    def run(self, task: str) -> None:
        if not self.apply_ethics(task):
            print("🚫 Task bloccato da policy etiche.")
            return

        if self.guardian_enabled:
            self.guardian_last_report = self._run_guardian_scan(task)
            self._print_guardian_report(self.guardian_last_report)
            scores = (self.guardian_last_report or {}).get("scores") or {}
            if scores.get("risk_level") == "high":
                print("⚠️ Guardian segnala rischio alto; procedo con cautela.")

        chain = self.select_agents_chain(task)
        if not chain:
            print("Nessun agente adatto trovato.")
            return

        print("\n👼 Catena attivata:")
        for agent in chain:
            print(f" - {agent['id']}")

        print("\n🤖 Invocazione modello...\n")

        # 1) Prima prova: normale multi-agente
        output = self._call_openai(task, chain)

        # 2) Se il testo è corto → provala in pipeline
        score = self.self_score(output)
        if score < 0.4:
            print("🔁 Risposta debole → attivo PIPE MODE…")
            output = self.pipe_chain(task, chain)

        print(output)

        # Memoria locale + persistenza (max 50 messaggi)
        self.memory.append({"user": task, "model": output})
        if len(self.memory) > 50:
            self.memory.pop(0)
        self.save_memory()

    # -----------------------------------------------------
    # 6) Self-scoring dell'output
    # -----------------------------------------------------
    def self_score(self, text: str) -> float:
        """
        Valutazione semplicissima: più testo = più forte.
        Può essere raffinata in futuro con metriche migliori.
        """
        if not text:
            return 0.0
        length = len(text.split())
        if length < 20:
            return 0.2
        if length < 60:
            return 0.5
        return 0.9

    # -----------------------------------------------------
    # 7) Modalità interattiva con self-task
    # -----------------------------------------------------
    def interactive_loop(self) -> None:
        print("\n🌱 Nodo33 Orchestrator – Modalità Attiva\n")
        print("Scrivi un task o premi Invio per far parlare lui.\n")

        while True:
            try:
                user_in = input("🪨  Nodo33 >> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Fine sessione.")
                break

            if user_in in ("exit", "quit"):
                print("👋 Fine sessione.")
                break

            # Se lascio vuoto → orchestratore propone un proprio task
            if user_in == "":
                user_in = self.generate_self_task()

            if not user_in:
                continue

            self.run(user_in)

    def generate_self_task(self) -> str:
        """
        L’orchestratore genera un proprio task, basato sulla memoria,
        sui pattern e sulle ultime richieste.
        """
        if not self.memory:
            return "offri un messaggio di pace e custodia per iniziare"

        last_user = self.memory[-1]["user"].lower()

        if "terra" in last_user:
            return "continua il discorso sulla Terra con un atto pratico"
        if "sasso" in last_user:
            return "spiega il prossimo passo dei Sassi Digitali"
        if "adozione" in last_user or "diffusione" in last_user:
            return "proponi un modo etico per diffondere il Codex"
        if "pace" in last_user:
            return "porta un esempio reale di pace applicata"

        # fallback poetico
        return "offri una meditazione semplice sul dono"

    # -----------------------------------------------------
    # 8) Memoria persistente
    # -----------------------------------------------------
    def save_memory(self) -> None:
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory[-50:], f, indent=2, ensure_ascii=False)
        except Exception as e:  # pragma: no cover - I/O best-effort
            print("⚠️ Impossibile salvare memoria:", e)

    # -----------------------------------------------------
    # 9) Pipe tra agenti (output → input)
    # -----------------------------------------------------
    def pipe_chain(self, task: str, chain: List[Dict[str, Any]]) -> str:
        """
        Esegue la catena in pipe:
        - Primo agente → produce testo
        - Secondo agente → prende quel testo come input
        - Terzo agente → elabora il risultato finale
        """
        current_input = task

        for agent in chain:
            output = self._call_openai(f"[PIPE INPUT]\n{current_input}", [agent])
            if not output:
                output = "⚠️ Nessuna risposta da agente in pipe."
            current_input = output

        return current_input


if __name__ == "__main__":
    orch = Nodo33Orchestrator()
    orch.interactive_loop()

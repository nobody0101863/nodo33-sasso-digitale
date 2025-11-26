import importlib
import json
import os
import textwrap
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import requests
except ImportError:
    requests = None  # gestiamo il caso senza requests

from agents.base import AgentResult


@dataclass
class AgentSpec:
    id: str
    name: str
    role: str
    module: str
    tags: List[str]


class GuardianOrchestrator:
    """
    Orchestratore etico di Nodo33.
    Carica gli agenti dal registry e li esegue in pipeline su testo/url.
    """

    def __init__(
        self,
        registry_path: str = "agents/registry.yaml",
        default_pipeline: Optional[List[str]] = None,
        log_dir: str = "logs/guardian",
        enable_logging: bool = True,
    ) -> None:
        self.registry_path = registry_path
        self.agent_specs: Dict[str, AgentSpec] = {}
        self._load_registry()
        # pipeline di default: audit nsfw + malta + trasmutazione
        self.default_pipeline = default_pipeline or [
            "guardian_ethics",
            "malta_scanner",
            "light_agent",
        ]

        self.enable_logging = enable_logging
        self.log_dir = Path(os.path.expanduser(log_dir))
        if self.enable_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ---------- caricamento registry ----------

    def _load_registry(self) -> None:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for a in data.get("agents", []):
            spec = AgentSpec(
                id=a["id"],
                name=a["name"],
                role=a["role"],
                module=a["module"],
                tags=a.get("tags", []),
            )
            self.agent_specs[spec.id] = spec

    def list_agents(self) -> List[AgentSpec]:
        return list(self.agent_specs.values())

    def get_agent_spec(self, agent_id: str) -> AgentSpec:
        if agent_id not in self.agent_specs:
            raise ValueError(f"Agente sconosciuto: {agent_id}")
        return self.agent_specs[agent_id]

    def _instantiate_agent(self, spec: AgentSpec):
        module_path, class_name = spec.module.split(":")
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    # ---------- helpers contenuto ----------

    def _fetch_url(self, url: str) -> str:
        if requests is None:
            raise RuntimeError(
                "Il modulo 'requests' non è installato. "
                "Esegui: pip install requests"
            )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text

    # ---------- esecuzione pipeline ----------

    def run_pipeline(
        self,
        url: Optional[str] = None,
        text: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Esegue una pipeline di agenti su url e/o testo.
        Ritorna un dizionario con i risultati per agente.
        """
        pipeline = agent_ids or self.default_pipeline

        if not url and not text:
            raise ValueError("Serve almeno uno tra url= o text= nella pipeline.")

        payload: Dict[str, Any] = dict(extra_payload or {})
        payload["url"] = url

        # se non è fornito testo, lo scarichiamo dall'url
        if text:
            payload["text"] = text
        elif url:
            payload["text"] = self._fetch_url(url)

        results: Dict[str, Dict[str, Any]] = {}
        for agent_id in pipeline:
            spec = self.get_agent_spec(agent_id)
            agent = self._instantiate_agent(spec)

            res: AgentResult = agent.run(dict(payload))  # copia payload

            results[agent_id] = {
                "agent": asdict(spec),
                "result": {
                    "summary": res.summary,
                    "details": res.details,
                    "level": res.level,
                },
            }

        report = {
            "input": {
                "url": url,
                "text_excerpt": (payload.get("text") or "")[:300],
            },
            "pipeline": pipeline,
            "results": results,
        }

        # Attacca punteggi globali
        report["scores"] = self._compute_scores(report)

        # Logga su file se attivo
        self._log_report(report)

        return report

    # ---------- scoring & logging ----------

    @staticmethod
    def _compute_scores(report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcola un punteggio etico grezzo e un livello di rischio globale.
        Non è scienza esatta, è un indicatore per il Sasso.
        """
        results = report.get("results", {})
        levels = []

        for agent_id, block in results.items():
            level = block.get("result", {}).get("level", "info")
            levels.append(level)

        # Mappa livelli -> peso
        weights = {"info": 0, "warning": 1, "critical": 2}
        raw_score = sum(weights.get(l, 0) for l in levels)

        # Punteggio etico 0-100 (100 = tutto pulito)
        ethical_score = max(0, 100 - raw_score * 10)

        if "critical" in levels:
            risk_level = "high"
        elif "warning" in levels:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Bonus extra: se AntiAddiction è nella pipeline, prova a stimare “ore a rischio”
        dependency_hours_risk = 0.0
        anti_block = results.get("anti_addiction")
        if anti_block:
            details = anti_block.get("result", {}).get("details", {})
            session_minutes = details.get("session_minutes", 0)
            # Sopra i 45 minuti consideriamo “rischio dipendenza”
            if session_minutes > 45:
                dependency_hours_risk = max(0.0, (session_minutes - 45) / 60.0)

        return {
            "ethical_score": ethical_score,
            "risk_level": risk_level,
            "dependency_hours_risk": round(dependency_hours_risk, 2),
        }

    def _log_report(self, report: Dict[str, Any]) -> None:
        if not self.enable_logging:
            return

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base_name = f"guardian_{ts}"

        # JSON completo
        json_path = self.log_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Versione testo “umana”
        txt_path = self.log_dir / f"{base_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.format_report(report))
            f.write("\n")

    # ---------- pretty print per CLI ----------

    @staticmethod
    def format_report(report: Dict[str, Any]) -> str:
        lines: List[str] = []

        inp = report.get("input", {})
        scores = report.get("scores", {})
        lines.append("=== CODex Guardian Report ===")
        if inp.get("url"):
            lines.append(f"URL: {inp['url']}")
        if inp.get("text_excerpt"):
            lines.append("\n[Excerpt testo]:")
            lines.append(textwrap.indent(inp["text_excerpt"].strip(), "  "))

        if scores:
            lines.append("\n=== Punteggi globali ===")
            lines.append(f"  Ethical score        : {scores.get('ethical_score')}")
            lines.append(f"  Risk level          : {scores.get('risk_level')}")
            lines.append(
                f"  Dependency hours risk: {scores.get('dependency_hours_risk')} h (stima)"
            )

        lines.append("\n=== Risultati per agente ===")
        for agent_id in report.get("pipeline", []):
            block = report["results"][agent_id]
            agent = block["agent"]
            res = block["result"]

            lines.append("\n----------------------------------------")
            lines.append(f"[{agent_id}] {agent['name']}")
            lines.append(f"Ruolo: {agent['role']}")
            lines.append(f"Livello: {res['level']}")
            lines.append(f"Summary: {res['summary']}")

            details = res.get("details", {})
            if details:
                lines.append("Dettagli:")
                for k, v in details.items():
                    lines.append(f"  - {k}: {v}")

        return "\n".join(lines)

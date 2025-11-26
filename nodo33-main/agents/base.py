from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentResult:
    summary: str
    details: Dict[str, Any]
    level: str = "info"  # "info" | "warning" | "critical"


class BaseAgent(ABC):
    id: str = "base"
    name: str = "BaseAgent"
    description: str = "Abstract base agent for Nodo33."

    @abstractmethod
    def run(self, payload: Dict[str, Any]) -> AgentResult:
        """
        Esegue la logica dell'agente.
        `payload` può contenere:
          - text: testo da analizzare
          - url: url da valutare
          - meta: info extra
        """
        raise NotImplementedError

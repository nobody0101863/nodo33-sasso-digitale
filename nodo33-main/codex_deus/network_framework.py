"""Interfacce base per nodi, connessioni e orchestrazione etica leggera."""

from typing import Dict, List


class Node:
    """Interfaccia base per nodi del framework."""

    def process(self, payload: Dict) -> Dict:
        raise NotImplementedError


class Edge:
    """Connessione tipizzata con policy."""

    def __init__(self, src: Node, dst: Node, policy: Dict):
        self.src = src
        self.dst = dst
        self.policy = policy


class Orchestrator:
    """Instrada payload rispettando invarianti etici."""

    def __init__(self, invariants: List[str]):
        self.invariants = invariants

    def route(self, node: Node, payload: Dict) -> Dict:
        # Nota: gli invarianti sono dichiarati, non verificati qui.
        return node.process(payload)

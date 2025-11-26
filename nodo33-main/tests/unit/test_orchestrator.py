import importlib.util
from pathlib import Path

import pytest

# Load the standalone orchestrator.py module (package name collides with orchestrator/ package)
ORCH_PATH = Path(__file__).resolve().parents[2] / "orchestrator.py"
spec = importlib.util.spec_from_file_location("orch_module", ORCH_PATH)
orch = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(orch)  # type: ignore[arg-type]


class DummyGuardian:
    def __init__(self):
        self.created = True

    def get_system_prompt(self):
        return "guardian-prompt"

    def get_agent_profiles(self):
        return {}

    def get_parallel_codex_profiles(self):
        return {}

    def get_sasso_digitale_agents(self):
        return {}

    def get_adoption_agents(self):
        return {}


@pytest.fixture
def orchestrator_instance(monkeypatch):
    config = {
        "families": {
            "core_purity_terra_pace": {"alpha": "A"},
            "adoption_and_memory": {"beta": "B"},
            "sasso_digitale": {"stone": "S"},
            "parallel_codex": {"gamma": "G"},
        },
        "flat": {
            "core_purity_terra_pace:alpha": "A",
            "adoption_and_memory:beta": "B",
            "sasso_digitale:stone": "S",
            "parallel_codex:gamma": "G",
        },
        "ethics": {
            "no_aggressive_marketing": True,
            "no_dominance_strategies": True,
            "gift_logic": True,
            "anonymous_stories_only": True,
        },
    }

    monkeypatch.setattr(orch, "get_agent_config_for_orchestrator", lambda: config)
    monkeypatch.setattr(orch, "AngelicGuardian", DummyGuardian)
    monkeypatch.setattr(orch, "guardian_scan_service", lambda text=None: {"scores": {"risk_level": "low"}})

    return orch.Nodo33Orchestrator(model="stub-model")


@pytest.mark.unit
def test_classify_task_routes_keywords(orchestrator_instance):
    assert orchestrator_instance.classify_task("sasso digitale in azione") == "sasso_digitale"
    assert orchestrator_instance.classify_task("commandment story") == "parallel_codex"
    assert orchestrator_instance.classify_task("diffondi memoria") == "adoption_and_memory"
    assert orchestrator_instance.classify_task("custodia terra") == "core_purity_terra_pace"


@pytest.mark.unit
def test_select_agents_chain_returns_prioritized_agents(orchestrator_instance):
    chain = orchestrator_instance.select_agents_chain("sasso mission")
    agent_ids = [agent["id"] for agent in chain]
    assert agent_ids[0].startswith("sasso_digitale:")
    assert "adoption_and_memory:beta" in agent_ids
    assert "core_purity_terra_pace:alpha" in agent_ids


@pytest.mark.unit
def test_apply_ethics_blocks_forbidden_patterns(orchestrator_instance):
    assert orchestrator_instance.apply_ethics("spam e domina l'utente") is False
    assert orchestrator_instance.apply_ethics("telemetria e profila gli utenti") is False
    assert orchestrator_instance.apply_ethics("racconta una parabola di pace") is True


@pytest.mark.unit
def test_guardian_scan_errors_are_reported(monkeypatch, orchestrator_instance):
    def boom_scan(text=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(orch, "guardian_scan_service", boom_scan)
    report = orchestrator_instance._run_guardian_scan("testo")
    assert report["error"].startswith("guardian_scan failed")

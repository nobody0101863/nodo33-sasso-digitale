import pytest
from fastapi.testclient import TestClient

import mcp_server


@pytest.fixture
def mcp_client():
    return TestClient(mcp_server.app)


@pytest.fixture
def auth_header():
    return {"Authorization": "Bearer lux-mcp-token"}


@pytest.mark.unit
def test_authentication_required(mcp_client):
    response = mcp_client.post("/mcp/recommend_privacy_tool", json={})
    # Missing Authorization header is treated as validation error by FastAPI
    assert response.status_code == 422


@pytest.mark.unit
def test_scope_enforced_for_admin_override(mcp_client, auth_header):
    payload = {"action_type": "force_restart", "override_token": "invalid"}
    response = mcp_client.post("/mcp/override_sasso_protocol", json=payload, headers=auth_header)
    assert response.status_code == 403
    assert "Scope mancante" in response.text


@pytest.mark.unit
def test_codex_guidance_maps_sources(monkeypatch, mcp_client, auth_header):
    called = {}

    def fake_get(path: str):
        called["path"] = path
        return {"source": "test", "message": "ciao", "timestamp": "now"}

    monkeypatch.setattr(mcp_server, "_codex_get", fake_get)

    response = mcp_client.post(
        "/mcp/codex_guidance",
        json={"source": "biblical"},
        headers=auth_header,
    )

    assert response.status_code == 200
    assert called["path"] == "/api/guidance/biblical"
    body = response.json()
    assert body["message"] == "ciao"


@pytest.mark.unit
def test_guardian_scan_requires_payload(mcp_client, auth_header):
    response = mcp_client.post("/mcp/guardian_scan", json={"url": None, "text": None}, headers=auth_header)
    assert response.status_code == 400
    assert "Fornire almeno uno" in response.text


@pytest.mark.unit
def test_guardian_scan_uses_service(monkeypatch, mcp_client, auth_header):
    observed = {}

    def fake_scan(url=None, text=None, agent_ids=None):
        observed["args"] = (url, text, tuple(agent_ids or []))
        return {"scores": {"risk_level": "low"}}

    monkeypatch.setattr(mcp_server, "guardian_scan_service", fake_scan)

    response = mcp_client.post(
        "/mcp/guardian_scan",
        json={"url": None, "text": "ciao mondo", "agents": ["guardian-1"]},
        headers=auth_header,
    )

    assert response.status_code == 200
    assert observed["args"] == (None, "ciao mondo", ("guardian-1",))
    assert response.json()["scores"]["risk_level"] == "low"

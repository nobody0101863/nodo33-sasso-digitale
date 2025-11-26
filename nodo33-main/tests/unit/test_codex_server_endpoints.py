import base64
import sqlite3

import pytest
from fastapi.testclient import TestClient

import codex_server


class DummyProtector:
    def get_status(self):
        return {"status": "ok", "sealed": True}

    def protect_data(self, data: bytes):
        return {"status": "protected", "size": len(data)}

    def protect_http_request(self, headers):
        clean_headers = {k.lower(): v for k, v in headers.items() if k.lower() != "server"}
        clean_headers["gabriel"] = "active"
        return {"status": "protected", "headers": clean_headers}

    def protect_tower_node(self, node_id: str, node_data: bytes):
        return {"status": "protected", "node_id": node_id, "hash": len(node_data)}


@pytest.fixture
def codex_client(tmp_path, monkeypatch):
    """FastAPI TestClient with isolated DB and mocked heavy dependencies."""
    db_path = tmp_path / "codex_server.db"
    monkeypatch.setattr(codex_server, "DB_PATH", db_path)
    codex_server.init_db()

    # Avoid P2P side effects during tests
    codex_server._p2p_network = None

    def fake_filter(content: str, is_image: bool = False):
        is_impure = "impure" in content.lower() or "bad" in content.lower()
        return is_impure, "filtered" if is_impure else "clean"

    monkeypatch.setattr(codex_server, "filter_content", fake_filter)
    monkeypatch.setattr(codex_server, "metadata_protector", DummyProtector())

    client = TestClient(codex_server.app)
    yield client, db_path


def _count_rows(db_path, query, param):
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(query, (param,))
        return cur.fetchone()[0]


@pytest.mark.unit
def test_guidance_endpoints_log_requests_and_memories(codex_client):
    client, db_path = codex_client
    endpoints = [
        "/api/guidance",
        "/api/guidance/angel644",
        "/sasso",
    ]

    for url in endpoints:
        resp = client.get(url)
        assert resp.status_code == 200
        assert "timestamp" in resp.json() or "message" in resp.json()

    for url in endpoints:
        assert _count_rows(db_path, "SELECT COUNT(*) FROM request_log WHERE endpoint=?", url) == 1

    # Memories table should have at least the guidance entries
    assert _count_rows(db_path, "SELECT COUNT(*) FROM memories WHERE endpoint LIKE ?", "/api/guidance%") >= 2


@pytest.mark.unit
def test_filter_endpoint_records_impure_content(codex_client):
    client, db_path = codex_client

    resp = client.post("/api/filter", json={"content": "bad content", "is_image": False})
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_impure"] is True
    assert data["guidance"]  # guidance is attached for impure flows

    rows = _count_rows(db_path, "SELECT COUNT(*) FROM memories WHERE endpoint=?", "/api/filter")
    assert rows == 1


@pytest.mark.unit
def test_protection_endpoints_use_protector_and_log(codex_client):
    client, db_path = codex_client

    status_resp = client.get("/api/protection/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "ok"

    headers_resp = client.post(
        "/api/protection/headers",
        json={"headers": {"User-Agent": "pytest", "Server": "leak"}},
    )
    assert headers_resp.status_code == 200
    headers_body = headers_resp.json()
    assert headers_body["headers"]["gabriel"] == "active"
    assert "server" not in headers_body["headers"]

    payload = base64.b64encode(b"secret-bytes").decode()
    data_resp = client.post("/api/protection/data", json={"data": payload})
    assert data_resp.status_code == 200
    assert data_resp.json()["size"] == len(b"secret-bytes")

    assert _count_rows(db_path, "SELECT COUNT(*) FROM request_log WHERE endpoint LIKE ?", "/api/protection/%") == 3

from fastapi.testclient import TestClient

from marinfold_server.app import create_app


def test_healthz(monkeypatch):
    monkeypatch.setenv("MARINFOLD_SERVER_ENVIRONMENT", "test")
    client = TestClient(create_app())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["environment"] == "test"


def test_auth_check_requires_configured_token(monkeypatch):
    monkeypatch.delenv("MARINFOLD_SERVER_TOKEN", raising=False)
    client = TestClient(create_app())

    response = client.get("/v1/auth-check")

    assert response.status_code == 503


def test_auth_check_accepts_bearer_token(monkeypatch):
    monkeypatch.setenv("MARINFOLD_SERVER_TOKEN", "secret")
    client = TestClient(create_app())

    response = client.get("/v1/auth-check", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "authenticated": True}


def test_auth_check_rejects_bad_token(monkeypatch):
    monkeypatch.setenv("MARINFOLD_SERVER_TOKEN", "secret")
    client = TestClient(create_app())

    response = client.get("/v1/auth-check", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401

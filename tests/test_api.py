"""Task 5.1 — API: /health, /chat auth e contrato."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api as api_module
from src.config import get_settings

_API_KEY = "test-brain-key"


@pytest.fixture
def client(test_env: dict[str, str]) -> TestClient:
    get_settings.cache_clear()
    # Recria o grafo/ha_client do lifespan com settings de teste.
    with TestClient(api_module.app) as c:
        yield c


def test_health_no_auth(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_without_api_key_is_422(client: TestClient) -> None:
    # Header X-API-Key é obrigatório (Header(...)) → 422 quando ausente.
    response = client.post("/chat", json={"text": "oi"})
    assert response.status_code == 422


def test_chat_with_wrong_api_key_is_401(client: TestClient) -> None:
    response = client.post("/chat", json={"text": "oi"}, headers={"X-API-Key": "errada"})
    assert response.status_code == 401


def test_chat_valid_key_calls_graph(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Mocka o grafo compilado para evitar chamar o LLM real.
    async def fake_ainvoke(payload: dict) -> dict:  # noqa: ARG001
        return {
            "messages": [
                __import__("langchain_core.messages", fromlist=["AIMessage"]).AIMessage(
                    content="tudo certo"
                )
            ],
            "spoken": True,
            "error": None,
        }

    monkeypatch.setattr(client.app.state.graph, "ainvoke", fake_ainvoke)

    response = client.post("/chat", json={"text": "ligar luz"}, headers={"X-API-Key": _API_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "tudo certo"
    assert body["spoken"] is True


def test_chat_reply_unwraps_gemini_blocks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reply deve ser texto limpo mesmo quando o LLM devolve blocos (Gemini)."""
    from langchain_core.messages import AIMessage

    async def fake_ainvoke(payload: dict) -> dict:  # noqa: ARG001
        return {
            "messages": [
                AIMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Não encontrei a população atual.",
                            "extras": {"signature": "CqECA...sig..."},
                        }
                    ]
                )
            ],
            "spoken": False,
            "error": None,
        }

    monkeypatch.setattr(client.app.state.graph, "ainvoke", fake_ainvoke)

    response = client.post("/chat", json={"text": "população"}, headers={"X-API-Key": _API_KEY})
    assert response.status_code == 200
    body = response.json()
    # Sem serialização de lista/dict/assinatura — texto puro.
    assert body["reply"] == "Não encontrei a população atual."
    assert "{" not in body["reply"]
    assert "signature" not in body["reply"]

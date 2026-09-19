"""Task 5.1 — API: /health, /chat auth e contrato."""

from __future__ import annotations

import logging

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import api as api_module
from src.config import get_settings

_API_KEY = "test-brain-key"
_HA_URL = "http://homeassistant.local:8123"


@pytest.fixture
def client(test_env: dict[str, str]) -> TestClient:
    get_settings.cache_clear()
    # Mocka /api/states para o warm-up do catálogo no lifespan não depender
    # de rede real (lista vazia ⇒ catálogo vazio ⇒ degrada graciosa).
    with respx.mock(base_url=_HA_URL, assert_all_called=False) as router:
        router.get("/api/states").mock(return_value=httpx.Response(200, json=[]))
        with TestClient(api_module.app) as c:
            yield c


def test_health_no_auth(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_logs_active_llm(
    test_env: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    with respx.mock(base_url=_HA_URL, assert_all_called=False) as router:
        router.get("/api/states").mock(return_value=httpx.Response(200, json=[]))
        with TestClient(api_module.app):
            pass

    assert "llm_provider=gemini" in caplog.text
    assert "llm_model=gemini-3.5-flash" in caplog.text


def test_chat_without_api_key_is_422(client: TestClient) -> None:
    # Header X-API-Key é obrigatório (Header(...)) → 422 quando ausente.
    response = client.post("/chat", json={"text": "oi"})
    assert response.status_code == 422


def test_chat_with_wrong_api_key_is_401(client: TestClient) -> None:
    response = client.post("/chat", json={"text": "oi"}, headers={"X-API-Key": "errada"})
    assert response.status_code == 401


def test_chat_valid_key_calls_graph(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Mocka o grafo compilado para evitar chamar o LLM real.
    async def fake_astream(payload: dict, **kwargs: object):  # noqa: ARG001
        message = __import__("langchain_core.messages", fromlist=["AIMessage"]).AIMessage(
            content="tudo certo"
        )
        yield {"chatbot": {"messages": [message]}}
        yield {"speak": {"spoken": True, "error": None}}

    monkeypatch.setattr(client.app.state.graph, "astream", fake_astream)

    response = client.post("/chat", json={"text": "ligar luz"}, headers={"X-API-Key": _API_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "tudo certo"
    assert body["spoken"] is True


def test_chat_without_metadata_defaults_to_satellite(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regressão: sem metadata ⇒ source="satellite" ⇒ speak roda ⇒ spoken=True."""
    from langchain_core.messages import AIMessage

    async def fake_astream(payload: dict, **kwargs: object):  # noqa: ARG001
        yield {"chatbot": {"messages": [AIMessage(content="ok")]}}
        yield {"speak": {"spoken": True, "error": None}}

    monkeypatch.setattr(client.app.state.graph, "astream", fake_astream)

    response = client.post("/chat", json={"text": "oi"}, headers={"X-API-Key": _API_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "satellite"
    assert body["spoken"] is True
    assert body["metadata"]["tools_used"] == []
    # Campo aditivo: sem tool_diagnostics no update ⇒ lista vazia na resposta.
    assert body["metadata"]["tool_diagnostics"] == []


def test_chat_telegram_source_skips_speak_and_collects_tools(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """source=telegram: grafo termina sem speak (spoken=False) e tools_used populado."""
    from langchain_core.messages import AIMessage, ToolMessage

    async def fake_astream(payload: dict, **kwargs: object):  # noqa: ARG001
        ai = AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {}, "id": "1"}])
        yield {"chatbot": {"messages": [ai]}}
        yield {
            "tools": {
                "messages": [ToolMessage(content="22C", name="get_weather", tool_call_id="1")]
            }
        }
        yield {"chatbot": {"messages": [AIMessage(content="Máxima de 22.")]}}
        # Sem nó speak — telegram_end.

    monkeypatch.setattr(client.app.state.graph, "astream", fake_astream)

    response = client.post(
        "/chat",
        json={"text": "clima em cascavel", "metadata": {"source": "telegram"}},
        headers={"X-API-Key": _API_KEY},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "telegram"
    assert body["spoken"] is False
    assert body["metadata"]["tools_used"] == ["get_weather"]
    assert body["reply"] == "Máxima de 22."


def test_chat_source_is_case_insensitive(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Telegram" (maiúsculo) normaliza para telegram e pula a Alexa."""
    from langchain_core.messages import AIMessage

    async def fake_astream(payload: dict, **kwargs: object):  # noqa: ARG001
        yield {"chatbot": {"messages": [AIMessage(content="oi")]}}

    monkeypatch.setattr(client.app.state.graph, "astream", fake_astream)

    response = client.post(
        "/chat",
        json={"text": "oi", "metadata": {"source": "Telegram"}},
        headers={"X-API-Key": _API_KEY},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "telegram"
    assert body["spoken"] is False


def test_chat_empty_source_defaults_to_satellite(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """source vazio/string em branco ⇒ satélite (voz)."""
    from langchain_core.messages import AIMessage

    async def fake_astream(payload: dict, **kwargs: object):  # noqa: ARG001
        yield {"chatbot": {"messages": [AIMessage(content="ok")]}}
        yield {"speak": {"spoken": True, "error": None}}

    monkeypatch.setattr(client.app.state.graph, "astream", fake_astream)

    response = client.post(
        "/chat",
        json={"text": "oi", "metadata": {"source": "   "}},
        headers={"X-API-Key": _API_KEY},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "satellite"


def test_chat_reply_unwraps_gemini_blocks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """reply deve ser texto limpo mesmo quando o LLM devolve blocos (Gemini)."""
    from langchain_core.messages import AIMessage

    caplog.set_level(logging.INFO, logger="uvicorn.error")

    async def fake_astream(payload: dict, **kwargs: object):  # noqa: ARG001
        message = AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "Não encontrei a população atual.",
                    "extras": {"signature": "CqECA...sig..."},
                }
            ]
        )
        yield {"chatbot": {"messages": [message]}}
        yield {"tools": {"messages": []}}
        yield {"chatbot": {"messages": [message]}}
        yield {"speak": {"spoken": False, "error": None}}

    monkeypatch.setattr(client.app.state.graph, "astream", fake_astream)

    response = client.post("/chat", json={"text": "população"}, headers={"X-API-Key": _API_KEY})
    assert response.status_code == 200
    body = response.json()
    # Sem serialização de lista/dict/assinatura — texto puro.
    assert body["reply"] == "Não encontrei a população atual."
    assert "{" not in body["reply"]
    assert "signature" not in body["reply"]
    # A timeline registra a sequência de nós percorrida.
    assert "Chatbot Agent" in caplog.text
    assert "Tool Agent" in caplog.text
    assert "Speak Agent" in caplog.text


def test_chat_exposes_tool_diagnostics_on_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Falha de tool: diagnóstico exposto em metadata.tool_diagnostics e timeline.

    O diagnóstico NÃO pode vazar para ``reply`` (texto amigável p/ Alexa).
    """
    from langchain_core.messages import AIMessage, ToolMessage

    caplog.set_level(logging.INFO, logger="uvicorn.error")

    async def fake_astream(payload: dict, **kwargs: object):  # noqa: ARG001
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "control_device",
                    "args": {"action": "on", "entity_id": "light.cozinha"},
                    "id": "1",
                }
            ],
        )
        yield {"chatbot": {"messages": [ai]}}
        yield {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="Não consegui acionar o dispositivo.",
                        name="control_device",
                        tool_call_id="1",
                    )
                ],
                "tool_diagnostics": [
                    {
                        "tool": "control_device",
                        "entity_id": "light.cozinha",
                        "action": "on",
                        "status_code": 404,
                        "body": "entity not found",
                        "error": "HTTPStatusError",
                    }
                ],
            }
        }
        friendly_reply = "Não consegui acionar a luz da cozinha."
        yield {"chatbot": {"messages": [AIMessage(content=friendly_reply)]}}
        yield {"speak": {"spoken": True, "error": None}}

    monkeypatch.setattr(client.app.state.graph, "astream", fake_astream)

    response = client.post(
        "/chat", json={"text": "ligar a luz da cozinha"}, headers={"X-API-Key": _API_KEY}
    )
    assert response.status_code == 200
    body = response.json()

    diagnostics = body["metadata"]["tool_diagnostics"]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["tool"] == "control_device"
    assert diagnostic["entity_id"] == "light.cozinha"
    assert diagnostic["status_code"] == 404
    assert "entity not found" in diagnostic["body"]
    assert diagnostic["error"] == "HTTPStatusError"

    # Texto amigável p/ a Alexa — diagnóstico não vaza para a fala.
    assert body["reply"] == "Não consegui acionar a luz da cozinha."
    assert "entity not found" not in body["reply"]

    # Timeline registra o erro da tool com o diagnóstico.
    assert "Erro na tool" in caplog.text
    assert "light.cozinha" in caplog.text
    assert "404" in caplog.text

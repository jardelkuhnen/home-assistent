"""Dashboard: GET /dashboard, overview, detalhe do turno e integração com o /chat."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

import api as api_module
from src.config import get_settings

_API_KEY = "test-brain-key"
_HA_URL = "http://homeassistant.local:8123"


@pytest.fixture
def client(test_env: dict[str, str]) -> Iterator[TestClient]:
    get_settings.cache_clear()
    with respx.mock(base_url=_HA_URL, assert_all_called=False) as router:
        router.get("/api/states").mock(return_value=httpx.Response(200, json=[]))
        with TestClient(api_module.app) as c:
            yield c


def _chat(client: TestClient, source: str | None = None) -> httpx.Response:
    body: dict[str, Any] = {"text": "ligar a luz da cozinha"}
    if source:
        body["metadata"] = {"source": source, "session_id": "s-1"}
    return client.post("/chat", json=body, headers={"X-API-Key": _API_KEY})


def test_dashboard_page_is_public_html(client: TestClient) -> None:
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_overview_without_api_key_has_expected_shape(client: TestClient) -> None:
    response = client.get("/api/dashboard/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["cards"] == {"active_runs": 0, "tools_total": 3, "runs_24h": 0, "errors_24h": 0}
    assert body["brain"]["llm_provider"] == "gemini"
    assert body["brain"]["llm_model"] == "gemini-3.5-flash"
    assert body["brain"]["uptime_s"] >= 0
    assert isinstance(body["brain"]["catalog_loaded"], bool)
    assert body["active"] == []
    assert body["recent"] == []
    assert sorted(t["name"] for t in body["tools"]) == [
        "control_device",
        "get_weather",
        "web_search",
    ]
    assert {"description", "calls", "errors", "last_used"} <= body["tools"][0].keys()
    assert {"chatbot", "tools", "speak", "__start__", "__end__"} <= set(body["topology"]["nodes"])
    assert {"source": "tools", "target": "chatbot", "conditional": False} in body["topology"][
        "edges"
    ]
    # Canais conhecidos aparecem mesmo sem atividade.
    assert body["channels"]["satellite"] == {"last_activity": None, "runs_24h": 0}
    assert body["channels"]["telegram"] == {"last_activity": None, "runs_24h": 0}


def test_chat_turn_shows_up_in_overview_and_run_detail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_astream(payload: dict, **kwargs: object):  # noqa: ARG001
        ai = AIMessage(content="", tool_calls=[{"name": "control_device", "args": {}, "id": "1"}])
        yield {"chatbot": {"messages": [ai]}}
        yield {
            "tools": {
                "messages": [
                    ToolMessage(content="falhou", name="control_device", tool_call_id="1")
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
        yield {"chatbot": {"messages": [AIMessage(content="Não consegui.")]}}
        yield {"speak": {"spoken": True, "error": None}}

    monkeypatch.setattr(client.app.state.graph, "astream", fake_astream)

    assert _chat(client, "telegram").status_code == 200

    overview = client.get("/api/dashboard/overview").json()
    assert overview["cards"]["runs_24h"] == 1
    assert overview["cards"]["errors_24h"] == 0  # falha de tool não torna o turno "error"
    [recent] = overview["recent"]
    assert recent["source"] == "telegram"
    assert recent["tools"] == ["control_device"]
    assert recent["status"] == "ok"
    assert recent["spoken"] is True
    assert overview["channels"]["telegram"]["runs_24h"] == 1
    tool = next(t for t in overview["tools"] if t["name"] == "control_device")
    assert (tool["calls"], tool["errors"]) == (1, 1)

    detail = client.get(f"/api/dashboard/runs/{recent['id']}")
    assert detail.status_code == 200
    run = detail.json()
    assert [(s["node"], s["tool"], s["status"]) for s in run["steps"]] == [
        ("chatbot", None, "ok"),
        ("tools", "control_device", "error"),
        ("chatbot", None, "ok"),
        ("speak", None, "ok"),
    ]
    tool_step = run["steps"][1]
    assert tool_step["error"] == "404 HTTPStatusError"
    # Só metadados: nem o body da falha nem o texto do pedido/resposta são guardados.
    raw = detail.text
    assert "entity not found" not in raw
    assert "ligar a luz" not in raw
    assert "Não consegui" not in raw


def test_run_detail_unknown_is_404(client: TestClient) -> None:
    assert client.get("/api/dashboard/runs/nao-existe").status_code == 404


def test_active_run_is_listed_and_finishes(client: TestClient) -> None:
    runs = client.app.state.runs
    run_id = runs.start("satellite", None)

    overview = client.get("/api/dashboard/overview").json()
    assert overview["cards"]["active_runs"] == 1
    assert overview["active"][0]["id"] == run_id
    assert overview["active"][0]["elapsed_ms"] >= 0
    assert client.get(f"/api/dashboard/runs/{run_id}").json()["status"] == "running"


def test_graph_exception_records_error_run_and_propagates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_astream(payload: dict, **kwargs: object):  # noqa: ARG001
        yield {"chatbot": {"messages": [AIMessage(content="ok")]}}
        raise RuntimeError("boom com conteúdo sensível")

    monkeypatch.setattr(client.app.state.graph, "astream", failing_astream)

    with pytest.raises(RuntimeError, match="boom"):
        _chat(client)

    overview = client.get("/api/dashboard/overview").json()
    assert overview["active"] == []
    assert overview["cards"]["errors_24h"] == 1
    [recent] = overview["recent"]
    assert recent["status"] == "error"
    run = client.get(f"/api/dashboard/runs/{recent['id']}").json()
    assert run["error"] == "RuntimeError"
    assert "sensível" not in str(run)


def test_state_error_makes_run_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_astream(payload: dict, **kwargs: object):  # noqa: ARG001
        yield {"chatbot": {"messages": [AIMessage(content="Desculpe")], "error": "LLM caiu"}}

    monkeypatch.setattr(client.app.state.graph, "astream", fake_astream)

    assert _chat(client).status_code == 200

    [recent] = client.get("/api/dashboard/overview").json()["recent"]
    assert recent["status"] == "error"

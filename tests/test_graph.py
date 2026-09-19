"""Task 4 — speak_node (mock ok/falha), prompt sem Markdown, workflow compila."""

from __future__ import annotations

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.config import Settings
from src.graph.nodes import (
    build_speak_node,
    build_tool_node,
    chatbot_node,
    content_to_text,
    route_tools,
)
from src.graph.prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_TELEGRAM
from src.graph.state import AgentState
from src.services.ha_client import HomeAssistantClient
from src.tools.home import _FALLBACK

_HA_URL = "http://homeassistant.local:8123"


class _FakeHA(HomeAssistantClient):
    """HA client de teste sem abrir conexão real."""

    def __init__(self, speak_result: dict[str, object]) -> None:  # noqa: D107
        self._speak_result = speak_result

    async def speak(self, text: str) -> dict[str, object]:  # type: ignore[override]
        return self._speak_result


def _state_with_reply(reply: str) -> AgentState:
    return {"messages": [AIMessage(content=reply)], "spoken": False, "error": None}


async def test_speak_node_ok(test_settings: Settings) -> None:  # type: ignore[no-untyped-def]
    ha = _FakeHA({"ok": True})
    node = build_speak_node(ha)  # type: ignore[arg-type]
    result = await node(_state_with_reply("olá"))
    assert result["spoken"] is True
    assert result["error"] is None


async def test_speak_node_failure_sets_error(test_settings: Settings) -> None:  # type: ignore[no-untyped-def]
    ha = _FakeHA({"ok": False, "error": "timeout"})
    node = build_speak_node(ha)  # type: ignore[arg-type]
    result = await node(_state_with_reply("olá"))
    assert result["spoken"] is False
    assert "timeout" in result["error"]


async def test_speak_node_no_content(test_settings: Settings) -> None:  # type: ignore[no-untyped-def]
    ha = _FakeHA({"ok": True})
    node = build_speak_node(ha)  # type: ignore[arg-type]
    result = await node({"messages": [], "spoken": False, "error": None})
    assert result["spoken"] is False


def test_route_tools_with_tool_calls() -> None:
    ai = AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {}, "id": "1"}])
    assert route_tools({"messages": [ai], "spoken": False, "error": None}) == "tools"


def test_route_tools_without_tool_calls_routes_to_speak() -> None:
    """Canal de voz (source ausente) sem tool calls → speak (Alexa)."""
    ai = AIMessage(content="pronto")
    assert route_tools({"messages": [ai], "spoken": False, "error": None}) == "speak"


def test_route_tools_satellite_routes_to_speak() -> None:
    ai = AIMessage(content="pronto")
    state = {"messages": [ai], "spoken": False, "error": None, "source": "satellite"}
    assert route_tools(state) == "speak"


def test_route_tools_telegram_routes_to_end() -> None:
    """Telegram sem tool calls → telegram_end (pula a Alexa)."""
    ai = AIMessage(content="pronto")
    state = {"messages": [ai], "spoken": False, "error": None, "source": "telegram"}
    assert route_tools(state) == "telegram_end"


def test_route_tools_telegram_case_insensitive() -> None:
    ai = AIMessage(content="pronto")
    state = {"messages": [ai], "spoken": False, "error": None, "source": "Telegram"}
    assert route_tools(state) == "telegram_end"


def test_route_tools_telegram_with_tool_calls_still_tools() -> None:
    """Telegram com tool calls → tools (o loop de tools roda em qualquer canal)."""
    ai = AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {}, "id": "1"}])
    state = {"messages": [ai], "spoken": False, "error": None, "source": "telegram"}
    assert route_tools(state) == "tools"


def _state_with_control_device_call(action: str, entity_id: str) -> AgentState:
    """Estado com uma AIMessage pendendo ``control_device`` (última mensagem)."""
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "control_device",
                "args": {"action": action, "entity_id": entity_id},
                "id": "call_1",
            }
        ],
    )
    return {"messages": [ai], "spoken": False, "error": None}


async def test_tool_node_success_returns_tool_message(
    test_settings: Settings,
) -> None:  # type: ignore[no-untyped-def]
    """Tool call bem-sucedida → ToolMessage com o retorno amigável, sem diagnóstico."""
    node = build_tool_node()
    with respx.mock(base_url=_HA_URL) as router:
        router.post("/api/services/light/turn_on").mock(return_value=httpx.Response(200, json=[]))
        result = await node(_state_with_control_device_call("on", "light.luz_sala"))
    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].name == "control_device"
    assert messages[0].tool_call_id == "call_1"
    assert messages[0].content == "Liguei o luz da sala."
    assert result["tool_diagnostics"] == []


async def test_tool_node_failure_returns_fallback_and_diagnostic(
    test_settings: Settings,
) -> None:  # type: ignore[no-untyped-def]
    """Erro HTTP do HA → ToolMessage _FALLBACK + diagnóstico no estado."""
    node = build_tool_node()
    with respx.mock(base_url=_HA_URL) as router:
        router.post("/api/services/light/turn_on").mock(
            return_value=httpx.Response(404, text="entity not found")
        )
        result = await node(_state_with_control_device_call("on", "light.luz_sala"))
    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].content == _FALLBACK
    diagnostics = result["tool_diagnostics"]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["tool"] == "control_device"
    assert diagnostic["entity_id"] == "light.luz_sala"
    assert diagnostic["action"] == "on"
    assert diagnostic["status_code"] == 404
    assert "entity not found" in diagnostic["body"]
    assert diagnostic["error"] == "HTTPStatusError"


async def test_tool_node_without_pending_tool_calls_returns_empty(
    test_settings: Settings,
) -> None:  # type: ignore[no-untyped-def]
    """Última mensagem sem tool calls → nó não faz nada ({"messages": []})."""
    node = build_tool_node()
    state: AgentState = {
        "messages": [AIMessage(content="pronto")],
        "spoken": False,
        "error": None,
    }
    result = await node(state)
    assert result == {"messages": []}


async def test_tool_node_unknown_tool_returns_toolnotfound_diagnostic(
    test_settings: Settings,
) -> None:  # type: ignore[no-untyped-def]
    """Tool inexistente em ALL_TOOLS → ToolMessage amigável + diagnostic ToolNotFound."""
    node = build_tool_node()
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "nope_device",
                "args": {"action": "on", "entity_id": "light.luz_sala"},
                "id": "call_9",
            }
        ],
    )
    state: AgentState = {"messages": [ai], "spoken": False, "error": None}
    result = await node(state)
    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].content == "Tool não encontrada"
    diagnostics = result["tool_diagnostics"]
    assert len(diagnostics) == 1
    assert diagnostics[0]["tool"] == "nope_device"
    assert diagnostics[0]["error"] == "ToolNotFound"


def test_system_prompt_has_no_markdown() -> None:
    assert "#" not in SYSTEM_PROMPT
    assert "**" not in SYSTEM_PROMPT
    assert "```" not in SYSTEM_PROMPT


def test_telegram_prompt_allows_markdown() -> None:
    """O prompt do Telegram permite Markdown/listas (diferente do de voz)."""
    assert "Markdown" in SYSTEM_PROMPT_TELEGRAM
    assert "Telegram" in SYSTEM_PROMPT_TELEGRAM


async def test_chatbot_node_selects_telegram_prompt(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """chatbot_node injeta SYSTEM_PROMPT_TELEGRAM quando source=="telegram"."""
    captured: dict[str, object] = {}

    class _Bound:
        async def ainvoke(self, messages, config=None, **kwargs):  # noqa: ANN001, ARG002
            captured["messages"] = messages
            return AIMessage(content="ok")

    class _LLM:
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ARG002
            return _Bound()

    monkeypatch.setattr("src.graph.nodes.get_llm", lambda: _LLM())
    state: AgentState = {
        "messages": [HumanMessage(content="oi")],
        "spoken": False,
        "error": None,
        "source": "telegram",
        "session_id": "telegram_42",
    }
    await chatbot_node(state)
    messages = list(captured["messages"])  # type: ignore[arg-type]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == SYSTEM_PROMPT_TELEGRAM


async def test_chatbot_node_selects_voice_prompt_by_default(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """Sem source (satélite), chatbot_node injeta SYSTEM_PROMPT (voz)."""
    captured: dict[str, object] = {}

    class _Bound:
        async def ainvoke(self, messages, config=None, **kwargs):  # noqa: ANN001, ARG002
            captured["messages"] = messages
            return AIMessage(content="ok")

    class _LLM:
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ARG002
            return _Bound()

    monkeypatch.setattr("src.graph.nodes.get_llm", lambda: _LLM())
    state: AgentState = {
        "messages": [HumanMessage(content="oi")],
        "spoken": False,
        "error": None,
        "source": None,
        "session_id": None,
    }
    await chatbot_node(state)
    messages = list(captured["messages"])  # type: ignore[arg-type]
    assert messages[0].content == SYSTEM_PROMPT


async def test_chatbot_node_injects_catalog_context(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """chatbot_node injeta o catálogo de dispositivos como SystemMessage após o prompt."""

    class _StubCatalog:
        def as_context(self) -> str:
            return "Dispositivos disponíveis:\n- Principal Sala (switch.principal_sala)"

    captured: dict[str, object] = {}

    class _Bound:
        async def ainvoke(self, messages, config=None, **kwargs):  # noqa: ANN001, ARG002
            captured["messages"] = messages
            return AIMessage(content="ok")

    class _LLM:
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ARG002
            return _Bound()

    monkeypatch.setattr("src.graph.nodes.get_llm", lambda: _LLM())
    monkeypatch.setattr("src.graph.nodes.get_catalog", lambda: _StubCatalog())
    state: AgentState = {
        "messages": [HumanMessage(content="ligar principal sala")],
        "spoken": False,
        "error": None,
        "source": None,
        "session_id": None,
    }
    await chatbot_node(state)
    messages = list(captured["messages"])  # type: ignore[arg-type]
    # [0]=prompt de voz, [1]=catálogo, [2]=HumanMessage.
    assert isinstance(messages[1], SystemMessage)
    assert "switch.principal_sala" in messages[1].content
    assert "Principal Sala" in messages[1].content


async def test_chatbot_node_omits_catalog_when_empty(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """Catálogo vazio (HA fora no boot) → não injeta SystemMessage de catálogo."""

    class _StubCatalog:
        def as_context(self) -> str:
            return ""

    captured: dict[str, object] = {}

    class _Bound:
        async def ainvoke(self, messages, config=None, **kwargs):  # noqa: ANN001, ARG002
            captured["messages"] = messages
            return AIMessage(content="ok")

    class _LLM:
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ARG002
            return _Bound()

    monkeypatch.setattr("src.graph.nodes.get_llm", lambda: _LLM())
    monkeypatch.setattr("src.graph.nodes.get_catalog", lambda: _StubCatalog())
    state: AgentState = {
        "messages": [HumanMessage(content="oi")],
        "spoken": False,
        "error": None,
        "source": None,
        "session_id": None,
    }
    await chatbot_node(state)
    messages = list(captured["messages"])  # type: ignore[arg-type]
    # Sem catálogo: [0]=prompt, [1]=HumanMessage (sem SystemMessage extra).
    assert isinstance(messages[0], SystemMessage)
    assert not isinstance(messages[1], SystemMessage)


async def test_chatbot_node_handles_llm_timeout_gracefully(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """LLM timeout/erro → chatbot_node devolve AIMessage amigável, não propaga.

    Regressão de robusteza: antes, qualquer exceção do motor cognitivo
    (timeout do Ollama, erro de rede) propagava sem tratamento e virava 500
    no endpoint /chat. Agora o nó captura, loga e devolve uma AIMessage de
    fallback — o grafo segue para speak/telegram_end e o usuário ouve/leve
    um erro natural em vez de 500.
    """

    class _FailingBound:
        async def ainvoke(self, messages, config=None, **kwargs):  # noqa: ANN001, ARG002
            raise httpx.ReadTimeout("ollama demorou")

    class _LLM:
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ARG002
            return _FailingBound()

    monkeypatch.setattr("src.graph.nodes.get_llm", lambda: _LLM())
    state: AgentState = {
        "messages": [HumanMessage(content="ligar luz")],
        "spoken": False,
        "error": None,
        "source": None,
        "session_id": None,
    }
    result = await chatbot_node(state)
    # Não propagou: devolveu estado com uma AIMessage amigável.
    assert "messages" in result
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content  # não vazio
    # Sinaliza erro no estado p/ o endpoint registrar (sem virar 500).
    assert result.get("error") is not None


async def test_chatbot_node_skips_catalog_after_tool_call(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """Pós-tool call (há ToolMessage) → catálogo NÃO é reenviado ao LLM.

    O catálogo só é necessário para a decisão de qual entity_id acionar (1ª
    chamada do turno). Após o tool_node resolver o dispositivo, a 2ª chamada
    só formata a resposta — reenviar o catálogo dobra o custo do prompt no
    caminho crítico (controle de dispositivo) e empurra o TTFB do Ollama
    local além do timeout. Pular aqui halve o custo sem perder cobertura.
    """

    class _StubCatalog:
        def as_context(self) -> str:
            return "Dispositivos:\n- Principal Sala (switch.principal_sala)"

    captured: dict[str, object] = {}

    class _Bound:
        async def ainvoke(self, messages, config=None, **kwargs):  # noqa: ANN001, ARG002
            captured["messages"] = messages
            return AIMessage(content="Liguei.")

    class _LLM:
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ARG002
            return _Bound()

    monkeypatch.setattr("src.graph.nodes.get_llm", lambda: _LLM())
    monkeypatch.setattr("src.graph.nodes.get_catalog", lambda: _StubCatalog())
    # Estado pós-tool: HumanMessage -> AIMessage(tool_call) -> ToolMessage.
    state: AgentState = {
        "messages": [
            HumanMessage(content="ligar principal sala"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "control_device",
                        "args": {"action": "on", "entity_id": "switch.principal_sala"},
                        "id": "c1",
                    }
                ],
            ),
            ToolMessage(
                content="Liguei o Principal Sala.", tool_call_id="c1", name="control_device"
            ),
        ],
        "spoken": False,
        "error": None,
        "source": None,
        "session_id": None,
    }
    await chatbot_node(state)
    messages = list(captured["messages"])  # type: ignore[arg-type]
    # [0]=prompt de voz, [1]=HumanMessage (catálogo OMITIDO pós-tool).
    assert isinstance(messages[0], SystemMessage)
    assert not isinstance(messages[1], SystemMessage)
    assert "switch.principal_sala" not in "".join(
        str(getattr(m, "content", "")) for m in messages if isinstance(m, SystemMessage)
    )


def test_build_graph_compiles(test_settings: Settings) -> None:  # type: ignore[no-untyped-def]
    from src.graph import build_graph

    ha = _FakeHA({"ok": True})
    graph = build_graph(ha)  # type: ignore[arg-type]
    assert graph is not None


def test_content_to_text_plain_string() -> None:
    assert content_to_text("olá") == "olá"


def test_content_to_text_unwraps_gemini_blocks() -> None:
    # Payload real retornado pelo Gemini (blocos com signature/extras).
    content = [
        {
            "type": "text",
            "text": "Não encontrei a população atual de Cascavel.",
            "extras": {"signature": "CqECARFNMg/6TdUxbCW+KFxkJGWKghn5ojhOjKRiU+AFtil95+YV"},
        }
    ]
    assert content_to_text(content) == "Não encontrei a população atual de Cascavel."


def test_content_to_text_joins_multiple_blocks() -> None:
    content = [{"type": "text", "text": "Máxima de 22."}, {"type": "text", "text": "Mínima de 14."}]
    assert content_to_text(content) == "Máxima de 22. Mínima de 14."


def test_content_to_text_empty_blocks_returns_empty() -> None:
    assert content_to_text([]) == ""
    assert content_to_text([{"type": "text", "text": ""}]) == ""


async def test_speak_node_unwraps_blocks_before_speaking(test_settings: Settings) -> None:  # type: ignore[no-untyped-def]
    """speak_node extrai texto limpo de blocos do Gemini antes de falar."""
    spoken_text: list[str] = []

    class _CapturingHA(_FakeHA):
        async def speak(self, text: str) -> dict[str, object]:  # type: ignore[override]
            spoken_text.append(text)
            return {"ok": True}

    ha = _CapturingHA({"ok": True})
    node = build_speak_node(ha)  # type: ignore[arg-type]
    state: AgentState = {
        "messages": [
            AIMessage(
                content=[
                    {"type": "text", "text": "Não sei.", "extras": {"signature": "abc"}},
                ]
            )
        ],
        "spoken": False,
        "error": None,
    }
    result = await node(state)
    assert result["spoken"] is True
    # Falou texto limpo, sem a assinatura/extras.
    assert spoken_text == ["Não sei."]

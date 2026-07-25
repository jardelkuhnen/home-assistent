"""Task 4 — speak_node (mock ok/falha), prompt sem Markdown, workflow compila."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.config import Settings
from src.graph.nodes import build_speak_node, content_to_text, route_tools
from src.graph.prompt import SYSTEM_PROMPT
from src.graph.state import AgentState
from src.services.ha_client import HomeAssistantClient


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


def test_route_tools_without_tool_calls() -> None:
    ai = AIMessage(content="pronto")
    assert route_tools({"messages": [ai], "spoken": False, "error": None}) == "end"


def test_system_prompt_has_no_markdown() -> None:
    assert "#" not in SYSTEM_PROMPT
    assert "**" not in SYSTEM_PROMPT
    assert "```" not in SYSTEM_PROMPT


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

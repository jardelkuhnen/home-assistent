"""Task 4 — speak_node (mock ok/falha), prompt sem Markdown, workflow compila."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.graph.nodes import build_speak_node, route_tools
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


async def test_speak_node_ok(test_settings) -> None:  # type: ignore[no-untyped-def]
    ha = _FakeHA({"ok": True})
    node = build_speak_node(ha)  # type: ignore[arg-type]
    result = await node(_state_with_reply("olá"))
    assert result["spoken"] is True
    assert result["error"] is None


async def test_speak_node_failure_sets_error(test_settings) -> None:  # type: ignore[no-untyped-def]
    ha = _FakeHA({"ok": False, "error": "timeout"})
    node = build_speak_node(ha)  # type: ignore[arg-type]
    result = await node(_state_with_reply("olá"))
    assert result["spoken"] is False
    assert "timeout" in result["error"]


async def test_speak_node_no_content(test_settings) -> None:  # type: ignore[no-untyped-def]
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


def test_build_graph_compiles(test_settings) -> None:  # type: ignore[no-untyped-def]
    from src.graph import build_graph

    ha = _FakeHA({"ok": True})
    graph = build_graph(ha)  # type: ignore[arg-type]
    assert graph is not None

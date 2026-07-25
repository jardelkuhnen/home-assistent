"""Nós do grafo: chatbot, tool_node, speak_node (terminal — ADR-0002)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from src.config import get_llm
from src.graph.prompt import SYSTEM_PROMPT
from src.graph.state import AgentState
from src.services.ha_client import HomeAssistantClient
from src.tools import ALL_TOOLS

# Alias de tipo para os nós do grafo. Nós devolvem estado parcial (reducer
# ``add_messages``/sobrescrita mesclam no estado completo).
Node = Callable[[AgentState], Awaitable[dict[str, Any]]]


async def chatbot_node(state: AgentState) -> dict[str, Any]:
    """Invoca o motor cognitivo com tools e system prompt injetados."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = await llm_with_tools.ainvoke(messages)

    ai_message = response if isinstance(response, BaseMessage) else AIMessage(content=str(response))

    return {"messages": [ai_message]}


def build_tool_node() -> ToolNode:
    """Nó que executa as tool calls pendentes."""
    return ToolNode(ALL_TOOLS)


def route_tools(state: AgentState) -> str:
    """Roteamento condicional: ``tools`` se há tool calls, senão ``end``."""
    messages = state.get("messages", [])
    if not messages:
        return "end"
    last = messages[-1]
    if isinstance(last, AIMessage) and bool(last.tool_calls):
        return "tools"
    return "end"


def build_speak_node(
    ha_client: HomeAssistantClient,
) -> Node:
    """Factory do nó terminal de TTS (ADR-0002).

    Recebe ``ha_client`` por injeção para testabilidade. Lê a última
    ``AIMessage``, chama ``ha_client.speak`` e seta ``spoken``/``error``.
    Sempre devolve a resposta textual para a API.
    """

    async def speak_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        text = ""
        if messages:
            last = messages[-1]
            text = last.content if isinstance(last.content, str) else str(last.content)

        if not text:
            return {"spoken": False, "error": "sem conteúdo para falar"}

        result = await ha_client.speak(text)
        if isinstance(result, dict) and result.get("ok") is True:
            return {"spoken": True, "error": None}
        error = str(result.get("error", "falha desconhecida")) if isinstance(result, dict) else ""
        return {"spoken": False, "error": error}

    return speak_node


__all__ = [
    "Node",
    "chatbot_node",
    "build_tool_node",
    "build_speak_node",
    "route_tools",
]

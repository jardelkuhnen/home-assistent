"""Montagem e compilação do StateGraph (ADR-0002: speak é nó terminal)."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.graph.nodes import build_speak_node, build_tool_node, chatbot_node, route_tools
from src.graph.state import AgentState
from src.services.ha_client import HomeAssistantClient


def build_graph(ha_client: HomeAssistantClient) -> Any:
    """Compila o grafo: chatbot → (tools → chatbot)* → speak | END.

    O ``add_conditional_edges`` pós-``chatbot`` tem 3 destinos:
    * ``tools`` — há tool calls (qualquer canal), executa e volta ao chatbot.
    * ``speak`` — canal de voz (satélite/Alexa): fala e termina em ``END``.
    * ``telegram_end`` — canal Telegram: termina direto em ``END``, sem Alexa.
    """
    graph = StateGraph(AgentState)

    # langgraph 1.x tipa add_node com TypeVars de método que não inferem do
    # callable em mypy strict; a closure do speak_node dispara um falso-positivo.
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tools", build_tool_node())
    graph.add_node("speak", build_speak_node(ha_client))  # type: ignore[arg-type]

    graph.set_entry_point("chatbot")

    graph.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", "speak": "speak", "telegram_end": END},
    )
    graph.add_edge("tools", "chatbot")
    graph.add_edge("speak", END)

    return graph.compile()

"""Estado do grafo (decisão Q6)."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Estado que flui entre os nós do grafo.

    * ``messages`` — histórico acumulado (reducer ``add_messages``).
    * ``spoken`` — indicador de que o TTS foi acionado com sucesso.
    * ``error`` — mensagem de erro do nó terminal, se houver.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    spoken: bool
    error: str | None

"""Estado do grafo (decisão Q6)."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Estado que flui entre os nós do grafo.

    * ``messages`` — histórico acumulado (reducer ``add_messages``).
    * ``spoken`` — indicador de que o TTS foi acionado com sucesso.
    * ``error`` — mensagem de erro do nó terminal, se houver.
    * ``source`` — canal de origem da solicitação (``"telegram"`` para o
      bot de texto; ``None``/``"satellite"`` ⇒ comportamento de voz). O
      roteamento usa ``source`` para pular o nó ``speak`` (Alexa) no caminho
      Telegram, mantendo os canais isolados.
    * ``session_id`` — identificador de sessão do canal (ex. ``telegram_<chat>``).
      Aceito e levado ao estado, **não consumido** — reservado para um
      checkpointer de memória futuro.
    * ``tool_diagnostics`` — diagnósticos de falha das tool calls, um dict por
      erro (contrato: ``{tool, entity_id, action, status_code, body, error}``),
      preenchido pelo nó ``tools`` (sobrescrita mesclada, sem reducer — cada
      execução do nó substitui a lista). Default ``[]`` (campo aditivo).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    spoken: bool
    error: str | None
    source: str | None
    session_id: str | None
    tool_diagnostics: list[dict[str, Any]]

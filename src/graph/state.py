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
    * ``source`` — canal de origem da solicitação (``"telegram"`` para o
      bot de texto; ``None``/``"satellite"`` ⇒ comportamento de voz). O
      roteamento usa ``source`` para pular o nó ``speak`` (Alexa) no caminho
      Telegram, mantendo os canais isolados.
    * ``session_id`` — identificador de sessão do canal (ex. ``telegram_<chat>``).
      Aceito e levado ao estado, **não consumido** — reservado para um
      checkpointer de memória futuro.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    spoken: bool
    error: str | None
    source: str | None
    session_id: str | None

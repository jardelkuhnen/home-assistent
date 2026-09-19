"""Nós do grafo: chatbot, tool_node, speak_node (terminal — ADR-0002)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from src.config import get_llm
from src.graph.prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_TELEGRAM
from src.graph.state import AgentState
from src.services.ha_client import HomeAssistantClient
from src.tools import ALL_TOOLS
from src.tools.home import _FALLBACK

# Mesmo logger do ``api.py``: os logs aparecem no console do brain (uvicorn).
logger = logging.getLogger("uvicorn.error")

# Alias de tipo para os nós do grafo. Nós devolvem estado parcial (reducer
# ``add_messages``/sobrescrita mesclam no estado completo).
Node = Callable[[AgentState], Awaitable[dict[str, Any]]]

# Catálogo de dispositivos injetado pelo lifespan no boot. ``None`` ⇒ ausente
# (testes sem warm-up, ou boot sem catálogo): o chatbot_node simplesmente não
# injeta a SystemMessage de catálogo — comportamento pré-catálogo.
_CATALOG: DeviceCatalogLike | None = None


class DeviceCatalogLike(Protocol):
    """Contrato mínimo do catálogo que o grafo consome (contexto p/ o LLM)."""

    def as_context(self) -> str: ...


def set_catalog(catalog: DeviceCatalogLike | None) -> None:
    """Injeta o catálogo (lifespan). ``None`` reseta (testes)."""
    global _CATALOG
    _CATALOG = catalog


def get_catalog() -> DeviceCatalogLike | None:
    """Acesso ao catálogo injetado (factory-style p/ monkeypatch em testes)."""
    return _CATALOG


def content_to_text(content: Any) -> str:
    """Extrai texto limpo do ``content`` de uma mensagem do LLM.

    O Gemini (via langchain-google-genai) devolve ``content`` como **lista de
    blocos** ``[{"type": "text", "text": "...", "extras": {...}}]`` em vez de
    string pura. Fazer ``str(content)`` serializa a lista inteira — incluindo
    assinaturas criptográficas do Google — e esse lixo chega à Alexa.

    Esta função desempacota só o campo ``text`` de cada bloco e junta com
    espaço, devolvendo sempre uma string limpa e falável.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return " ".join(parts).strip()
    return str(content)


async def chatbot_node(state: AgentState) -> dict[str, Any]:
    """Invoca o motor cognitivo com tools e system prompt injetados.

    O system prompt varia por canal: ``SYSTEM_PROMPT_TELEGRAM`` quando
    ``source=="telegram"`` (permite Markdown/respostas mais longas),
    ``SYSTEM_PROMPT`` caso contrário (voz, texto plano falável).
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    source = state.get("source")
    is_telegram = isinstance(source, str) and source.lower() == "telegram"
    prompt = SYSTEM_PROMPT_TELEGRAM if is_telegram else SYSTEM_PROMPT

    messages: list[BaseMessage] = [SystemMessage(content=prompt)]
    # Catálogo de dispositivos: injetado como SystemMessage extra quando
    # populado, para o LLM escolher o entity_id correto ao chamar
    # control_device. Vazio/ausente ⇒ omitido (sem regressão).
    catalog = get_catalog()
    if catalog is not None:
        context = catalog.as_context()
        if context:
            messages.append(SystemMessage(content=context))
    messages.extend(state["messages"])
    response = await llm_with_tools.ainvoke(messages)

    ai_message = response if isinstance(response, BaseMessage) else AIMessage(content=str(response))

    return {"messages": [ai_message]}


def build_tool_node() -> Node:
    """Factory do nó ``tools``: executa cada tool call pendente com diagnóstico.

    Substitui o ``ToolNode`` pré-construído. Em sucesso devolve a ``ToolMessage``
    normal (retorno da tool) para o LLM. Em falha (exceção da tool — ex. erro
    HTTP do Home Assistant), loga o diagnóstico, devolve uma ``ToolMessage``
    amigável (``_FALLBACK``) para o LLM e acumula o diagnóstico no estado
    (``tool_diagnostics``), sem interromper o grafo.
    """

    async def tool_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}
        last = messages[-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return {"messages": []}

        tool_messages: list[ToolMessage] = []
        diagnostics: list[dict[str, Any]] = []
        for tool_call in last.tool_calls:
            tool = next((t for t in ALL_TOOLS if t.name == tool_call["name"]), None)
            if tool is None:
                diagnostic: dict[str, Any] = {
                    "tool": tool_call["name"],
                    "entity_id": tool_call["args"].get("entity_id"),
                    "action": tool_call["args"].get("action"),
                    "status_code": None,
                    "body": "",
                    "error": "ToolNotFound",
                }
                logger.error("tool %s falhou: %r", tool_call["name"], diagnostic)
                tool_messages.append(
                    ToolMessage(
                        content="Tool não encontrada",
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )
                diagnostics.append(diagnostic)
                continue
            try:
                content = await tool.ainvoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=str(content),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )
            except Exception as exc:  # noqa: BLE001 — diagnóstico, não crash do grafo
                diagnostic = {
                    "tool": tool_call["name"],
                    "entity_id": tool_call["args"].get("entity_id"),
                    "action": tool_call["args"].get("action"),
                    "status_code": getattr(getattr(exc, "response", None), "status_code", None),
                    "body": (getattr(getattr(exc, "response", None), "text", None) or "")[:500],
                    "error": type(exc).__name__,
                }
                logger.error("tool %s falhou: %r", tool_call["name"], diagnostic)
                tool_messages.append(
                    ToolMessage(
                        content=_FALLBACK,
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )
                diagnostics.append(diagnostic)

        return {"messages": tool_messages, "tool_diagnostics": diagnostics}

    return tool_node


def route_tools(state: AgentState) -> str:
    """Roteamento condicional pós-``chatbot``.

    * ``"tools"`` — há tool calls pendentes (qualquer canal).
    * ``"telegram_end"`` — sem tool calls e ``source=="telegram"``: termina
      direto em ``END``, **sem** acionar a Alexa (canal de texto isolado).
    * ``"speak"`` — sem tool calls e canal de voz (``source`` ausente/vazio/
      ``"satellite"``): segue para o nó ``speak`` (TTS via Home Assistant).
    """
    messages = state.get("messages", [])
    if not messages:
        last_is_tools = False
    else:
        last = messages[-1]
        last_is_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
    if last_is_tools:
        return "tools"

    source = state.get("source")
    if isinstance(source, str) and source.lower() == "telegram":
        return "telegram_end"
    return "speak"


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
            text = content_to_text(last.content)

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
    "content_to_text",
    "chatbot_node",
    "build_tool_node",
    "build_speak_node",
    "route_tools",
]

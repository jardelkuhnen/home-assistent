"""Cérebro — servidor FastAPI expondo POST /chat (contrato Q5).

* Auth via header ``X-API-Key`` (baseline item 4).
* Constrói ``ha_client`` + grafo no lifespan e processa um turno por chamada.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, TypedDict

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.config import Settings, configured_llm_model, get_settings
from src.dashboard.routes import router as dashboard_router
from src.graph import build_graph
from src.graph.nodes import content_to_text
from src.graph.nodes import set_catalog as set_graph_catalog
from src.services.catalog import DeviceCatalog
from src.services.ha_client import HomeAssistantClient
from src.services.runs import RunStore
from src.tools.home import set_catalog as set_tool_catalog

# Uvicorn configura este logger para o console; usar o logger do módulo faria
# o registro depender da configuração do root logger da aplicação chamadora.
logger = logging.getLogger("uvicorn.error")


class TimelineEvent(TypedDict):
    """Evento de execução do grafo, seguro para ser exibido no console."""

    timestamp: str
    agent: str
    action: str
    details: str | None


def timeline_event(agent: str, action: str, details: str | None = None) -> TimelineEvent:
    """Cria um evento de timeline sem incluir conteúdo ou argumentos do usuário."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "agent": agent,
        "action": action,
        "details": details,
    }


def _tool_names_from_update(update: dict[str, Any]) -> list[str]:
    """Extrai os nomes das tools executadas a partir do update do nó ``tools``.

    Cada mensagem do ``ToolNode`` carrega o atributo ``name`` com o nome da
    tool. Reaproveitado tanto para a timeline de auditoria quanto para o
    ``metadata.tools_used`` da resposta.
    """
    return [
        name
        for message in update.get("messages", [])
        if isinstance((name := getattr(message, "name", None)), str) and name
    ]


def _tool_errors_from_update(update: dict[str, Any]) -> dict[str, str]:
    """Erro por tool (casado pelo nome) a partir de ``tool_diagnostics``.

    Guarda só ``status_code`` e o tipo do erro — nunca o ``body`` do HA.
    """
    return {
        diagnostic["tool"]: " ".join(
            str(part) for part in (diagnostic.get("status_code"), diagnostic.get("error")) if part
        )
        or "erro"
        for diagnostic in update.get("tool_diagnostics", [])
    }


def events_from_node_update(
    node_name: str, update: dict[str, Any], settings: Settings
) -> list[TimelineEvent]:
    """Converte a atualização de um nó LangGraph em eventos de auditoria."""
    if node_name == "chatbot":
        details = f"provider: {settings.llm_provider} | model: {configured_llm_model(settings)}"
        return [timeline_event("Chatbot Agent", "Resposta gerada pelo motor cognitivo", details)]

    if node_name == "tools":
        tool_names = _tool_names_from_update(update)
        events = [
            timeline_event("Tool Agent", "Tool executada", f"tool: {tool_name}")
            for tool_name in tool_names
        ]
        # Diagnóstico do HA (nó de tool custom): um evento por falha, além da
        # timeline de "Tool executada" mantida para sucesso.
        for diagnostic in update.get("tool_diagnostics", []):
            events.append(
                timeline_event(
                    "Tool Agent",
                    "Erro na tool",
                    (
                        f"tool: {diagnostic['tool']} | {diagnostic['tool']}"
                        f" {diagnostic['action']} {diagnostic['entity_id']}"
                        f" -> {diagnostic['status_code']} {diagnostic['body'][:120]}"
                    ),
                )
            )
        return events or [timeline_event("Tool Agent", "Nó de tools executado")]

    if node_name == "speak":
        if update.get("spoken") is True:
            return [timeline_event("Speak Agent", "Resposta enviada para a Alexa")]
        return [
            timeline_event(
                "Speak Agent",
                "Resposta não enviada para a Alexa",
                update.get("error"),
            )
        ]

    return [timeline_event(node_name, "Nó executado")]


def format_timeline_for_terminal(timeline: list[TimelineEvent]) -> str:
    """Formata a timeline para uma única entrada INFO legível no terminal."""
    lines = ["Timeline da solicitação:"]
    for event in timeline:
        timestamp = datetime.fromisoformat(event["timestamp"]).astimezone().strftime("%H:%M:%S")
        details = f" | {event['details']}" if event["details"] else ""
        lines.append(f"  {timestamp} | {event['agent']:<14} | {event['action']}{details}")
    return "\n".join(lines)


class Metadata(BaseModel):
    """Metadados opcionais da solicitação, carregando o canal de origem.

    * ``source`` — canal: ``"telegram"`` (texto, pula Alexa) ou ausente/vazio
      (⇒ satélite/voz). Case-insensitive.
    * ``session_id`` — identificador de sessão do canal (ex. ``telegram_<chat>``).
      Aceito e levado ao estado, **não consumido** (reservado p/ checkpointer).
    """

    source: str | None = None
    session_id: str | None = None


class ToolDiagnostic(BaseModel):
    """Diagnóstico do retorno do Home Assistant em uma execução de tool.

    Preenchido pelo nó de tool custom quando a tool falha (ex. erro HTTP do
    HA): status HTTP + body + entity/action. É observabilidade de máquina —
    não vaza para a fala (``reply``) nem para a Alexa.
    """

    tool: str
    entity_id: str | None = None
    action: str | None = None
    status_code: int | None = None
    body: str = ""
    error: str = ""


class ResponseMetadata(BaseModel):
    """Metadados da resposta — ferramentas acionadas no turno.

    * ``tools_used`` — nomes das tools executadas no turno.
    * ``tool_diagnostics`` — diagnóstico do HA por tool que falhou (default
      ``[]``; campo aditivo, retrocompatível).
    """

    tools_used: list[str] = []
    tool_diagnostics: list[ToolDiagnostic] = []


class ChatRequest(BaseModel):
    text: str
    metadata: Metadata | None = None


class ChatResponse(BaseModel):
    reply: str
    spoken: bool
    source: str
    metadata: ResponseMetadata
    error: str | None = None


async def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> Settings:
    """Valida o header ``X-API-Key`` contra ``Settings.brain_api_key``."""
    if x_api_key != settings.brain_api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )
    return settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Ciclo de vida: cria HA client, aquece o catálogo e compila o grafo."""
    settings = get_settings()
    logger.info(
        "Brain iniciado | llm_provider=%s | llm_model=%s",
        settings.llm_provider,
        configured_llm_model(settings),
    )
    ha_client = HomeAssistantClient(settings)
    app.state.ha_client = ha_client
    # Catálogo de dispositivos: warm-up no boot (best-effort — em falha do HA
    # cai no fallback estático). Compartilha o ha_client (fechado no finally).
    catalog = DeviceCatalog(
        ha_client=ha_client,
        ttl_s=settings.catalog_ttl_s,
    )
    await catalog.refresh()
    app.state.catalog = catalog
    set_graph_catalog(catalog)
    set_tool_catalog(catalog)
    app.state.graph = build_graph(ha_client)
    runs = RunStore(settings.dashboard_db_path)
    await runs.init()
    app.state.runs = runs
    app.state.started_at = time.monotonic()
    try:
        yield
    finally:
        await ha_client.close()


app = FastAPI(title="home-assistent-brain", lifespan=lifespan)
app.include_router(dashboard_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(verify_api_key),
) -> ChatResponse:
    """Processa um turno: roda o grafo e devolve reply + spoken + source.

    ``source`` derivado de ``request.metadata.source`` (default ``"satellite"``)
    decide o roteamento: ``"telegram"`` pula a Alexa (texto isolado); demais
    valores seguem o caminho de voz (satélite/Alexa), preservando a regressão.
    """
    graph = app.state.graph
    raw_source = request.metadata.source if request.metadata else None
    source = (
        raw_source.strip().lower()
        if isinstance(raw_source, str) and raw_source.strip()
        else "satellite"
    )
    session_id = request.metadata.session_id if request.metadata else None

    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=request.text)],
        "spoken": False,
        "error": None,
        "source": source,
        "session_id": session_id,
    }
    result = initial_state.copy()
    result["messages"] = list(initial_state["messages"])
    timeline: list[TimelineEvent] = []
    tools_used: set[str] = set()
    # Diagnóstico bruto chega como list[dict] no update do nó ``tools``
    # (sempre presente, ``[]`` em sucesso/no-op); converte ao montar a resposta.
    tool_diagnostics_accum: list[dict[str, Any]] = []

    runs: RunStore = app.state.runs
    run_id = runs.start(source, session_id)
    last_update_at = time.monotonic()
    try:
        async for updates in graph.astream(initial_state, stream_mode="updates"):
            # O astream só emite após o nó terminar: a duração é o intervalo
            # entre updates consecutivos (o primeiro conta desde o start).
            now = time.monotonic()
            duration_ms = int((now - last_update_at) * 1000)
            last_update_at = now
            for node_name, update in updates.items():
                timeline.extend(events_from_node_update(node_name, update, settings))
                tool_names = _tool_names_from_update(update) if node_name == "tools" else []
                runs.record_node(
                    run_id, node_name, duration_ms, tool_names, _tool_errors_from_update(update)
                )
                if node_name == "tools":
                    tools_used.update(tool_names)
                    tool_diagnostics_accum.extend(update.get("tool_diagnostics", []))
                if "messages" in update:
                    result["messages"].extend(update["messages"])
                for field in ("spoken", "error", "source"):
                    if field in update:
                        result[field] = update[field]
    except BaseException as exc:
        # BaseException: um cliente que desconecta cancela a task, e o turno
        # não pode ficar "rodando" para sempre no dashboard.
        await runs.finish(run_id, "error", False, type(exc).__name__)
        raise

    logger.info("%s", format_timeline_for_terminal(timeline))

    messages = result.get("messages", [])
    reply = ""
    if messages:
        last = messages[-1]
        reply = content_to_text(last.content)

    spoken = bool(result.get("spoken", False))
    error = result.get("error")
    await runs.finish(run_id, "error" if error else "ok", spoken, error)
    final_source = str(result.get("source") or source)
    return ChatResponse(
        reply=reply,
        spoken=spoken,
        source=final_source,
        metadata=ResponseMetadata(
            tools_used=sorted(tools_used),
            tool_diagnostics=[
                ToolDiagnostic(**diagnostic) for diagnostic in tool_diagnostics_accum
            ],
        ),
        error=error,
    )


# Cliente HTTP do Cérebro — exportado p/ o Satélite reaproveitar com timeout.
def brain_http_client(settings: Settings | None = None) -> httpx.AsyncClient:
    """Cria um ``httpx.AsyncClient`` apontando para ``BRAIN_URL`` com timeout."""
    s = settings or get_settings()
    return httpx.AsyncClient(
        base_url=str(s.brain_url),
        headers={"X-API-Key": s.brain_api_key.get_secret_value()},
        timeout=s.brain_timeout_s,
    )

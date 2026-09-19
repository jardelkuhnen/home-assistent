"""Rotas do dashboard: ``GET /dashboard`` e ``GET /api/dashboard/*``.

Públicas e somente leitura (decisão do usuário): só metadados dos turnos,
nunca texto de conversa. Erros de leitura do banco propagam como 500.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import Settings, configured_llm_model, get_settings
from src.services.runs import RunStore, utc_iso
from src.tools import ALL_TOOLS

router = APIRouter()

_INDEX = Path(__file__).with_name("index.html")
_RECENT_LIMIT = 20
# Canais conhecidos aparecem no painel mesmo sem atividade registrada.
_KNOWN_CHANNELS = ("satellite", "telegram")


class Step(BaseModel):
    seq: int
    node: str
    tool: str | None
    duration_ms: int | None
    status: str
    error: str | None


class RunDetail(BaseModel):
    id: str
    started_at: str
    finished_at: str | None
    duration_ms: int | None
    source: str
    session_id: str | None
    status: str
    spoken: bool | None
    error: str | None
    steps: list[Step]


class ActiveRun(BaseModel):
    id: str
    source: str
    started_at: str
    elapsed_ms: int
    steps: list[Step]


class RunSummary(BaseModel):
    id: str
    started_at: str
    source: str
    duration_ms: int | None
    tools: list[str]
    status: str
    spoken: bool


class Cards(BaseModel):
    active_runs: int
    tools_total: int
    runs_24h: int
    errors_24h: int


class Brain(BaseModel):
    llm_provider: str
    llm_model: str
    uptime_s: int
    catalog_loaded: bool


class Channel(BaseModel):
    last_activity: str | None
    runs_24h: int


class ToolInfo(BaseModel):
    name: str
    description: str
    calls: int
    errors: int
    last_used: str | None


class Edge(BaseModel):
    source: str
    target: str
    conditional: bool


class Topology(BaseModel):
    nodes: list[str]
    edges: list[Edge]


class Overview(BaseModel):
    cards: Cards
    brain: Brain
    channels: dict[str, Channel]
    active: list[ActiveRun]
    recent: list[RunSummary]
    tools: list[ToolInfo]
    topology: Topology


@router.get("/dashboard", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    return FileResponse(_INDEX, media_type="text/html")


@router.get("/api/dashboard/overview", response_model=Overview)
async def overview(request: Request, settings: Settings = Depends(get_settings)) -> Overview:
    state = request.app.state
    runs: RunStore = state.runs
    since = utc_iso(datetime.now(UTC) - timedelta(hours=24))

    active = runs.active()
    stats = await runs.stats(since)
    activity = await runs.channel_activity(since)
    tool_stats = await runs.tool_stats()
    recent = await runs.recent(_RECENT_LIMIT)

    channels = {
        source: Channel(**activity.get(source, {"last_activity": None, "runs_24h": 0}))
        for source in (*_KNOWN_CHANNELS, *(s for s in activity if s not in _KNOWN_CHANNELS))
    }
    empty: dict[str, Any] = {"calls": 0, "errors": 0, "last_used": None}
    graph = state.graph.get_graph()

    return Overview(
        cards=Cards(
            active_runs=len(active),
            tools_total=len(ALL_TOOLS),
            runs_24h=stats["runs"],
            errors_24h=stats["errors"],
        ),
        brain=Brain(
            llm_provider=settings.llm_provider,
            llm_model=configured_llm_model(settings),
            uptime_s=int(time.monotonic() - state.started_at),
            catalog_loaded=state.catalog.has_devices(),
        ),
        channels=channels,
        active=[ActiveRun(**run) for run in active],
        recent=[RunSummary(**run) for run in recent],
        tools=[
            ToolInfo(
                name=tool.name, description=tool.description, **tool_stats.get(tool.name, empty)
            )
            for tool in ALL_TOOLS
        ],
        topology=Topology(
            nodes=list(graph.nodes),
            edges=[
                Edge(source=e.source, target=e.target, conditional=e.conditional)
                for e in graph.edges
            ],
        ),
    )


@router.get("/api/dashboard/runs/{run_id}", response_model=RunDetail)
async def run_detail(run_id: str, request: Request) -> RunDetail:
    run = await request.app.state.runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="turno não encontrado")
    return RunDetail(**run)

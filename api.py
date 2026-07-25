"""Cérebro — servidor FastAPI expondo POST /chat (contrato Q5).

* Auth via header ``X-API-Key`` (baseline item 4).
* Constrói ``ha_client`` + grafo no lifespan e processa um turno por chamada.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.config import Settings, get_settings
from src.graph import build_graph
from src.services.ha_client import HomeAssistantClient


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    reply: str
    spoken: bool


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
    """Ciclo de vida: cria HA client e grafo compilado na inicialização."""
    settings = get_settings()
    ha_client = HomeAssistantClient(settings)
    app.state.ha_client = ha_client
    app.state.graph = build_graph(ha_client)
    try:
        yield
    finally:
        await ha_client.close()


app = FastAPI(title="home-assistent-brain", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(verify_api_key),
) -> ChatResponse:
    """Processa um turno: roda o grafo e devolve reply + spoken."""
    graph = app.state.graph
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=request.text)],
            "spoken": False,
            "error": None,
        }
    )

    messages = result.get("messages", [])
    reply = ""
    if messages:
        last = messages[-1]
        reply = last.content if isinstance(last.content, str) else str(last.content)

    spoken = bool(result.get("spoken", False))
    return ChatResponse(reply=reply, spoken=spoken)


# Cliente HTTP do Cérebro — exportado p/ o Satélite reaproveitar com timeout.
def brain_http_client(settings: Settings | None = None) -> httpx.AsyncClient:
    """Cria um ``httpx.AsyncClient`` apontando para ``BRAIN_URL`` com timeout."""
    s = settings or get_settings()
    return httpx.AsyncClient(
        base_url=str(s.brain_url),
        headers={"X-API-Key": s.brain_api_key.get_secret_value()},
        timeout=s.ha_timeout_s,
    )

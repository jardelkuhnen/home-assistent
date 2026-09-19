"""Tool de controle de dispositivos via Home Assistant.

Validação estrita de ``entity_id`` (baseline de segurança item 3): apenas os
domínios permitidos chegam ao HA. TTS não vive aqui — é nó do grafo (ADR-0002).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from src.services.ha_client import HomeAssistantClient

# Apelidos amigáveis para confirmação falável. v1: dict em memória.
# Hoje é o **fallback estático** do catálogo: usado quando o catálogo dinâmico
# (HA) não está injetado (ex. testes legados) ou não tem o friendly_name.
_ENTITY_ALIASES: dict[str, str] = {
    "switch.tomada_sala": "tomada da sala",
    "switch.tomada_quarto": "tomada do quarto",
    "light.luz_sala": "luz da sala",
    "light.luz_quarto": "luz do quarto",
    "media_player.alexa_sala": "Alexa da sala",
}

_FALLBACK = "Não consegui acionar o dispositivo."

# Catálogo injetado pelo lifespan no boot (DeviceCatalog). Quando ausente
# (testes legados, ou boot sem warm-up), a tool degrada graciosa: pula o
# known-check e usa o fallback estático para o apelido — comportamento
# pré-catálogo, sem regressão.
_catalog: DeviceCatalogLike | None = None


class DeviceCatalogLike(Protocol):
    """Contrato mínimo do catálogo que a tool consome (sync, do cache)."""

    def is_known(self, entity_id: str) -> bool: ...
    def friendly_name(self, entity_id: str) -> str | None: ...
    def has_devices(self) -> bool: ...


def set_catalog(catalog: DeviceCatalogLike | None) -> None:
    """Injeta o catálogo (lifespan). ``None`` reseta (testes)."""
    global _catalog
    _catalog = catalog


def _alias(entity_id: str) -> str:
    """Apelido falável: friendly_name do catálogo > fallback estático > entity_id."""
    if _catalog is not None:
        friendly = _catalog.friendly_name(entity_id)
        if friendly:
            return friendly
    return _ENTITY_ALIASES.get(entity_id, entity_id)


def _build_client() -> HomeAssistantClient:
    """Factory do HA client — isolada para testabilidade."""
    from src.services.ha_client import HomeAssistantClient

    return HomeAssistantClient()


class HomeInput(BaseModel):
    action: Literal["on", "off", "toggle"]
    entity_id: str = Field(..., description="Entity ID, ex: switch.tomada_sala")

    @field_validator("entity_id")
    @classmethod
    def _validate_entity_id(cls, value: str) -> str:
        if not value.startswith(("switch.", "light.", "media_player.")):
            raise ValueError("entity_id deve seguir o padrão ^(switch|light|media_player)\\..+")
        return value


@tool("control_device", args_schema=HomeInput)
async def control_device(action: Literal["on", "off", "toggle"], entity_id: str) -> str:
    """Liga, desliga ou alterna um dispositivo do Home Assistant."""
    # Known-check: se o catálogo está populado (HA vivo) e o entity_id não está
    # nele, bloqueia antes de chamar o HA — evita acionar entity alucinada pelo
    # LLM. Catálogo vazio/ausente ⇒ degrada graciosa (segue para o HA), já que o
    # HA rejeitará com 404 e o nó de tool devolverá _FALLBACK.
    if _catalog is not None and _catalog.has_devices() and not _catalog.is_known(entity_id):
        return f"Não encontrei o dispositivo {entity_id} no catálogo."

    # Sem catch genérico: erros do HA (httpx.HTTPStatusError, httpx.HTTPError)
    # propagam para o nó de tool do grafo, que loga o diagnóstico e devolve o
    # fallback amigável (_FALLBACK) ao LLM. O ``finally`` garante fechar o client.
    client = _build_client()
    try:
        if action == "toggle":
            await client.toggle(entity_id)
        elif action == "on":
            await client.turn_on(entity_id)
        else:
            await client.turn_off(entity_id)
    finally:
        await client.close()

    apelido = _alias(entity_id)
    if action == "on":
        return f"Liguei o {apelido}."
    if action == "off":
        return f"Desliguei o {apelido}."
    return f"Alternei o {apelido}."

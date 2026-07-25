"""Tool de controle de dispositivos via Home Assistant.

Validação estrita de ``entity_id`` (baseline de segurança item 3): apenas os
domínios permitidos chegam ao HA. TTS não vive aqui — é nó do grafo (ADR-0002).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from src.services.ha_client import HomeAssistantClient

# Apelidos amigáveis para confirmação falável. v1: dict em memória.
_ENTITY_ALIASES: dict[str, str] = {
    "switch.tomada_sala": "tomada da sala",
    "switch.tomada_quarto": "tomada do quarto",
    "light.luz_sala": "luz da sala",
    "light.luz_quarto": "luz do quarto",
    "media_player.alexa_sala": "Alexa da sala",
}

_FALLBACK = "Não consegui acionar o dispositivo."


class HomeInput(BaseModel):
    action: Literal["on", "off", "toggle"]
    entity_id: str = Field(..., description="Entity ID, ex: switch.tomada_sala")

    @field_validator("entity_id")
    @classmethod
    def _validate_entity_id(cls, value: str) -> str:
        if not value.startswith(("switch.", "light.", "media_player.")):
            raise ValueError("entity_id deve seguir o padrão ^(switch|light|media_player)\\..+")
        return value


def _alias(entity_id: str) -> str:
    return _ENTITY_ALIASES.get(entity_id, entity_id)


def _build_client() -> HomeAssistantClient:
    """Factory do HA client — isolada para testabilidade."""
    from src.services.ha_client import HomeAssistantClient

    return HomeAssistantClient()


@tool("control_device", args_schema=HomeInput)
async def control_device(action: Literal["on", "off", "toggle"], entity_id: str) -> str:
    """Liga, desliga ou alterna um dispositivo do Home Assistant."""
    try:
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
    except Exception:  # noqa: BLE001 — ferramenta defensiva para voz
        return _FALLBACK

    apelido = _alias(entity_id)
    if action == "on":
        return f"Liguei o {apelido}."
    if action == "off":
        return f"Desliguei o {apelido}."
    return f"Alternei o {apelido}."

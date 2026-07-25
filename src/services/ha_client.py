"""Cliente HTTP do Home Assistant + síntese de voz via Alexa Media Player.

O método ``speak`` é o nó terminal do grafo (ADR-0002): ele **não levanta**
em falha de TTS — captura ``httpx.HTTPError`` e retorna ``{"ok": False, ...}``
para que o nó do grafo decida o que fazer.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.config import Settings, get_settings


class HomeAssistantClient:
    """Cliente async do Home Assistant."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            base_url=str(self._settings.ha_url),
            headers={
                "Authorization": f"Bearer {self._settings.ha_token.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=self._settings.ha_timeout_s,
        )

    async def call_service(
        self, domain: str, service: str, service_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Wrapper do POST ``/api/services/{domain}/{service}``."""
        response = await self._client.post(
            f"/api/services/{domain}/{service}",
            json=service_data or {},
        )
        response.raise_for_status()
        result: Any = response.json()
        if isinstance(result, dict):
            return result
        return {"result": result}

    async def toggle(self, entity_id: str) -> dict[str, Any]:
        """Aciona ``homeassistant.toggle`` para o ``entity_id``."""
        return await self.call_service("homeassistant", "toggle", {"entity_id": entity_id})

    async def turn_on(self, entity_id: str) -> dict[str, Any]:
        domain = entity_id.split(".", 1)[0]
        return await self.call_service(domain, "turn_on", {"entity_id": entity_id})

    async def turn_off(self, entity_id: str) -> dict[str, Any]:
        domain = entity_id.split(".", 1)[0]
        return await self.call_service(domain, "turn_off", {"entity_id": entity_id})

    async def get_state(self, entity_id: str) -> dict[str, Any]:
        """GET ``/api/states/{entity_id}``."""
        response = await self._client.get(f"/api/states/{entity_id}")
        response.raise_for_status()
        result: Any = response.json()
        if isinstance(result, dict):
            return result
        return {"state": result}

    async def speak(self, text: str) -> dict[str, Any]:
        """Síntese de voz via ``notify.alexa_media``.

        Usa ``target`` (campo padrão do serviço ``notify`` do Home Assistant)
        para selecionar o ``media_player`` do Alexa — não ``data.entity_id``,
        que o ``alexa_media`` rejeita com 500.

        Não propaga exceções de TTS: em falha retorna ``{"ok": False, "error": ...}``
        para que o nó terminal do grafo decida o fluxo (ADR-0002).
        """
        try:
            result = await self.call_service(
                "notify",
                "alexa_media",
                {
                    "message": text,
                    "target": self._settings.alexa_media_entity,
                },
            )
        except httpx.HTTPError as exc:
            return {"ok": False, "error": str(exc)}
        # Sucesso: marca explicitamente para o nó do grafo (shape consistente).
        if isinstance(result, dict) and "ok" not in result:
            result = {"ok": True, **result}
        return result

    async def close(self) -> None:
        await self._client.aclose()

"""Known-check do catálogo na tool control_device + degradação graciosa.

O catálogo é injetado via ``set_catalog`` (lifespan aquece no boot). Quando o
catálogo está populado, ``entity_id`` fora dele é rejeitado antes de chamar o
HA (bloqueia alucinação do LLM). Quando vazio/ausente, a tool degrada
graciosamente: não bloqueia e segue para o HA (comportamento pré-catálogo).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from src.tools import home as home_mod
from src.tools.home import control_device

_HA_URL = "http://homeassistant.local:8123"


class _StubCatalog:
    """Catálogo fake para testar a tool sem abrir conexão com o HA."""

    def __init__(self, devices: dict[str, str]) -> None:
        self._devices = devices

    def is_known(self, entity_id: str) -> bool:
        return entity_id in self._devices

    def friendly_name(self, entity_id: str) -> str | None:
        return self._devices.get(entity_id)

    def has_devices(self) -> bool:
        return bool(self._devices)

    def as_context(self) -> str:
        return "\n".join(self._devices) if self._devices else ""


@pytest.fixture(autouse=True)
def _reset_catalog() -> Any:
    """Garante que nenhum catalog vaze entre testes."""
    home_mod.set_catalog(None)
    yield
    home_mod.set_catalog(None)


@pytest.mark.asyncio
async def test_control_device_rejects_unknown_entity_when_catalog_populated(
    test_settings,  # noqa: ANN001 — fixture
) -> None:
    """Catálogo populado + entity fora dele → mensagem amigável, sem chamar o HA."""
    home_mod.set_catalog(_StubCatalog({"switch.principal_sala": "Principal Sala"}))
    with respx.mock(base_url=_HA_URL, assert_all_called=False) as router:
        route = router.post("/api/services/switch/turn_on").mock(
            return_value=httpx.Response(200, json=[])
        )
        result = await control_device.ainvoke({"action": "on", "entity_id": "switch.fantasma"})
        # Não bateu no HA — bloqueado pelo known-check.
        assert not route.called
        assert "não encontrei" in result.lower() or "não conheço" in result.lower()


@pytest.mark.asyncio
async def test_control_device_proceeds_when_entity_known(
    test_settings,  # noqa: ANN001
) -> None:
    """Entity no catálogo → chama o HA e confirma com friendly_name do catálogo."""
    home_mod.set_catalog(_StubCatalog({"switch.principal_sala": "Principal Sala"}))
    with respx.mock(base_url=_HA_URL) as router:
        router.post("/api/services/switch/turn_on").mock(return_value=httpx.Response(200, json=[]))
        result = await control_device.ainvoke(
            {"action": "on", "entity_id": "switch.principal_sala"}
        )
        assert result == "Liguei o Principal Sala."


@pytest.mark.asyncio
async def test_control_device_degrades_when_catalog_empty(
    test_settings,  # noqa: ANN001
) -> None:
    """Catálogo vazio (HA fora no boot) → não bloqueia, segue para o HA."""
    home_mod.set_catalog(_StubCatalog({}))
    with respx.mock(base_url=_HA_URL) as router:
        router.post("/api/services/switch/turn_on").mock(return_value=httpx.Response(200, json=[]))
        result = await control_device.ainvoke({"action": "on", "entity_id": "switch.qualquer"})
        # Sem friendly_name no catálogo, cai no entity_id cru.
        assert "switch.qualquer" in result


@pytest.mark.asyncio
async def test_control_device_degrades_when_no_catalog_set(
    test_settings,  # noqa: ANN001
) -> None:
    """Sem catalog injetado (ex. test do tool_node legado) → segue para o HA."""
    with respx.mock(base_url=_HA_URL) as router:
        router.post("/api/services/light/turn_on").mock(return_value=httpx.Response(200, json=[]))
        result = await control_device.ainvoke({"action": "on", "entity_id": "light.luz_sala"})
        # Fallback estático (_ENTITY_ALIASES) continua funcionando sem catálogo.
        assert result == "Liguei o luz da sala."

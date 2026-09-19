"""DeviceCatalog: fetch do HA /api/states, filtro por domínio, cache/TTL e fallback."""

from __future__ import annotations

import httpx
import pytest
import respx

from src.services.catalog import DeviceCatalog

_HA_URL = "http://homeassistant.local:8123"


def _states_payload() -> list[dict[str, object]]:
    """Mistura de domínios permitidos e irrelevantes — só os permitidos entram."""
    return [
        {
            "entity_id": "switch.principal_sala",
            "attributes": {"friendly_name": "Principal Sala"},
        },
        {
            "entity_id": "switch.principal_cozinha",
            "attributes": {"friendly_name": "Principal Cozinha"},
        },
        {
            "entity_id": "light.luz_sala",
            "attributes": {"friendly_name": "Luz da Sala"},
        },
        {
            "entity_id": "media_player.alexa_sala",
            "attributes": {"friendly_name": "Alexa da Sala"},
        },
        # Domínios irrelevantes — devem ser filtrados fora.
        {
            "entity_id": "sensor.temperatura",
            "attributes": {"friendly_name": "Temperatura"},
        },
        {
            "entity_id": "automation.rotina",
            "attributes": {"friendly_name": "Rotina"},
        },
        {
            "entity_id": "script.malicioso",
            "attributes": {"friendly_name": "Malicioso"},
        },
    ]


async def _make_catalog(
    test_settings,  # noqa: ANN001 — fixture
    ttl_s: float = 300.0,
    fallback: dict[str, str] | None = None,
) -> DeviceCatalog:
    from src.services.ha_client import HomeAssistantClient

    ha = HomeAssistantClient(test_settings)
    return DeviceCatalog(
        ha_client=ha,
        ttl_s=ttl_s,
        fallback=fallback or {},
    )


@pytest.mark.asyncio
async def test_refresh_filters_allowed_domains(test_settings) -> None:  # type: ignore[no-untyped-def]
    catalog = await _make_catalog(test_settings)
    try:
        with respx.mock(base_url=_HA_URL) as router:
            router.get("/api/states").mock(return_value=httpx.Response(200, json=_states_payload()))
            await catalog.refresh()
        assert catalog.is_known("switch.principal_sala")
        assert catalog.is_known("switch.principal_cozinha")
        assert catalog.is_known("light.luz_sala")
        assert catalog.is_known("media_player.alexa_sala")
        # Domínios irrelevantes filtrados fora.
        assert not catalog.is_known("sensor.temperatura")
        assert not catalog.is_known("automation.rotina")
        assert not catalog.is_known("script.malicioso")
    finally:
        await catalog.aclose()


@pytest.mark.asyncio
async def test_friendly_name_returns_ha_attribute(test_settings) -> None:  # type: ignore[no-untyped-def]
    catalog = await _make_catalog(test_settings)
    try:
        with respx.mock(base_url=_HA_URL) as router:
            router.get("/api/states").mock(return_value=httpx.Response(200, json=_states_payload()))
            await catalog.refresh()
        assert catalog.friendly_name("switch.principal_sala") == "Principal Sala"
        assert catalog.friendly_name("light.luz_sala") == "Luz da Sala"
    finally:
        await catalog.aclose()


@pytest.mark.asyncio
async def test_friendly_name_unknown_returns_none(test_settings) -> None:  # type: ignore[no-untyped-def]
    catalog = await _make_catalog(test_settings)
    try:
        with respx.mock(base_url=_HA_URL) as router:
            router.get("/api/states").mock(return_value=httpx.Response(200, json=_states_payload()))
            await catalog.refresh()
        assert catalog.friendly_name("switch.inexistente") is None
    finally:
        await catalog.aclose()


@pytest.mark.asyncio
async def test_as_context_lists_devices(test_settings) -> None:  # type: ignore[no-untyped-def]
    """as_context() expõe entity_id + friendly_name para o LLM escolher."""
    catalog = await _make_catalog(test_settings)
    try:
        with respx.mock(base_url=_HA_URL) as router:
            router.get("/api/states").mock(return_value=httpx.Response(200, json=_states_payload()))
            await catalog.refresh()
        ctx = catalog.as_context()
        assert "switch.principal_sala" in ctx
        assert "Principal Sala" in ctx
        assert "switch.principal_cozinha" in ctx
        assert "Principal Cozinha" in ctx
        # Domínios irrelevantes não vazam para o contexto do LLM.
        assert "sensor.temperatura" not in ctx
        assert "script.malicioso" not in ctx
    finally:
        await catalog.aclose()


@pytest.mark.asyncio
async def test_cache_avoids_redundant_fetch(test_settings) -> None:  # type: ignore[no-untyped-def]
    """Dentro do TTL, is_known() não refaz GET /api/states."""
    catalog = await _make_catalog(test_settings, ttl_s=300.0)
    try:
        with respx.mock(base_url=_HA_URL) as router:
            route = router.get("/api/states").mock(
                return_value=httpx.Response(200, json=_states_payload())
            )
            await catalog.refresh()
            # Segunda leitura sem refresh — não deve bater no HA.
            assert catalog.is_known("switch.principal_sala")
            assert route.call_count == 1
    finally:
        await catalog.aclose()


@pytest.mark.asyncio
async def test_is_known_false_when_catalog_empty(test_settings) -> None:  # type: ignore[no-untyped-def]
    """Catálogo nunca aquecido → is_known() False para qualquer entity (degrada graciosa)."""
    catalog = await _make_catalog(test_settings)
    try:
        assert not catalog.is_known("switch.principal_sala")
        assert not catalog.is_known("switch.qualquer")
    finally:
        await catalog.aclose()


@pytest.mark.asyncio
async def test_is_known_false_when_no_allowed_domains_in_payload(test_settings) -> None:  # type: ignore[no-untyped-def]
    """HA responde só com sensores → catálogo vazio efetivo → degrade graciosa."""
    catalog = await _make_catalog(test_settings)
    try:
        with respx.mock(base_url=_HA_URL) as router:
            router.get("/api/states").mock(
                return_value=httpx.Response(
                    200,
                    json=[{"entity_id": "sensor.temp", "attributes": {"friendly_name": "Temp"}}],
                )
            )
            await catalog.refresh()
        assert not catalog.is_known("switch.principal_sala")
        assert catalog.as_context() == ""
    finally:
        await catalog.aclose()


@pytest.mark.asyncio
async def test_refresh_failure_falls_back_to_static(test_settings) -> None:  # type: ignore[no-untyped-def]
    """HA fora no boot → catálogo cai no fallback estático configurado."""
    fallback = {
        "switch.principal_sala": "Principal Sala",
        "light.luz_sala": "Luz da Sala",
    }
    catalog = await _make_catalog(test_settings, fallback=fallback)
    try:
        with respx.mock(base_url=_HA_URL) as router:
            router.get("/api/states").mock(side_effect=httpx.ConnectError("HA fora"))
            await catalog.refresh()
        assert catalog.is_known("switch.principal_sala")
        assert catalog.is_known("light.luz_sala")
        assert not catalog.is_known("switch.inexistente")
        assert catalog.friendly_name("switch.principal_sala") == "Principal Sala"
        ctx = catalog.as_context()
        assert "switch.principal_sala" in ctx
    finally:
        await catalog.aclose()


@pytest.mark.asyncio
async def test_refresh_failure_no_fallback_means_empty(test_settings) -> None:  # type: ignore[no-untyped-def]
    """HA fora e sem fallback → catálogo vazio (degrade graciosa, sem crash)."""
    catalog = await _make_catalog(test_settings)
    try:
        with respx.mock(base_url=_HA_URL) as router:
            router.get("/api/states").mock(side_effect=httpx.ConnectError("HA fora"))
            await catalog.refresh()
        assert not catalog.is_known("switch.principal_sala")
        assert catalog.as_context() == ""
    finally:
        await catalog.aclose()

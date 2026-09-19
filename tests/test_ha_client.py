"""Task 2.1 — HomeAssistantClient: sucesso, timeout e erro 4xx/5xx no speak()."""

from __future__ import annotations

import logging

import httpx
import pytest
import respx

from src.services.ha_client import HomeAssistantClient

_HA_URL = "http://homeassistant.local:8123"


@pytest.mark.asyncio
async def test_call_service_success(test_settings) -> None:  # type: ignore[no-untyped-def]
    client = HomeAssistantClient(test_settings)
    with respx.mock(base_url=_HA_URL) as router:
        route = router.post("/api/services/homeassistant/toggle").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        result = await client.call_service("homeassistant", "toggle", {"entity_id": "switch.x"})
        assert result == {"ok": True}
        assert route.called
    await client.close()


@pytest.mark.asyncio
async def test_call_service_logs_4xx_error(test_settings, caplog: pytest.LogCaptureFixture) -> None:  # type: ignore[no-untyped-def]
    """call_service loga (ERROR) status + body do HA ANTES de propagar o erro."""
    client = HomeAssistantClient(test_settings)
    with respx.mock(base_url=_HA_URL) as router:
        router.post("/api/services/homeassistant/toggle").mock(
            return_value=httpx.Response(404, text="entity not found")
        )
        with (
            caplog.at_level(logging.ERROR, logger="uvicorn.error"),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await client.call_service("homeassistant", "toggle", {"entity_id": "switch.x"})
    assert "404" in caplog.text
    assert "entity not found" in caplog.text
    await client.close()


@pytest.mark.asyncio
async def test_speak_success_returns_ok_true(test_settings) -> None:  # type: ignore[no-untyped-def]
    client = HomeAssistantClient(test_settings)
    with respx.mock(base_url=_HA_URL) as router:
        route = router.post("/api/services/notify/alexa_media").mock(
            return_value=httpx.Response(200, json=[])
        )
        result = await client.speak("olá")
        assert result.get("ok") is True
        # Payload usa `target` (não data.entity_id) — formato que o HA aceita.
        sent = route.calls.last.request.content.decode()
        assert '"target"' in sent
        assert "media_player.alexa_sala" in sent
        assert '"data"' not in sent
    await client.close()


@pytest.mark.asyncio
async def test_speak_timeout_returns_ok_false(test_settings) -> None:  # type: ignore[no-untyped-def]
    client = HomeAssistantClient(test_settings)
    with respx.mock(base_url=_HA_URL) as router:
        router.post("/api/services/notify/alexa_media").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        result = await client.speak("olá")
        assert result.get("ok") is False
        assert "timeout" in str(result.get("error"))
    await client.close()


@pytest.mark.asyncio
async def test_speak_4xx_returns_ok_false(test_settings) -> None:  # type: ignore[no-untyped-def]
    client = HomeAssistantClient(test_settings)
    with respx.mock(base_url=_HA_URL) as router:
        router.post("/api/services/notify/alexa_media").mock(
            return_value=httpx.Response(401, text="unauthorized")
        )
        result = await client.speak("olá")
        assert result.get("ok") is False
    await client.close()

"""Task 3.1-3.3 — tools: weather, search, home (schema + fallbacks)."""

from __future__ import annotations

import httpx
import pytest
import respx

from src.tools.home import HomeInput
from src.tools.weather import get_weather

_OPEN_METEO_GEO = "https://geocoding-api.open-meteo.com"
_OPEN_METEO_FCST = "https://api.open-meteo.com"


def test_home_input_rejects_invalid_entity_id() -> None:
    with pytest.raises(ValueError):
        HomeInput(action="on", entity_id="script.malicioso")  # noqa: F841


def test_home_input_accepts_valid_domains() -> None:
    for entity in ("switch.tomada", "light.luz", "media_player.alexa"):
        HomeInput(action="toggle", entity_id=entity)


def test_weather_success() -> None:
    with respx.mock as router:  # type: ignore[attr-defined]
        router.get(f"{_OPEN_METEO_GEO}/v1/search").mock(
            return_value=httpx.Response(
                200,
                json={"results": [{"latitude": -23.5, "longitude": -46.6}]},
            )
        )
        router.get(f"{_OPEN_METEO_FCST}/v1/forecast").mock(
            return_value=httpx.Response(
                200,
                json={
                    "current": {"weather_code": 80},
                    "daily": {
                        "temperature_2m_max": [28.4],
                        "temperature_2m_min": [18.9],
                        "weather_code": [80],
                    },
                },
            )
        )
        summary = get_weather.invoke({"location": "São Paulo"})
        assert "28" in summary and "19" in summary


def test_weather_city_not_found_returns_fallback() -> None:
    with respx.mock as router:  # type: ignore[attr-defined]
        router.get(f"{_OPEN_METEO_GEO}/v1/search").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        summary = get_weather.invoke({"location": "Cidade Inexistente XYZ"})
        assert "Não encontrei essa cidade" in summary


def test_weather_network_error_returns_fallback() -> None:
    with respx.mock as router:  # type: ignore[attr-defined]
        router.get(f"{_OPEN_METEO_GEO}/v1/search").mock(side_effect=httpx.ConnectError("sem rede"))
        summary = get_weather.invoke({"location": "São Paulo"})
        assert "Não consegui obter o clima" in summary

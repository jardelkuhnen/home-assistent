"""Tool de clima via Open-Meteo (geocoding + forecast).

Design defensivo: em qualquer falha (rede, cidade não encontrada) retorna uma
frase de fallback falável — nunca propaga exceção ao motor.
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

_OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
_OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_S = 5.0

# Tradução parcial dos weather codes do Open-Meteo para linguagem falável.
_WMO: dict[int, str] = {
    0: "céu limpo",
    1: "predominantemente limpo",
    2: "parcialmente nublado",
    3: "nublado",
    45: "neblina",
    48: "neblina com geada",
    51: "garoa leve",
    53: "garoa moderada",
    55: "garoa intensa",
    61: "chuva leve",
    63: "chuva moderada",
    65: "chuva forte",
    71: "neve fraca",
    73: "neve moderada",
    75: "neve intensa",
    80: "pancadas de chuva",
    81: "pancadas de chuva moderadas",
    82: "pancadas de chuva fortes",
    95: "tempestade",
    96: "tempestade com granizo",
    99: "tempestade severa com granizo",
}


class WeatherInput(BaseModel):
    location: str = Field(..., description="Nome da cidade, ex: São Paulo")


class WeatherOutput(BaseModel):
    summary: str


def _format(latitude: float, longitude: float) -> str:
    """Busca o forecast e comprime em frase falável."""
    with httpx.Client(timeout=_TIMEOUT_S) as client:
        forecast = client.get(
            _OPEN_METEO_FORECAST,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                "current": "weather_code",
                "forecast_days": 1,
                "timezone": "auto",
            },
        ).json()

    daily = forecast.get("daily", {})
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    codes = daily.get("weather_code", [])
    current = forecast.get("current", {}).get("weather_code")

    if not max_temps or not min_temps:
        return "Não consegui obter o clima agora."

    temp_max = round(max_temps[0])
    temp_min = round(min_temps[0])
    condicao = _WMO.get(current if current is not None else (codes[0] if codes else 0), "")
    base = f"Máxima de {temp_max}, mínima de {temp_min}"
    return f"{base}, {condicao}" if condicao else base


@tool("get_weather", args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """Retorna a previsão do tempo para uma cidade, em frase curta para voz."""
    try:
        with httpx.Client(timeout=_TIMEOUT_S) as client:
            geo = client.get(
                _OPEN_METEO_GEOCODING, params={"name": location, "count": 1, "language": "pt"}
            ).json()

        results = geo.get("results") or []
        if not results:
            return "Não encontrei essa cidade."

        first = results[0]
        return _format(float(first["latitude"]), float(first["longitude"]))
    except httpx.HTTPError:
        return "Não consegui obter o clima agora."

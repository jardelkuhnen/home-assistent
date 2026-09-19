"""Fixtures compartilhadas: Settings de teste determinístico."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.config import Settings, get_settings

_ENV: dict[str, str] = {
    "LLM_PROVIDER": "gemini",
    "GEMINI_API_KEY": "test-gemini",
    "OPENAI_API_KEY": "test-openai",
    "OPENAI_API_BASE": "https://api.openai.com/v1",
    "HA_URL": "http://homeassistant.local:8123",
    "HA_TOKEN": "test-ha-token",
    "ALEXA_MEDIA_ENTITY": "media_player.alexa_sala",
    "TAVILY_API_KEY": "test-tavily",
    "BRAIN_API_KEY": "test-brain-key",
    "BRAIN_URL": "http://localhost:8000",
    "WHISPER_MODEL": "tiny",
    "TELEGRAM_BOT_TOKEN": "test-telegram-token",
    "ALLOWED_USERS": "11111,22222",
}


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Iterator[None]:
    """Garante cache de get_settings limpo entre testes."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def reset_catalog_globals() -> Iterator[None]:
    """Reseta os globais de catálogo (tool + grafo) entre testes.

    O lifespan do ``api.py`` injeta o catálogo em ``src.tools.home`` e
    ``src.graph.nodes`` via ``set_catalog``. Sem reset, um teste que sobe o
    TestClient (lifespan real) vazaria o catálogo para testes posteriores
    que não o monkeypatch.
    """
    from src.graph import nodes as nodes_mod
    from src.tools import home as home_mod

    home_mod.set_catalog(None)
    nodes_mod.set_catalog(None)
    yield
    home_mod.set_catalog(None)
    nodes_mod.set_catalog(None)


@pytest.fixture
def test_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Popula o ambiente com variáveis de teste determinísticas."""
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)
    return _ENV


@pytest.fixture
def test_settings(test_env: dict[str, str]) -> Settings:
    get_settings.cache_clear()
    return get_settings()

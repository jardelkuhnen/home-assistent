"""Task 1.4 — Settings: carrega obrigatórios e rejeita extras (extra='forbid')."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings


def test_settings_loads_required_fields(test_env: dict[str, str]) -> None:
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_provider == "gemini"
    assert settings.gemini_api_key.get_secret_value() == "test-gemini"
    assert settings.brain_api_key.get_secret_value() == "test-brain-key"
    # AnyHttpUrl normaliza com barra final.
    assert str(settings.brain_url).rstrip("/") == "http://localhost:8000"


def test_settings_rejects_unknown_field(test_env: dict[str, str]) -> None:
    # extra="forbid": campos extras passados explicitamente são rejeitados
    # (proteção contra typos em chaves de configuração).
    with pytest.raises(ValidationError):
        Settings(unknown_field="surpresa")

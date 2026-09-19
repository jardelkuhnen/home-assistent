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


def test_settings_supports_local_ollama(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    settings = get_settings()

    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_model == "llama3.2:3b"


def test_settings_wake_word_defaults(test_env: dict[str, str]) -> None:
    """Sem nenhuma variável nova no ambiente, valem os defaults (.env antigo continua válido)."""
    settings = get_settings()

    assert settings.wake_word_model == "hey_jarvis"
    assert settings.wake_word_threshold == 0.5
    assert settings.end_silence_s == 1.0
    assert settings.no_speech_timeout_s == 5.0
    assert settings.max_record_s == 15.0


def test_settings_wake_word_overridable_by_env(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WAKE_WORD_MODEL", "alexa")
    monkeypatch.setenv("WAKE_WORD_THRESHOLD", "0.7")
    monkeypatch.setenv("END_SILENCE_S", "1.5")
    monkeypatch.setenv("NO_SPEECH_TIMEOUT_S", "3")
    monkeypatch.setenv("MAX_RECORD_S", "10")
    settings = get_settings()

    assert settings.wake_word_model == "alexa"
    assert settings.wake_word_threshold == 0.7
    assert settings.end_silence_s == 1.5
    assert settings.no_speech_timeout_s == 3.0
    assert settings.max_record_s == 10.0

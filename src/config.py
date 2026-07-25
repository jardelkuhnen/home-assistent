"""Configuração tipada e factory do motor cognitivo.

Centraliza:
* ``Settings`` — leitura estrita de variáveis de ambiente (``extra="forbid"``).
* ``get_settings()`` — acesso com cache (``lru_cache``).
* ``CognitiveMotor`` — Protocol que desacopla o grafo do LLM concreto (ADR-0001).
* ``get_llm()`` — factory que devolve Gemini ou backend OpenAI-compatível.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import AnyHttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Variáveis de ambiente do Cérebro e do Satélite.

    ``extra="forbid"`` rejeita variáveis desconhecidas — proteção contra typos
    silenciosos em chaves de configuração.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="forbid",
        case_sensitive=False,
    )

    # --- Motor cognitivo ---
    llm_provider: Literal["gemini", "openai"]
    gemini_api_key: SecretStr
    openai_api_key: SecretStr
    openai_api_base: str = ""
    llm_timeout_s: float = 30.0

    # --- Home Assistant / Alexa ---
    ha_url: AnyHttpUrl
    ha_token: SecretStr
    ha_timeout_s: float = 5.0
    alexa_media_entity: str

    # --- Tavily (busca web) ---
    tavily_api_key: SecretStr

    # --- API do Cérebro (auth) ---
    brain_api_key: SecretStr

    # --- Satélite → Cérebro (fechado na Task 5.3) ---
    brain_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8000")

    # --- STT (faster-whisper) ---
    whisper_model: str = "small"


@lru_cache
def get_settings() -> Settings:
    """Retorna a instância singleton de ``Settings``."""
    return Settings()


@runtime_checkable
class CognitiveMotor(Protocol):
    """Contrato do motor cognitivo — compatível com ``BaseChatModel``.

    O grafo depende apenas deste Protocol, nunca de uma classe concreta de
    LLM, permitindo trocar o motor sem alterar o binding (ADR-0001).

    Retornos anotados como ``Any``: os backends concretos devolvem
    ``Runnable``/``AIMessage`` especializados, que são covariantemente
    compatíveis — o que importa ao grafo é o contrato, não o tipo exato.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any: ...

    async def ainvoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> Any: ...


def get_llm() -> CognitiveMotor:
    """Factory do motor cognitivo.

    Seleciona o backend por ``llm_provider`` e aplica timeout explícito em
    ambos os caminhos (baseline de segurança item 2).
    """
    settings = get_settings()

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.gemini_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_api_base or None,
        timeout=settings.llm_timeout_s,
    )

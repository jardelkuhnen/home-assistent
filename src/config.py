"""Configuração tipada e factory do motor cognitivo.

Centraliza:
* ``Settings`` — leitura estrita de variáveis de ambiente (``extra="forbid"``).
* ``get_settings()`` — acesso com cache (``lru_cache``).
* ``CognitiveMotor`` — Protocol que desacopla o grafo do LLM concreto (ADR-0001).
* ``get_llm()`` — factory que devolve Gemini, OpenAI ou Ollama local.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import AnyHttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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
    llm_provider: Literal["gemini", "openai", "ollama"]
    gemini_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openai_api_base: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    llm_timeout_s: float = 30.0

    # --- Home Assistant / Alexa ---
    ha_url: AnyHttpUrl
    ha_token: SecretStr
    ha_timeout_s: float = 5.0
    alexa_media_entity: str
    # TTL do cache do catálogo de dispositivos (DeviceCatalog). Default 300s;
    # como tem default, não quebra .env existente (extra="forbid").
    catalog_ttl_s: float = 300.0

    # --- Tavily (busca web) ---
    tavily_api_key: SecretStr

    # --- API do Cérebro (auth) ---
    brain_api_key: SecretStr

    # --- Satélite → Cérebro (fechado na Task 5.3) ---
    brain_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8000")
    # Um turno completo (LLM + tools + 2ª chamada + TTS) pode levar bem mais
    # que o timeout do HA: timeout próprio, maior, para o hop Satélite→Cérebro.
    brain_timeout_s: float = 90.0

    # --- STT (faster-whisper) ---
    whisper_model: str = "small"

    # --- Telegram ↔ Cérebro (canal de texto, isolado da Alexa) ---
    # Token do bot criado no BotFather. Default vazio: o Cérebro não precisa
    # dessas vars para subir — só o ``telegram_bot.py`` as consome. Assim o
    # brain inicia mesmo sem o canal Telegram configurado.
    telegram_bot_token: SecretStr = SecretStr("")
    # IDs de usuários autorizados a conversar com o bot. No .env vem como
    # string separada por vírgula ("11111,22222"); o validador converte para
    # lista de ints. Whitelist de segurança — mensagens de outros usuários
    # são ignoradas.
    #
    # ``NoDecode`` evita que o EnvSettingsSource tente JSON-decodificar o
    # valor (lista é um tipo "complexo"): a string crua chega ao
    # ``field_validator`` abaixo, que faz o parsing por vírgula.
    allowed_users: Annotated[list[int], NoDecode] = []

    @field_validator("allowed_users", mode="before")
    @classmethod
    def _parse_allowed_users(cls, v: Any) -> Any:
        """Aceita ``"11111,22222"`` (env string) e devolve ``[11111, 22222]``.

        ``case_sensitive=False`` no pydantic-settings cobre apenas chaves de
        env, não valores — por isso o parsing manual. Também tolera espaços,
        valores já em lista e ``None`` (default ausente).
        """
        if v is None:
            return []
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            return [int(p) for p in parts]
        return v


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
    todos os caminhos (baseline de segurança item 2).
    """
    settings = get_settings()

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=settings.gemini_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
        )

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            client_kwargs={"timeout": settings.llm_timeout_s},
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_api_base or None,
        timeout=settings.llm_timeout_s,
    )

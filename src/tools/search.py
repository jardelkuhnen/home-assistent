"""Tool de busca web via Tavily, resultado digerido para voz.

Design defensivo: falhas de rede/API retornam frase de fallback, nunca
propagam exceção ao motor.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import get_settings

_FALLBACK = "Não encontrei nada sobre isso."


class SearchInput(BaseModel):
    query: str = Field(..., description="Pergunta ou termo de busca")
    max_results: int = Field(default=3, ge=1, le=10)


class SearchOutput(BaseModel):
    answer: str


@tool("web_search", args_schema=SearchInput)
def web_search(query: str, max_results: int = 3) -> str:
    """Busca na web e devolve uma resposta curta, em poucas frases."""
    try:
        # Import tardio: evita carregar o Tavily em ambientes sem a dep.
        from tavily import TavilyClient  # type: ignore[import-untyped]

        settings = get_settings()
        client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
        response = client.search(query=query, max_results=max_results)

        answer = response.get("answer")
        if answer:
            return str(answer).strip()

        results = response.get("results") or []
        snippets = [r.get("content", "").strip() for r in results if r.get("content")]
        if snippets:
            return " ".join(snippets)[:300]

        return _FALLBACK
    except Exception:  # noqa: BLE001 — ferramenta defensiva para voz
        return _FALLBACK

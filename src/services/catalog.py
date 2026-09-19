"""Catálogo de dispositivos do Home Assistant para resolução NL → entity_id.

O motor cognitivo precisa escolher um ``entity_id`` válido a partir da fala
do usuário ("desligar principal cozinha"). Para isso, injetamos no contexto
do LLM o catálogo real de dispositivos (``friendly_name`` + ``entity_id``),
filtrado pelos domínios permitidos. A validação no código (``is_known``)
bloqueia ``entity_id`` alucinado que não exista no HA.

Fonte primária: ``GET /api/states`` do HA, com cache em memória por TTL.
Fallback: catálogo estático (``fallback``) quando o HA está indisponível no
boot — degradação graciosa: o sistema sobe, e a validação relaxa (catálogo
vazio ⇒ ``is_known`` sempre False ⇒ a tool não bloqueia, deixando o HA
responder/errar por si, como já fazia antes do catálogo).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from src.services.ha_client import HomeAssistantClient

logger = logging.getLogger("uvicorn.error")

# Domínios que a tool ``control_device`` aceita — mesma lista do validator
# em ``HomeInput``. Filtrar aqui evita vazar sensores/automações/scripts para
# o contexto do LLM (superfície de ataque menor + menos ruído).
_ALLOWED_DOMAINS = ("switch", "light", "media_player")


class DeviceCatalog:
    """Catálogo de dispositivos acionáveis, com cache em memória.

    * ``refresh()`` — busca ``/api/states`` no HA e popula o cache. Em falha
      (``httpx.HTTPError``), cai no ``fallback`` estático e loga o motivo.
    * ``is_known(entity_id)`` / ``friendly_name(entity_id)`` — leitura sync do
      cache (a tool é chamada em contexto async, mas o cache já está quente
      após o warm-up do lifespan; se vazio, devolve False/None).
    * ``as_context()`` — texto para injetar como ``SystemMessage`` no LLM.
    * ``aclose()`` — fecha o HA client injetado (no test; em produção o client
      é compartilhado e fechado pelo lifespan — usar ``shared=False``).
    """

    def __init__(
        self,
        ha_client: HomeAssistantClient,
        ttl_s: float = 300.0,
        fallback: dict[str, str] | None = None,
    ) -> None:
        self._ha = ha_client
        self._ttl_s = ttl_s
        self._fallback = dict(fallback or {})
        # entity_id -> friendly_name. Vazio até o primeiro refresh bem-sucedido
        # (ou fallback). Vazio ⇒ degrade graciosa: is_known sempre False.
        self._devices: dict[str, str] = {}
        self._fetched_at: float = 0.0

    async def refresh(self) -> None:
        """Busca ``/api/states`` no HA e popula o cache; em falha usa fallback."""
        try:
            raw = await self._ha.list_states()
        except httpx.HTTPError as exc:
            logger.warning("catalog: HA indisponível (%s) — usando fallback estático", exc)
            self._devices = dict(self._fallback)
            self._fetched_at = time.monotonic()
            return

        devices: dict[str, str] = {}
        for entry in raw:
            entity_id = entry.get("entity_id") if isinstance(entry, dict) else None
            if not isinstance(entity_id, str):
                continue
            if not entity_id.startswith(tuple(f"{d}." for d in _ALLOWED_DOMAINS)):
                continue
            raw_attrs = entry.get("attributes")
            attrs = raw_attrs if isinstance(raw_attrs, dict) else {}
            friendly = attrs.get("friendly_name")
            devices[entity_id] = friendly if isinstance(friendly, str) and friendly else entity_id

        self._devices = devices
        self._fetched_at = time.monotonic()
        logger.info("catalog: %d dispositivos carregados", len(self._devices))

    def _ensure_fresh(self) -> None:
        """Rebusca em background-friendly: se o TTL expirou, marca para refresh.

        O refresh é async; esta leitura sync apenas invalida o cache expirado
        para a próxima chamada de ``refresh`` (pelo lifespan/periodic). Não
        bloqueia ``is_known`` — prefere servir staled do que negar.
        """
        if self._devices and self._ttl_s > 0 and time.monotonic() - self._fetched_at > self._ttl_s:
            # Cache expirado: servimos o atual; o refresh efetivo roda no próximo
            # ciclo do lifespan. A flag evita negar serviço.
            logger.debug("catalog: cache expirado, servindo staled até próximo refresh")

    def is_known(self, entity_id: str) -> bool:
        """True se o ``entity_id`` está no catálogo (HA vivo) ou no fallback."""
        self._ensure_fresh()
        return entity_id in self._devices

    def friendly_name(self, entity_id: str) -> str | None:
        """``friendly_name`` do HA, ou ``None`` se desconhecido/vazio."""
        self._ensure_fresh()
        return self._devices.get(entity_id)

    def has_devices(self) -> bool:
        """True se o catálogo tem ao menos um dispositivo (HA ou fallback)."""
        return bool(self._devices)

    def as_context(self) -> str:
        """Texto para injetar como ``SystemMessage`` no LLM.

        Lista ``friendly_name (entity_id)`` um por linha. Vazio se o catálogo
        estiver vazio — o LLM segue sem catálogo (comportamento pré-catálogo).
        """
        self._ensure_fresh()
        if not self._devices:
            return ""
        lines = ["Dispositivos disponíveis (use exatamente o entity_id ao chamar control_device):"]
        for entity_id, friendly in sorted(self._devices.items()):
            lines.append(f"- {friendly} ({entity_id})")
        return "\n".join(lines)

    async def aclose(self) -> None:
        """Fecha o HA client injetado (uso em testes com client dedicado)."""
        await self._ha.close()

"""Telegram ↔ Cérebro — canal de texto bidirecional, isolado da Alexa.

Espelho do ``satelite.py``: long polling com ``python-telegram-bot``, hop
HTTP ao Cérebro reusando ``brain_url``/``brain_api_key``/``brain_timeout_s``.
Mensagens trafegam com ``metadata.source="telegram"`` para que o grafo **pule**
o nó ``speak`` (Alexa) — o canal de texto nunca aciona a voz.

Segurança: whitelist ``ALLOWED_USERS``; mensagens de outros usuários são
ignoradas (log ``WARNING`` com o ``user.id``). O bot nunca cai: erros viram
mensagem genérica ao usuário.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx

from src.config import Settings, get_settings

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

logger = logging.getLogger("telegram_bot")

# Reenvio do indicador "digitando..." a cada N segundos enquanto o Cérebro
# processa (o Telegram oculta o indicador após ~5s).
_TYPING_REFRESH_S = 4.0
_FALLBACK_REPLY = "Algo deu errado, tente de novo."


async def send_to_brain(
    text: str,
    chat_id: int,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """POST /chat ao Cérebro com ``source="telegram"`` e header ``X-API-Key``.

    Espelha ``satelite.send_to_brain`` (mesmo padrão de hop), acrescentando o
    ``metadata`` que isola o canal de texto do caminho de voz.
    """
    s = settings or get_settings()
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            base_url=str(s.brain_url),
            headers={"X-API-Key": s.brain_api_key.get_secret_value()},
            timeout=s.brain_timeout_s,
        )
    payload: dict[str, Any] = {
        "text": text,
        "metadata": {"source": "telegram", "session_id": f"telegram_{chat_id}"},
    }
    try:
        response = await client.post("/chat", json=payload)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
    finally:
        if owns_client:
            await client.aclose()


async def _typing_loop(bot: Any, chat_id: int, stop: asyncio.Event) -> None:
    """Reenvia ``typing`` a cada ~4s até ``stop`` ser setado.

    Usa ``wait_for`` no evento para cancelar imediatamente quando o turno
    acaba (sem esperar o intervalo inteiro). Falhas de envio do indicador são
    silenciadas — o typing é cosmético e não pode derrubar o handler.
    """
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:  # noqa: BLE001 - indicador é cosmético
            logger.debug("typing action failed", exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=_TYPING_REFRESH_S)
        except TimeoutError:
            continue


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler de mensagens de texto: whitelist → typing → cérebro → reply.

    Nunca propaga exceções: erros viram mensagem genérica ao usuário, e o bot
    segue rodando (decisão de UX: o bot nunca cai).
    """
    user = update.effective_user
    uid = getattr(user, "id", None)
    settings = get_settings()

    if uid not in settings.allowed_users:
        logger.warning("unauthorized user_id=%s", uid)
        return

    message = update.message
    text = message.text if message is not None else None
    if not text:
        return

    chat_id = message.chat_id

    stop = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(context.bot, chat_id, stop))
    try:
        result = await send_to_brain(text, chat_id, settings)
        reply = result.get("reply")
        if not reply:
            reply = _FALLBACK_REPLY
        await context.bot.send_message(chat_id=chat_id, text=reply)
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        logger.error("brain request failed: %s", exc, exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text=_FALLBACK_REPLY)
    except Exception as exc:  # noqa: BLE001 - bot nunca cai
        logger.error("unexpected error handling message: %s", exc, exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text=_FALLBACK_REPLY)
    finally:
        stop.set()
        try:
            await typing_task
        except Exception:  # noqa: BLE001
            logger.debug("typing task ended with error", exc_info=True)


async def main() -> None:
    """Sobe o bot em long polling."""
    from telegram.ext import ApplicationBuilder, MessageHandler, filters

    settings = get_settings()
    token = settings.telegram_bot_token.get_secret_value()
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN não configurado. Crie um bot no BotFather "
            "(/newbot) e preencha no .env."
        )
    if not settings.allowed_users:
        logger.warning(
            "ALLOWED_USERS vazio — ninguém poderá conversar com o bot. "
            "Defina seu user_id no .env (ALLOWED_USERS=11111,22222)."
        )
    application = ApplicationBuilder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT, handle_text_message))
    logger.info("Telegram bot iniciado | allowed_users=%s", settings.allowed_users)
    await application.initialize()
    await application.start()
    try:
        await application.updater.start_polling()  # type: ignore[union-attr]
        # Roda até interrompido (Ctrl-C).
        while True:
            await asyncio.sleep(3600)
    finally:
        await application.updater.stop()  # type: ignore[union-attr]
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

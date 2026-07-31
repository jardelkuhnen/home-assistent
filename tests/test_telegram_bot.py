"""Telegram bot — whitelist, payload, fallback e typing."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest
import respx

import telegram_bot
from src.config import get_settings

# --- Fakes mínimos de PTB (Update/Context) construídos à mão ---


class _User:
    def __init__(self, uid: int) -> None:
        self.id = uid


class _Message:
    def __init__(self, text: str, chat_id: int) -> None:
        self.text = text
        self.chat_id = chat_id


class _Update:
    def __init__(self, uid: int, text: str, chat_id: int) -> None:
        self.effective_user = _User(uid)
        self.message = _Message(text, chat_id)


class _Bot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.chat_actions: list[tuple[int, str]] = []

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.chat_actions.append((chat_id, action))

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


class _Context:
    def __init__(self, bot: _Bot) -> None:
        self.bot = bot


@pytest.mark.asyncio
async def test_unauthorized_user_ignored_and_logged(
    test_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Usuário fora de ALLOWED_USERS → log WARNING e send_to_brain não chamado."""
    get_settings.cache_clear()
    caplog.set_level(logging.WARNING, logger="telegram_bot")

    calls: list[Any] = []

    async def fake_send_to_brain(*args: object, **kwargs: object) -> dict[str, Any]:  # noqa: ARG001
        calls.append(args)
        return {"reply": "x"}

    monkeypatch.setattr(telegram_bot, "send_to_brain", fake_send_to_brain)

    bot = _Bot()
    update = _Update(uid=99999, text="oi", chat_id=99999)
    await telegram_bot.handle_text_message(update, _Context(bot))

    assert calls == []
    assert bot.sent == []
    assert "unauthorized user_id=99999" in caplog.text


@pytest.mark.asyncio
async def test_authorized_user_calls_send_to_brain(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    get_settings.cache_clear()

    calls: list[tuple[str, int]] = []

    async def fake_send_to_brain(
        text: str, chat_id: int, *args: object, **kwargs: object
    ) -> dict[str, Any]:  # noqa: ARG001
        calls.append((text, chat_id))
        return {"reply": "pronto"}

    monkeypatch.setattr(telegram_bot, "send_to_brain", fake_send_to_brain)

    bot = _Bot()
    update = _Update(uid=11111, text="clima", chat_id=123)
    await telegram_bot.handle_text_message(update, _Context(bot))

    assert calls == [("clima", 123)]
    assert bot.sent == [(123, "pronto")]


@pytest.mark.asyncio
async def test_send_to_brain_payload_has_telegram_metadata(test_env: dict[str, str]) -> None:
    """send_to_brain monta payload com source=telegram e session_id=telegram_<chat>."""
    get_settings.cache_clear()
    with respx.mock(base_url="http://localhost:8000") as router:
        route = router.post("/chat").mock(
            return_value=httpx.Response(200, json={"reply": "ok", "spoken": False})
        )
        result = await telegram_bot.send_to_brain("clima", chat_id=456)

    assert result["reply"] == "ok"
    sent_headers = route.calls.last.request.headers
    assert sent_headers.get("X-API-Key") == "test-brain-key"
    payload = json.loads(route.calls.last.request.read())
    assert payload["text"] == "clima"
    assert payload["metadata"] == {"source": "telegram", "session_id": "telegram_456"}


@pytest.mark.asyncio
async def test_empty_reply_falls_back_to_generic(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    get_settings.cache_clear()

    async def fake_send_to_brain(*args: object, **kwargs: object) -> dict[str, Any]:  # noqa: ARG001
        return {"reply": ""}

    monkeypatch.setattr(telegram_bot, "send_to_brain", fake_send_to_brain)

    bot = _Bot()
    update = _Update(uid=11111, text="oi", chat_id=123)
    await telegram_bot.handle_text_message(update, _Context(bot))

    assert bot.sent == [(123, "Algo deu errado, tente de novo.")]


@pytest.mark.asyncio
async def test_brain_timeout_sends_fallback_and_does_not_propagate(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    get_settings.cache_clear()

    async def fake_send_to_brain(*args: object, **kwargs: object) -> dict[str, Any]:  # noqa: ARG001
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(telegram_bot, "send_to_brain", fake_send_to_brain)

    bot = _Bot()
    update = _Update(uid=11111, text="oi", chat_id=123)
    # Não deve levantar.
    await telegram_bot.handle_text_message(update, _Context(bot))

    assert bot.sent == [(123, "Algo deu errado, tente de novo.")]


@pytest.mark.asyncio
async def test_typing_sent_at_least_once_and_stops(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    get_settings.cache_clear()

    async def fake_send_to_brain(*args: object, **kwargs: object) -> dict[str, Any]:  # noqa: ARG001
        # Dá tempo ao loop de typing disparar pelo menos um send_chat_action.
        await asyncio.sleep(0.05)
        return {"reply": "pronto"}

    monkeypatch.setattr(telegram_bot, "send_to_brain", fake_send_to_brain)

    bot = _Bot()
    update = _Update(uid=11111, text="oi", chat_id=123)
    await telegram_bot.handle_text_message(update, _Context(bot))

    assert len(bot.chat_actions) >= 1
    assert all(action == "typing" for _, action in bot.chat_actions)
    # Após o turno, a resposta foi enviada (loop de typing encerrado).
    assert bot.sent == [(123, "pronto")]

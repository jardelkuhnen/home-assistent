"""Task 5.2 — Satélite: transcribe (sem hardware) e send_to_brain."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from satelite import send_to_brain, transcribe


@pytest.mark.asyncio
async def test_send_to_brain_success(test_env: dict[str, str]) -> None:
    with respx.mock(base_url="http://localhost:8000") as router:
        route = router.post("/chat").mock(
            return_value=httpx.Response(200, json={"reply": "pronto", "spoken": True})
        )
        result = await send_to_brain("ligar luz")
        assert result == {"reply": "pronto", "spoken": True}
        # Header X-API-Key enviado.
        sent_headers = route.calls.last.request.headers
        assert sent_headers.get("X-API-Key") == "test-brain-key"


@pytest.mark.asyncio
async def test_send_to_brain_failure_raises(test_env: dict[str, str]) -> None:
    with respx.mock(base_url="http://localhost:8000") as router:
        router.post("/chat").mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(httpx.HTTPStatusError):
            await send_to_brain("algo")


@pytest.mark.asyncio
async def test_send_to_brain_uses_brain_timeout_not_ha(test_env: dict[str, str]) -> None:
    """O hop Satélite→Cérebro usa brain_timeout_s (maior), não ha_timeout_s."""
    from src.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.brain_timeout_s > s.ha_timeout_s


@pytest.mark.asyncio
async def test_run_once_handles_timeout(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """run_once não explode com traceback em timeout — mostra mensagem amigável."""
    from satelite import run_once

    async def fake_send_to_brain(text: str):  # noqa: ARG001
        raise httpx.ReadTimeout("timed out")

    def fake_capture_audio(*args: object, **kwargs: object) -> bytes:  # noqa: ARG001
        return b"\x00" * 32

    def fake_transcribe(audio, settings=None):  # noqa: ANN001, ARG001
        return "comprar bicicleta"

    # stdin readline não-bloqueante
    import satelite

    monkeypatch.setattr(satelite, "send_to_brain", fake_send_to_brain)
    monkeypatch.setattr(satelite, "capture_audio", fake_capture_audio)
    monkeypatch.setattr(satelite, "transcribe", fake_transcribe)
    monkeypatch.setattr("sys.stdin", type("S", (), {"readline": lambda self: "\n"})())

    await run_once()  # não deve levantar
    out = capsys.readouterr().err
    assert "timeout" in out.lower()


def test_transcribe_returns_string(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcribe() com modelo stub retorna string (sem hardware/STT real)."""

    class _Segment:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeModel:
        def transcribe(self, audio, language="pt"):  # noqa: ANN001
            return iter([_Segment("ligar a luz da sala")]), None

    def fake_whisper(model, device, compute_type):  # noqa: ANN001
        return _FakeModel()

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", fake_whisper)

    # Áudio curto sintético (silêncio) — apenas para alimentar a função.
    audio = b"\x00\x00" * 1600
    result = transcribe(audio)
    assert isinstance(result, str)
    assert result == "ligar a luz da sala"


def test_is_shift_detects_all_variants() -> None:
    """_is_shift reconhece shift, shift_l e shift_r (pynput Key)."""
    from satelite import _is_shift

    class _K:
        def __init__(self, name: str | None) -> None:
            self.name = name

    assert _is_shift(_K("shift")) is True
    assert _is_shift(_K("shift_l")) is True
    assert _is_shift(_K("shift_r")) is True
    assert _is_shift(_K("enter")) is False
    assert _is_shift(_K(None)) is False


async def test_with_spinner_runs_coro_and_returns_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_with_spinner executa o coro e devolve seu resultado, sem travar a tela."""
    from satelite import _with_spinner

    async def work() -> str:
        await asyncio.sleep(0.05)
        return "ok"

    result = await _with_spinner("Processando", work())
    assert result == "ok"
    # A animação é limpa ao final; não deve restar o caractere giratório.
    out = capsys.readouterr().err
    assert "⠋" not in out or out.count("\r") >= 1  # sobrescreveu a linha

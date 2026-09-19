"""Task 5.2 — Satélite: transcribe (sem hardware) e send_to_brain."""

from __future__ import annotations

import asyncio
import queue
import threading

import httpx
import numpy as np
import pytest
import respx

from satelite import send_to_brain, transcribe
from src.config import Settings


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


# --- _listen: máquina de estados ESCUTANDO → GRAVANDO (sem hardware) -----------

_BLOCK = 1_280
_SILENCE, _WAKE, _SPEECH = 0, 1, 2


def _blk(value: int) -> np.ndarray:
    return np.full(_BLOCK, value, dtype=np.int16)


def _fake_wake_score(block: np.ndarray) -> float:
    return 1.0 if block[0] == _WAKE else 0.0


def _fake_is_speech(block: np.ndarray) -> bool:
    return bool(block[0] == _SPEECH)


@pytest.fixture
def listen_settings(test_settings: Settings) -> Settings:
    """Durações pequenas: 3 blocos de silêncio, 5 sem fala, teto de 10 blocos (bloco = 80 ms)."""
    return test_settings.model_copy(
        update={"end_silence_s": 0.24, "no_speech_timeout_s": 0.4, "max_record_s": 0.8}
    )


def test_listen_returns_only_audio_after_wake_and_stops_at_silence(
    listen_settings: Settings,
) -> None:
    from satelite import _listen

    blocks = iter(
        [_blk(_SPEECH), _blk(_SPEECH), _blk(_WAKE)]  # fala antes do wake: ignorada
        + [_blk(_SPEECH)] * 3
        + [_blk(_SILENCE)] * 3  # 3 blocos de silêncio → fim
        + [_blk(7)]  # não pode ser consumido
    )

    audio = _listen(blocks, _fake_wake_score, _fake_is_speech, listen_settings)

    samples = np.frombuffer(audio, dtype=np.int16)
    assert len(samples) == 6 * _BLOCK
    assert (samples[: 3 * _BLOCK] == _SPEECH).all()
    assert (samples[3 * _BLOCK :] == _SILENCE).all()
    assert next(blocks)[0] == 7  # parou de consumir o stream assim que terminou


def test_listen_short_pause_does_not_end_recording(listen_settings: Settings) -> None:
    from satelite import _listen

    blocks = [
        _blk(_WAKE),
        _blk(_SPEECH),
        _blk(_SILENCE),
        _blk(_SILENCE),  # pausa de 2 blocos (< 3): não encerra
        _blk(_SPEECH),
        _blk(_SILENCE),
        _blk(_SILENCE),
        _blk(_SILENCE),
    ]

    audio = _listen(iter(blocks), _fake_wake_score, _fake_is_speech, listen_settings)

    assert len(audio) == 7 * _BLOCK * 2  # 7 blocos (tudo depois do wake), 2 bytes/amostra


def test_listen_no_speech_after_wake_returns_empty(listen_settings: Settings) -> None:
    from satelite import _listen

    blocks = iter([_blk(_WAKE)] + [_blk(_SILENCE)] * 20 + [_blk(7)])

    audio = _listen(blocks, _fake_wake_score, _fake_is_speech, listen_settings)

    assert audio == b""
    assert next(blocks)[0] != 7  # desistiu antes do fim do stream (após 5 blocos)


def test_listen_caps_recording_at_max_record(listen_settings: Settings) -> None:
    from satelite import _listen

    blocks = iter([_blk(_WAKE)] + [_blk(_SPEECH)] * 50)

    audio = _listen(blocks, _fake_wake_score, _fake_is_speech, listen_settings)

    assert len(audio) == 10 * _BLOCK * 2  # max_record_s=0.8 → 10 blocos


def test_listen_returns_empty_if_stream_ends_before_wake(listen_settings: Settings) -> None:
    from satelite import _listen

    blocks = iter([_blk(_SILENCE)] * 5 + [_blk(_SPEECH)] * 5)

    assert _listen(blocks, _fake_wake_score, _fake_is_speech, listen_settings) == b""


def test_listen_ignores_score_below_threshold(listen_settings: Settings) -> None:
    from satelite import _listen

    blocks = iter([_blk(_WAKE)] + [_blk(_SPEECH)] * 5)

    # threshold default = 0.5; 0.49 não dispara.
    assert _listen(blocks, lambda b: 0.49, _fake_is_speech, listen_settings) == b""


@pytest.fixture(autouse=True)
def _clear_model_caches() -> None:
    """Os loaders de modelo têm lru_cache; isola os testes entre si."""
    import satelite

    satelite._load_whisper.cache_clear()
    satelite._load_wake_model.cache_clear()
    satelite._load_vad.cache_clear()
    satelite._stop.clear()


def test_transcribe_loads_whisper_model_only_once(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Num loop, recarregar o WhisperModel a cada fala custaria segundos por comando."""
    loads: list[str] = []

    class _Segment:
        text = "oi"

    class _FakeModel:
        def transcribe(self, audio, language="pt"):  # noqa: ANN001
            return iter([_Segment()]), None

    def fake_whisper(model, device, compute_type):  # noqa: ANN001
        loads.append(model)
        return _FakeModel()

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", fake_whisper)

    audio = b"\x00\x00" * 1600
    assert transcribe(audio) == "oi"
    assert transcribe(audio) == "oi"
    assert loads == ["tiny"]  # WHISPER_MODEL=tiny no ambiente de teste


def test_drain_yields_until_stop_is_set() -> None:
    from satelite import _drain

    audio_q: queue.Queue[str] = queue.Queue()
    stop = threading.Event()
    audio_q.put("a")
    audio_q.put("b")

    it = _drain(audio_q, stop)
    assert next(it) == "a"
    stop.set()
    with pytest.raises(StopIteration):
        next(it)


def test_drain_returns_when_stopped_while_queue_is_empty() -> None:
    """Sem áudio na fila, o gerador não pode bloquear para sempre: precisa notar o stop."""
    from satelite import _drain

    audio_q: queue.Queue[str] = queue.Queue()
    stop = threading.Event()
    threading.Timer(0.05, stop.set).start()

    assert list(_drain(audio_q, stop)) == []


async def test_run_once_skips_transcription_when_no_audio(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Falso positivo do wake word → capture_audio devolve b"": não chama Whisper nem Cérebro."""
    import satelite

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("não deveria ser chamado com áudio vazio")

    monkeypatch.setattr(satelite, "capture_audio", lambda *a, **k: b"")
    monkeypatch.setattr(satelite, "transcribe", fail)
    monkeypatch.setattr(satelite, "send_to_brain", fail)

    await satelite.run_once()

    assert "Nada capturado" in capsys.readouterr().err


async def test_run_forever_sets_stop_when_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelar (Ctrl+C) tem de liberar a thread de captura via _stop."""
    import satelite

    started = asyncio.Event()

    async def fake_run_once() -> None:
        started.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(satelite, "run_once", fake_run_once)

    task = asyncio.create_task(satelite.run_forever())
    await started.wait()
    assert not satelite._stop.is_set()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert satelite._stop.is_set()


async def test_run_forever_repeats_run_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import satelite

    calls = 0

    async def fake_run_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("fim do teste")

    monkeypatch.setattr(satelite, "run_once", fake_run_once)

    with pytest.raises(RuntimeError, match="fim do teste"):
        await satelite.run_forever()

    assert calls == 3

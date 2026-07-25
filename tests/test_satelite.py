"""Task 5.2 — Satélite: transcribe (sem hardware) e send_to_brain."""

from __future__ import annotations

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

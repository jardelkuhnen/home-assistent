"""Satélite — captação de voz e STT 100% offline (faster-whisper).

Push-to-talk: aguarda gatilho → grava áudio → transcreve (pt) → POST /chat
ao Cérebro com header ``X-API-Key`` → loga reply e spoken.

``capture_audio()`` fica isolada como ponto de extensão para VAD futuro sem
reescrever o pipeline.
"""

from __future__ import annotations

import asyncio
import sys

import httpx

from src.config import Settings, get_settings

# Duração padrão da captura (segundos) e taxa de amostragem.
_DEFAULT_RECORD_SECONDS = 6
_SAMPLE_RATE = 16_000


def transcribe(audio: bytes, settings: Settings | None = None) -> str:
    """Transcreve áudio PCM com faster-whisper (offline, language=pt)."""
    from faster_whisper import WhisperModel

    s = settings or get_settings()
    model = WhisperModel(s.whisper_model, device="cpu", compute_type="int8")
    import numpy as np

    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = model.transcribe(samples, language="pt")
    return " ".join(segment.text.strip() for segment in segments).strip()


def capture_audio(seconds: int = _DEFAULT_RECORD_SECONDS) -> bytes:
    """Captura áudio do microfone via sounddevice (ponto de extensão p/ VAD).

    Retorna PCM mono 16-bit a 16 kHz. Trocar push-to-talk por VAD significa
    substituir apenas esta função.
    """
    import numpy as np
    import sounddevice as sd

    recording = sd.rec(seconds * _SAMPLE_RATE, samplerate=_SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return np.asarray(recording, dtype=np.int16).tobytes()


async def send_to_brain(
    text: str,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, object]:
    """POST /chat ao Cérebro com header ``X-API-Key``."""
    s = settings or get_settings()
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            base_url=str(s.brain_url),
            headers={"X-API-Key": s.brain_api_key.get_secret_value()},
            timeout=s.brain_timeout_s,
        )
    try:
        response = await client.post("/chat", json={"text": text})
        response.raise_for_status()
        result: dict[str, object] = response.json()
        return result
    finally:
        if owns_client:
            await client.aclose()


async def run_once() -> None:
    """Um ciclo completo: captura → transcreve → envia ao Cérebro."""
    print("Pressione Enter para falar (push-to-talk)...", file=sys.stderr)
    await asyncio.to_thread(sys.stdin.readline)

    audio = await asyncio.to_thread(capture_audio)
    text = await asyncio.to_thread(transcribe, audio)
    print(f"Você disse: {text}", file=sys.stderr)

    if not text:
        print("Nada transcrito.", file=sys.stderr)
        return

    try:
        result = await send_to_brain(text)
    except httpx.TimeoutException:
        print(
            "O Cérebro demorou demais para responder (timeout). Tente de novo.",
            file=sys.stderr,
        )
        return
    except httpx.HTTPError as exc:
        print(f"Não consegui falar com o Cérebro: {exc}", file=sys.stderr)
        return
    print(f"Cérebro: {result.get('reply')} (spoken={result.get('spoken')})", file=sys.stderr)


def main() -> None:
    asyncio.run(run_once())


if __name__ == "__main__":
    main()

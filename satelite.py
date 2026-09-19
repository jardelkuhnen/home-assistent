"""Satélite — captação de voz e STT 100% offline (faster-whisper).

Push-to-talk (segurar Shift): aguarda o usuário segurar Shift → grava áudio
enquanto a tecla fica pressionada → transcreve (pt) → POST /chat ao Cérebro
com header ``X-API-Key`` → loga reply e spoken.

``capture_audio()`` fica isolada como ponto de extensão para VAD futuro sem
reescrever o pipeline.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

import httpx
import numpy as np

from src.config import Settings, get_settings

# Taxa de amostragem do áudio capturado (16 kHz, mono, 16-bit).
_SAMPLE_RATE = 16_000
# Teto de segurança: se o soltar do Shift não for detectado, para de gravar.
_MAX_RECORD_SECONDS = 30
_SHIFT_KEYS = ("shift", "shift_l", "shift_r")
# Blocos de 80 ms (1280 amostras): múltiplo exigido pelo openWakeWord.
_BLOCK_SAMPLES = 1_280
_BLOCK_SECONDS = _BLOCK_SAMPLES / _SAMPLE_RATE

_T = TypeVar("_T")


def transcribe(audio: bytes, settings: Settings | None = None) -> str:
    """Transcreve áudio PCM com faster-whisper (offline, language=pt)."""
    from faster_whisper import WhisperModel

    s = settings or get_settings()
    model = WhisperModel(s.whisper_model, device="cpu", compute_type="int8")
    import numpy as np

    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = model.transcribe(samples, language="pt")
    return " ".join(segment.text.strip() for segment in segments).strip()


def _blocks_for(seconds: float) -> int:
    """Nº de blocos de 80 ms que cobrem ``seconds`` (mínimo 1)."""
    return max(1, round(seconds / _BLOCK_SECONDS))


def _listen(
    blocks: Iterable[np.ndarray],
    wake_score: Callable[[np.ndarray], float],
    is_speech: Callable[[np.ndarray], bool],
    settings: Settings,
) -> bytes:
    """Máquina de estados ESCUTANDO → GRAVANDO, sem I/O (detectores injetados).

    ESCUTANDO: consome blocos até ``wake_score(bloco) >= wake_word_threshold``.
    O bloco que dispara não entra no buffer, então a frase de ativação não vai
    para o STT. GRAVANDO: acumula os blocos seguintes até ``end_silence_s`` de
    silêncio depois de ter ouvido fala, ``max_record_s`` (teto) ou
    ``no_speech_timeout_s`` sem nenhuma fala (falso positivo → ``b""``).
    Para de consumir ``blocks`` assim que termina.
    """
    end_silence = _blocks_for(settings.end_silence_s)
    no_speech = _blocks_for(settings.no_speech_timeout_s)
    max_blocks = _blocks_for(settings.max_record_s)

    recorded: list[np.ndarray] = []
    listening = True
    heard_speech = False
    silent = 0  # blocos de silêncio consecutivos

    for block in blocks:
        if listening:
            listening = wake_score(block) < settings.wake_word_threshold
            continue

        recorded.append(block)
        if is_speech(block):
            heard_speech = True
            silent = 0
        else:
            silent += 1

        if heard_speech and silent >= end_silence:
            break
        if not heard_speech and silent >= no_speech:
            return b""
        if len(recorded) >= max_blocks:
            break

    if not heard_speech:
        return b""
    return np.concatenate(recorded).tobytes()


def capture_audio(max_seconds: int = _MAX_RECORD_SECONDS) -> bytes:
    """Captura áudio via sounddevice enquanto o Shift é mantido pressionado.

    Push-to-talk: a gravação começa no ``key press`` do Shift e para no
    ``key release`` (ou ao atingir ``max_seconds``, por segurança). Retorna
    PCM mono 16-bit a 16 kHz.

    Usa ``pynput`` para detectar o Shift (tecla modificadora, sem caractere).
    No macOS, exige permissão de Accessibility para o terminal que roda o
    Satélite (System Settings → Privacy & Security → Accessibility).

    Ponto de extensão: trocar push-to-talk por VAD significa substituir só
    esta função — o restante do pipeline não muda.
    """
    import numpy as np
    import sounddevice as sd
    from pynput import keyboard

    chunks: list[np.ndarray] = []
    started = threading.Event()
    stopped = threading.Event()

    def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:  # noqa: ARG001
        chunks.append(indata.copy())

    def on_press(key: object) -> None:
        if _is_shift(key) and not started.is_set():
            started.set()

    def on_release(key: object) -> None:
        if _is_shift(key):
            stopped.set()

    stream = sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16", callback=callback)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Aguarda o primeiro press do Shift para iniciar a captura.
    started.wait()
    stream.start()
    try:
        # Para no release do Shift ou no teto de segurança.
        stopped.wait(timeout=max_seconds)
    finally:
        stream.stop()
        stream.close()
        listener.stop()

    if not chunks:
        return b""
    return np.concatenate(chunks, axis=0).tobytes()


def _is_shift(key: object) -> bool:
    """True se ``key`` (pynput) é qualquer variante do Shift."""
    name = getattr(key, "name", None) or getattr(key, "chars", None)
    if isinstance(name, str):
        return name in _SHIFT_KEYS
    return False


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


async def _with_spinner(message: str, coro: Awaitable[_T]) -> _T:
    """Roda ``coro`` mostrando uma animação giratória em ``stderr``.

    Evita a impressão de tela travada durante gravação/transcrição/envio.
    Sobrescreve a mesma linha com ``\\r`` e a limpa ao terminar.
    """
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    done = asyncio.Event()

    async def spin() -> None:
        i = 0
        while not done.is_set():
            sys.stderr.write(f"\r{message} {frames[i % len(frames)]}")
            sys.stderr.flush()
            i += 1
            await asyncio.sleep(0.08)
        # Limpa a linha da animação.
        sys.stderr.write("\r" + " " * (len(message) + 2) + "\r")
        sys.stderr.flush()

    task = asyncio.create_task(spin())
    try:
        return await coro
    finally:
        done.set()
        await task


async def run_once() -> None:
    """Um ciclo completo: captura → transcreve → envia ao Cérebro."""
    print("Mantenha o Shift pressionado para falar (solte para parar)...", file=sys.stderr)

    audio = await _with_spinner("Gravando áudio", asyncio.to_thread(capture_audio))
    text = await _with_spinner("Transcrevendo", asyncio.to_thread(transcribe, audio))
    print(f"Você disse: {text}", file=sys.stderr)

    if not text:
        print("Nada transcrito.", file=sys.stderr)
        return

    try:
        result = await _with_spinner("Consultando o Cérebro", send_to_brain(text))
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
    # Diagnóstico do HA (metadata.tool_diagnostics): só stderr, nunca fala.
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        for diagnostic in metadata.get("tool_diagnostics", []):
            print(
                f"  [tool] {diagnostic['tool']} {diagnostic['action']} {diagnostic['entity_id']}"
                f" -> {diagnostic['status_code']} {diagnostic['body'][:120]}",
                file=sys.stderr,
            )


def main() -> None:
    asyncio.run(run_once())


if __name__ == "__main__":
    main()

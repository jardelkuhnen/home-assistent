"""Satélite — wake word, captação de voz e STT 100% offline.

Fica escutando o microfone: ao ouvir "hey jarvis" (openWakeWord) grava o
comando até detectar silêncio (VAD) → transcreve com faster-whisper (pt) →
POST /chat ao Cérebro com header ``X-API-Key`` → loga reply e spoken → volta a
escutar.

``capture_audio()`` é a única costura com o hardware; a lógica de estados vive
em ``_listen()`` (pura, testável sem microfone).
"""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
from collections.abc import Awaitable, Callable, Iterable, Iterator
from functools import lru_cache
from typing import Any, TypeVar

import httpx
import numpy as np

from src.config import Settings, get_settings

# Taxa de amostragem do áudio capturado (16 kHz, mono, 16-bit).
_SAMPLE_RATE = 16_000
# Blocos de 80 ms (1280 amostras): múltiplo exigido pelo openWakeWord.
_BLOCK_SAMPLES = 1_280
_BLOCK_SECONDS = _BLOCK_SAMPLES / _SAMPLE_RATE
# VAD: score ≥ limiar é fala. frame_size precisa dividir o bloco de 1280.
_VAD_THRESHOLD = 0.5
_VAD_FRAME_SAMPLES = 640
# Setado ao encerrar (Ctrl+C) para liberar a thread de captura, que de outro
# modo ficaria bloqueada esperando o wake word e impediria o processo de sair.
_stop = threading.Event()

_T = TypeVar("_T")


@lru_cache
def _load_whisper(name: str) -> Any:
    """Carrega o WhisperModel uma vez por processo (o loop de escuta o reusa)."""
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8")


@lru_cache
def _load_wake_model(name: str) -> Any:
    """Carrega o modelo de wake word (ONNX). Os arquivos vêm de ``download_models``."""
    from openwakeword.model import Model

    return Model(wakeword_models=[name], inference_framework="onnx")


@lru_cache
def _load_vad() -> Any:
    """Carrega o VAD (Silero, ONNX) do openWakeWord."""
    from openwakeword.vad import VAD

    return VAD()


def transcribe(audio: bytes, settings: Settings | None = None) -> str:
    """Transcreve áudio PCM com faster-whisper (offline, language=pt)."""
    s = settings or get_settings()
    model = _load_whisper(s.whisper_model)

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


def _drain(audio_q: queue.Queue[np.ndarray], stop: threading.Event) -> Iterator[np.ndarray]:
    """Gera blocos da fila até ``stop`` ser setado.

    Consulta ``stop`` a cada 0,2 s: sem isso, ``get()`` bloquearia para sempre
    com o microfone mudo e o processo não conseguiria encerrar.
    """
    while not stop.is_set():
        try:
            yield audio_q.get(timeout=0.2)
        except queue.Empty:
            continue


def capture_audio(settings: Settings | None = None) -> bytes:
    """Espera o wake word e grava uma fala até o silêncio. Devolve PCM int16 16 kHz.

    Abre um único ``InputStream`` (não perde o começo do comando entre o wake
    word e a gravação) e o fecha ao retornar — durante a transcrição e a
    chamada ao Cérebro o microfone fica mudo. Devolve ``b""`` num falso positivo
    (wake word sem fala em seguida).

    Requer os modelos do openWakeWord já baixados (``download_models``); se
    faltarem, ``Model``/``VAD`` levantam na abertura (fail fast).
    """
    import sounddevice as sd

    s = settings or get_settings()
    wake_model = _load_wake_model(s.wake_word_model)
    vad = _load_vad()
    # Os modelos vivem o processo inteiro e guardam estado interno: zera para
    # que o áudio da fala anterior não contamine esta.
    wake_model.reset()
    vad.reset_states()

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:  # noqa: ARG001
        audio_q.put(indata[:, 0].copy())

    def wake_score(block: np.ndarray) -> float:
        return float(wake_model.predict(block)[s.wake_word_model])

    def is_speech(block: np.ndarray) -> bool:
        return bool(vad.predict(block, frame_size=_VAD_FRAME_SAMPLES) >= _VAD_THRESHOLD)

    with sd.InputStream(
        samplerate=_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=_BLOCK_SAMPLES,
        callback=callback,
    ):
        return _listen(_drain(audio_q, _stop), wake_score, is_speech, s)


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
    """Um ciclo completo: espera o wake word → grava → transcreve → envia ao Cérebro."""
    audio = await _with_spinner('Escutando (diga "hey jarvis")', asyncio.to_thread(capture_audio))
    if not audio:
        # Falso positivo do wake word (sem fala depois): não há o que transcrever.
        print("Nada capturado.", file=sys.stderr)
        return

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


async def run_forever() -> None:
    """Escuta continuamente. Ao ser cancelado (Ctrl+C), libera a thread de captura."""
    try:
        while True:
            await run_once()
    finally:
        _stop.set()


def main() -> None:
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        print("Encerrando.", file=sys.stderr)


if __name__ == "__main__":
    main()

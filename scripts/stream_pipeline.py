"""
Tara streaming pipeline — real-time mic input.

Architecture:
  sounddevice mic (512-sample chunks / 32ms)
    → rolling 2s pre-buffer
    → Silero VADIterator (per-chunk, streaming)
    → on utterance end: DeepFilterNet NS (batch on utterance buffer)
    → DeepgramWakeWord / Porcupine (utterance start probe)
    → faster-whisper STT (command audio after wake clip)

State machine: IDLE → COLLECTING → PROCESSING → IDLE

Usage:
  python scripts/stream_pipeline.py [--wake-word-backend deepgram|porcupine|openwakeword]
  Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tara_pipeline.config import (
    SAMPLE_RATE,
    DEFAULT_WAKE_WORD_BACKEND,
    VAD_THRESHOLD,
    VAD_SPEECH_PAD_MS,
    DEEPGRAM_WAKE_PROBE_S,
    DEEPGRAM_WAKE_CLIP_S,
    WAKE_WORD_BUFFER_S,
)
from tara_pipeline.stages.noise_suppression import create_suppressor
from tara_pipeline.stages.wake_word import create_wake_word_detector_with_fallback
from tara_pipeline.stages.stt import create_stt

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512                          # samples per chunk (32ms @ 16kHz)
CHUNK_S = CHUNK_SIZE / SAMPLE_RATE        # 0.032s
PRE_BUFFER_CHUNKS = int(2.0 / CHUNK_S)   # 2s back-pad (captures "Tara" before VAD)
MAX_UTTERANCE_S = 30.0                    # safety cap — avoid infinite accumulation
MAX_UTTERANCE_CHUNKS = int(MAX_UTTERANCE_S / CHUNK_S)
MIN_SILENCE_MS = 600                      # VADIterator silence gap to end utterance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tara streaming voice pipeline")
    p.add_argument(
        "--wake-word-backend",
        default=DEFAULT_WAKE_WORD_BACKEND,
        choices=["deepgram", "porcupine", "openwakeword", "whisper_phoneme"],
        help=f"Wake word backend (default: {DEFAULT_WAKE_WORD_BACKEND})",
    )
    p.add_argument(
        "--stt-backend",
        default="faster_whisper",
        choices=["faster_whisper", "deepgram"],
        help="STT backend (default: faster_whisper)",
    )
    p.add_argument(
        "--deepgram-stt-model",
        default="nova-3",
        help="Deepgram STT model name (default: nova-3, only used with --stt-backend deepgram)",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def load_models(wake_word_backend: str, stt_backend: str = "faster_whisper", deepgram_stt_model: str = "nova-3"):
    """Load all models. Returns (suppressor, vad_model, vad_iterator_cls, wake_word, stt)."""
    logger.info("Loading models…")

    suppressor = create_suppressor("deepfilternet")

    # Silero VAD
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=True,
    )
    VADIterator = vad_utils[3]

    if stt_backend == "deepgram":
        stt = create_stt("deepgram", model=deepgram_stt_model)
        logger.info(f"STT: deepgram {deepgram_stt_model}")
    else:
        stt = create_stt("faster_whisper", model_name="tiny.en")
        logger.info("STT: faster-whisper tiny.en")

    ww_kwargs: dict = {}
    if wake_word_backend == "whisper_phoneme" and stt_backend == "faster_whisper" and hasattr(stt, "_model"):
        ww_kwargs["model"] = stt._model
    wake_word = create_wake_word_detector_with_fallback(
        primary=wake_word_backend,
        fallback="whisper_phoneme",
        **ww_kwargs,
    )

    logger.info("All models loaded — listening…")
    return suppressor, vad_model, VADIterator, wake_word, stt


def process_utterance(
    utterance: np.ndarray,
    suppressor,
    wake_word,
    stt,
    wake_word_backend: str,
) -> None:
    """Run NS → wake word → STT on a completed utterance buffer."""
    t_start = time.perf_counter()

    duration_s = len(utterance) / SAMPLE_RATE
    logger.debug(f"Utterance: {duration_s:.2f}s — processing")

    # Stage 1: Noise suppression (batch on utterance)
    try:
        audio_clean, ns_ms = suppressor.suppress(utterance, SAMPLE_RATE)
    except Exception as e:
        logger.error(f"NS failed: {e}")
        return

    # Stage 2: Wake word — probe start of utterance
    try:
        ww_result = wake_word.detect_at_utterance_start(audio_clean, SAMPLE_RATE)
    except Exception as e:
        logger.error(f"Wake word failed: {e}")
        return

    ww_ms = (time.perf_counter() - t_start) * 1000 - ns_ms

    if not ww_result.triggered:
        logger.debug(
            f"Wake word not detected (score={ww_result.score:.3f}) — discarding"
        )
        return

    logger.info(
        f"Wake word TRIGGERED | score={ww_result.score:.3f} | "
        f"backend={ww_result.backend} | ww={ww_ms:.0f}ms"
    )

    # Stage 3: Clip wake word, STT command only
    if wake_word_backend == "whisper_phoneme":
        from tara_pipeline.config import WAKE_WORD_PROBE_S
        clip_samples = int(SAMPLE_RATE * WAKE_WORD_PROBE_S)
    elif wake_word_backend == "deepgram":
        clip_samples = int(SAMPLE_RATE * DEEPGRAM_WAKE_CLIP_S)
    else:
        clip_samples = int(SAMPLE_RATE * WAKE_WORD_BUFFER_S)

    min_command = int(SAMPLE_RATE * 0.3)
    if len(audio_clean) > clip_samples + min_command:
        command_audio = audio_clean[clip_samples:]
    else:
        command_audio = audio_clean

    try:
        transcript = stt.transcribe(command_audio, SAMPLE_RATE)
    except Exception as e:
        logger.error(f"STT failed: {e}")
        return

    total_ms = (time.perf_counter() - t_start) * 1000

    if transcript.text.strip():
        print(
            f"\n{'─'*55}\n"
            f"  TARA heard: \"{transcript.text.strip()}\"\n"
            f"  ns={ns_ms:.0f}ms | ww={ww_ms:.0f}ms | "
            f"stt={transcript.elapsed_ms:.0f}ms | total={total_ms:.0f}ms\n"
            f"{'─'*55}\n"
        )
    else:
        logger.info("Wake word triggered but no command audio — skipping")


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, format="{time:HH:mm:ss.SSS} | {level:<8} | {message}")

    try:
        import sounddevice as sd
    except ImportError:
        logger.error("sounddevice not installed: pip install sounddevice")
        sys.exit(1)

    suppressor, vad_model, VADIterator, wake_word, stt = load_models(
        args.wake_word_backend, args.stt_backend, args.deepgram_stt_model
    )

    # Processing runs in a background thread — mic callback stays non-blocking
    process_queue: queue.Queue[np.ndarray] = queue.Queue()

    def processing_worker() -> None:
        while True:
            utterance = process_queue.get()
            if utterance is None:
                break
            process_utterance(utterance, suppressor, wake_word, stt, args.wake_word_backend)

    worker = threading.Thread(target=processing_worker, daemon=True)
    worker.start()

    # Streaming state
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def mic_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning(f"Mic status: {status}")
        audio_queue.put(indata[:, 0].copy())

    vad_iter = VADIterator(
        vad_model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )

    pre_buffer: deque[np.ndarray] = deque(maxlen=PRE_BUFFER_CHUNKS)
    speech_buffer: list[np.ndarray] = []
    in_speech = False

    print("\nTara streaming pipeline active. Say 'Hey Tara' or 'Tara' to wake.")
    print("Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=mic_callback,
        ):
            while True:
                try:
                    chunk = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                chunk_tensor = torch.from_numpy(chunk)

                try:
                    result = vad_iter(chunk_tensor, return_seconds=False)
                except Exception:
                    result = None

                if not in_speech:
                    pre_buffer.append(chunk)

                if result is not None:
                    if "start" in result:
                        in_speech = True
                        speech_buffer = list(pre_buffer) + [chunk]
                        logger.debug("VAD: speech start")

                    elif "end" in result:
                        if in_speech:
                            speech_buffer.append(chunk)
                            in_speech = False
                            logger.debug(
                                f"VAD: speech end — utterance "
                                f"{len(speech_buffer) * CHUNK_S:.2f}s"
                            )
                            utterance = np.concatenate(speech_buffer).astype(np.float32)
                            speech_buffer = []
                            process_queue.put(utterance)

                elif in_speech:
                    speech_buffer.append(chunk)
                    # Safety cap
                    if len(speech_buffer) >= MAX_UTTERANCE_CHUNKS:
                        logger.warning("Utterance too long — forcing flush")
                        in_speech = False
                        utterance = np.concatenate(speech_buffer).astype(np.float32)
                        speech_buffer = []
                        vad_iter.reset_states()
                        process_queue.put(utterance)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        process_queue.put(None)
        worker.join(timeout=5)


if __name__ == "__main__":
    main()

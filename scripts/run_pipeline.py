"""
CLI entrypoint: python scripts/run_pipeline.py assets/tara_assignment_recording_clipped.flac

Usage:
    python scripts/run_pipeline.py <audio_file> [--iteration N] [--wake-word-backend BACKEND]

Options:
    --iteration         1|2|3|4  (default: 4)
    --wake-word-backend openwakeword|porcupine|whisper_phoneme|none  (default: whisper_phoneme)
    --log-level         DEBUG|INFO|WARNING  (default: INFO)

Wake word backends:
    whisper_phoneme  — transcribe first 0.5s, check starts with "tara"/"hey tara" (recommended)
    openwakeword     — sklearn classifier on OWW prediction scores (high false positive rate)
    porcupine        — Picovoice (requires PORCUPINE_ACCESS_KEY env var)
    none             — passthrough, no wake word filtering
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from tara_pipeline.pipeline import TaraPipeline
from tara_pipeline.utils.metrics import get_profiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tara voice command pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to audio file (.flac, .wav, .mp3)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Pipeline iteration (1=raw, 2=noisereduce, 3=dfn+vad, 4=full)",
    )
    parser.add_argument(
        "--wake-word-backend",
        choices=["openwakeword", "porcupine", "whisper_phoneme", "none"],
        default="whisper_phoneme",
        help="Wake word backend (only used in iteration 4)",
    )
    parser.add_argument(
        "--stt-backend",
        choices=["faster_whisper", "whisper", "deepgram"],
        default="faster_whisper",
        help="STT backend (deepgram requires DEEPGRAM_API_KEY env var)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Configure loguru
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
    )

    # Also log to file for methodology doc evidence
    log_file = Path("logs") / f"iteration_{args.iteration}.log"
    log_file.parent.mkdir(exist_ok=True)
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation="10 MB",
    )

    if not args.audio.exists():
        logger.error(f"Audio file not found: {args.audio}")
        return 1

    logger.info(f"Starting Iteration {args.iteration} | audio={args.audio}")

    pipeline = TaraPipeline(
        iteration=args.iteration,
        wake_word_backend=args.wake_word_backend,
        stt_backend=args.stt_backend,
    )
    result = pipeline.run(args.audio)

    # Print latency report
    profiler = get_profiler()
    print(profiler.report())

    # Print commands
    print(result.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())

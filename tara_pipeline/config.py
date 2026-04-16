"""
Pipeline configuration — latency budgets, model paths, thresholds.
All latency values in milliseconds.
"""

from dataclasses import dataclass
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR.parent / "assets"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
DOCS_DIR = ROOT_DIR / "docs"

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000          # Hz — Whisper, Silero, openWakeWord all expect 16kHz
CHUNK_DURATION_S: float = 0.032    # 32ms VAD frames (Silero default)
VAD_WINDOW_S: float = 0.5          # min speech segment to pass downstream
WAKE_WORD_BUFFER_S: float = 1.0    # buffer at utterance START for wake word check
WAKE_WORD_PROBE_S: float = 0.5    # seconds to transcribe for phoneme-based wake word check

# ── Latency budgets (ms) ─────────────────────────────────────────────────────
BUDGET_NOISE_SUPPRESSION_MS: int = 200
BUDGET_VAD_MS: int = 100
BUDGET_WAKE_WORD_MS: int = 300
BUDGET_STT_MS: int = 1_000
BUDGET_TOTAL_MS: int = 2_000

# ── Noise suppression ────────────────────────────────────────────────────────
# DeepFilterNet — final pipeline (Iteration 3+)
DEEPFILTERNET_ATTN_LIMIT: float = 100.0
DEEPFILTERNET_POST_FILTER: bool = True

# noisereduce — baseline (Iteration 2 only)
NOISEREDUCE_PROP_DECREASE: float = 0.75
NOISEREDUCE_STATIONARY: bool = False

# ── VAD (Silero) ─────────────────────────────────────────────────────────────
VAD_THRESHOLD: float = 0.5         # speech probability threshold
VAD_MIN_SPEECH_MS: int = 250       # discard segments shorter than this
VAD_MIN_SILENCE_MS: int = 100      # gap to split segments
VAD_SPEECH_PAD_MS: int = 30        # pad each side of detected speech

# ── Wake word ────────────────────────────────────────────────────────────────
# openWakeWord
OWW_THRESHOLD: float = 0.70
OWW_INFERENCE_FRAMEWORK: str = "onnx"  # or "tflite"
OWW_WAKEWORDS: list[str] = ["hey_tara", "tara"]

# Picovoice Porcupine (fallback)
PORCUPINE_ACCESS_KEY: str = ""     # set via env var PORCUPINE_ACCESS_KEY
PORCUPINE_SENSITIVITY: float = 0.5
PORCUPINE_KEYWORDS: list[str] = ["hey tara", "tara"]

# ── STT ──────────────────────────────────────────────────────────────────────
# Iteration 1 & 2 — baseline
WHISPER_BASE_MODEL: str = "base"
WHISPER_BASE_LANGUAGE: str = "en"

# Iteration 3 & 4 — final
FASTER_WHISPER_MODEL: str = "tiny.en"
FASTER_WHISPER_DEVICE: str = "cpu"
FASTER_WHISPER_COMPUTE_TYPE: str = "int8"  # quantised — fast on CPU
FASTER_WHISPER_BEAM_SIZE: int = 1          # greedy, lowest latency
FASTER_WHISPER_VAD_FILTER: bool = False    # we run our own VAD upstream

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_STAGE_TIMINGS: bool = True


@dataclass(frozen=True)
class LatencyBudget:
    """Immutable latency budget spec for validation."""
    noise_suppression_ms: int = BUDGET_NOISE_SUPPRESSION_MS
    vad_ms: int = BUDGET_VAD_MS
    wake_word_ms: int = BUDGET_WAKE_WORD_MS
    stt_ms: int = BUDGET_STT_MS
    total_ms: int = BUDGET_TOTAL_MS


LATENCY_BUDGET = LatencyBudget()

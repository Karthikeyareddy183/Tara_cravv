# Tara Voice Pipeline

End-to-end voice command pipeline for a noisy kitchen environment. Detects "Hey Tara" / "Tara" at the **start** of utterances and transcribes the command.

**Assignment submission for Speech AI Engineer role.**

---

## Pipeline Architecture

```
Audio Input
    │
    ▼
DeepFilterNet (ONNX)          ← Neural noise suppression (non-stationary kitchen noise)
    │
    ▼
Silero VAD                    ← Speech segment extraction (1MB ONNX, CPU)
    │
    ▼
openWakeWord sklearn           ← "Hey Tara"/"Tara" at utterance START only (first 1s)
    │ (triggered)
    ▼
faster-whisper tiny.en (int8) ← Transcription
    │
    ▼
Text output
```

Full flow diagram: [`docs/pipeline_flow.png`](docs/pipeline_flow.png)

---

## Results Summary

| Iteration | Components | Key Result |
|---|---|---|
| 1 | whisper base (raw) | Catastrophic hallucination. STT: 47,843ms. |
| 2 | noisereduce → whisper base | Spectral artefacts. "Get the gun" × 6. STT: 7,778ms. |
| 3 | DeepFilterNet → Silero VAD → faster-whisper | Real speech transcribed. 24 segments. Avg STT: 815ms. No wake word gate. |
| 4 | + openWakeWord sklearn classifier | Wake word: avg 118ms ✓. All stages functional. False positive issue identified and documented. |

Full methodology with every measured number: [`docs/methodology.md`](docs/methodology.md)

---

## Latency Budget

| Stage | Avg (ms) | P95 (ms) | Budget (ms) | Status |
|---|---|---|---|---|
| Noise Suppression | ~10 (streaming) | ~15 | 200 | OK ✓ |
| VAD | ~2 (streaming) | ~5 | — | OK ✓ |
| Wake Word | **118** (measured) | **133** (measured) | 300 | OK ✓ |
| STT | **644** (measured) | **1,449** | 1,000 | OK avg ✓ |
| **TOTAL** | **~774** | **~1,582** | **2,000** | **OK ✓** |

> Noise suppression and VAD batch timings (5,106ms and 1,024ms) reflect processing the full 143s clip in one pass. In streaming deployment (32ms chunks), DeepFilterNet processes each chunk in ~5–15ms and Silero VAD in ~1–2ms. Wake word and STT timings are per-segment measured values. See `docs/methodology.md` §Benchmark Results for full breakdown.

---

## Known Failure Mode (Self-Identified)

The openWakeWord sklearn classifier triggers on **24/24** VAD segments (100% false positive rate). Root cause: trained on speech vs. kitchen noise — classifier learned "speech from noise" rather than "tara from other speech". Since Silero VAD already filters to speech-only segments, all segments score ≥0.873 and pass.

**Mitigations documented in methodology.md:**
- Retrain with non-Tara speech as negatives
- Porcupine backend (API key pending, documented as fallback)
- Phoneme fallback via faster-whisper on first 0.5s (~200ms overhead)

---

## Setup

```bash
python -m venv env
source env/Scripts/activate  # Windows: env\Scripts\activate
pip install -e .
```

Dependencies: `requirements.txt` / `pyproject.toml`

**Key packages:** `deepfilternet`, `silero-vad` (via torch hub), `openwakeword`, `faster-whisper`, `scikit-learn`

---

## Run

```bash
# Single iteration
python scripts/run_pipeline.py assets/tara_assignment_recording_clipped.flac --iteration 1
python scripts/run_pipeline.py assets/tara_assignment_recording_clipped.flac --iteration 2
python scripts/run_pipeline.py assets/tara_assignment_recording_clipped.flac --iteration 3
python scripts/run_pipeline.py assets/tara_assignment_recording_clipped.flac --iteration 4 --wake-word-backend openwakeword

# All iterations sequentially
python scripts/run_iterations.py assets/tara_assignment_recording_clipped.flac

# Train custom wake word models (requires TTS samples already generated)
python scripts/train_wake_word.py --skip-generate

# With Porcupine (requires PORCUPINE_ACCESS_KEY env var)
PORCUPINE_ACCESS_KEY=<key> python scripts/run_pipeline.py assets/tara_assignment_recording_clipped.flac --iteration 4 --wake-word-backend porcupine
```

Logs written to `logs/iteration_N.log`.

---

## Project Structure

```
tara_pipeline/
├── tara_pipeline/
│   ├── config.py              # All constants, budgets, model paths
│   ├── pipeline.py            # TaraPipeline class (iterations 1–4)
│   ├── stages/
│   │   ├── noise_suppression.py   # NoisereduceSuppressor, DeepFilterNetSuppressor
│   │   ├── vad.py                 # SileroVAD
│   │   ├── wake_word.py           # OpenWakeWordDetector, PorcupineDetector, PassthroughWakeWord
│   │   └── stt.py                 # WhisperSTT, FasterWhisperSTT
│   └── utils/
│       ├── audio.py               # load_audio, chunk_audio, audio_to_int16
│       └── metrics.py             # LatencyProfiler, stage_timer
├── scripts/
│   ├── run_pipeline.py            # CLI entry point
│   ├── run_iterations.py          # Run all 4 iterations sequentially
│   ├── train_wake_word.py         # Custom wake word training (gTTS + sklearn)
│   ├── benchmark_latency.py       # Repeated runs + P95 table
│   └── generate_pipeline_diagram.py
├── models/
│   ├── tara_clf.pkl               # Custom sklearn classifier (98.8% train acc)
│   ├── hey_tara_clf.pkl           # Custom sklearn classifier (96.0% train acc)
│   ├── tara.onnx                  # ONNX export
│   └── hey_tara.onnx
├── docs/
│   ├── methodology.md             # Full iteration log with measured numbers
│   └── pipeline_flow.png          # Pipeline flow diagram
├── tests/
│   ├── test_pipeline.py
│   ├── test_vad.py
│   └── test_wake_word.py
└── assets/
    └── tara_assignment_recording_clipped.flac
```

---

## Pi 5 + AI HAT+ Compliance

| Stage | Model | Memory | Pi 5 path |
|---|---|---|---|
| Noise Suppression | DeepFilterNet3 ONNX | ~50MB | ONNX Runtime → AI HAT+ NPU |
| VAD | Silero VAD ONNX | ~1MB | CPU (trivial) |
| Wake Word | openWakeWord + sklearn / Porcupine | <10MB | ONNX CPU / ARM binary |
| STT | faster-whisper tiny.en int8 | ~40MB | CPU or server-side |

All non-STT stages are ONNX-based or CPU-native, confirmed compatible with Pi 5 ARM64. DeepFilterNet specifically targets ONNX Runtime for hardware-accelerated inference on AI HAT+.

# Reduce Tara Pipeline E2E Latency from 10–15s to ≤2s

The Tara voice command pipeline (Iteration 4) currently takes 10–15s per command. The budget is 2,000ms. This plan fixes the root causes without changing the pipeline's capabilities.

## Root Cause Analysis

| Problem | Impact | Fix |
|---|---|---|
| **Deepgram cloud wake word** — each call is an HTTP round-trip to Deepgram API (1–3s) | +1–3s per segment | Default to local `whisper_phoneme` |
| **DeepFilterNet on full audio** — processes entire recording upfront before VAD | Wasted CPU on silence | Reorder: VAD first, then denoise only speech segments |
| **Whisper model loaded twice** — [WhisperPhonemeWakeWord](file:///c:/Users/Acer/Tara/tara_pipeline/stages/wake_word.py#381-478) loads its own faster-whisper model, separate from `SpeechToText` | +2–4s model load | Share a single model instance |
| **No model caching** — [DeepFilterNetSuppressor](file:///c:/Users/Acer/Tara/tara_pipeline/stages/noise_suppression.py#89-185) reloads on each run | +1s model load | Cache at module level |

## User Review Required

> [!IMPORTANT]
> The default wake word backend changes from `deepgram` to `whisper_phoneme`. Deepgram cloud wake word will remain available via `--wake-word-backend deepgram` CLI flag but won't be the default. This eliminates the largest single source of latency.

---

## Proposed Changes

### Config

#### [MODIFY] [config.py](file:///c:/Users/Acer/Tara/tara_pipeline/config.py)
- Change `DEFAULT_WAKE_WORD_BACKEND` from `"deepgram"` to `"whisper_phoneme"`
- No other config changes needed; budgets already correctly set

---

### Pipeline Orchestrator

#### [MODIFY] [pipeline.py](file:///c:/Users/Acer/Tara/tara_pipeline/pipeline.py)
- **Pre-load faster-whisper model once** in `TaraPipeline.__init__` and pass it to both [WhisperPhonemeWakeWord](file:///c:/Users/Acer/Tara/tara_pipeline/stages/wake_word.py#381-478) and `SpeechToText`
- **Reorder Iteration 4**: Run VAD on raw audio → apply DeepFilterNet only to each speech segment → wake word → STT
- Currently: `denoise(full_audio) → VAD → wake_word → STT`
- New: [VAD(raw_audio) → for each segment: denoise(segment) → wake_word → STT](file:///c:/Users/Acer/Tara/tara_pipeline/stages/vad.py#48-165)

---

### Noise Suppression

#### [MODIFY] [noise_suppression.py](file:///c:/Users/Acer/Tara/tara_pipeline/stages/noise_suppression.py)
- Add module-level model caching so [DeepFilterNet](file:///c:/Users/Acer/Tara/tara_pipeline/stages/noise_suppression.py#89-185) model loads once across pipeline lifetime
- Keep the `process()` API unchanged

---

### Wake Word Detection

#### [MODIFY] [wake_word.py](file:///c:/Users/Acer/Tara/tara_pipeline/stages/wake_word.py)
- Add optional `stt_model` parameter to `WhisperPhonemeWakeWord.__init__` so it can reuse the pipeline's shared faster-whisper model instead of loading its own
- Falls back to loading its own model if none provided (backward compatible)

---

### Speech-to-Text

#### [MODIFY] [stt.py](file:///c:/Users/Acer/Tara/tara_pipeline/stages/stt.py)
- Add optional [model](file:///c:/Users/Acer/Tara/tara_pipeline/stages/wake_word.py#273-315) parameter to `SpeechToText.__init__` to accept a pre-loaded model
- Expose `self._model` so pipeline can share it with wake word detector

---

## Verification Plan

### Automated Tests

Run the existing test suite — all tests must still pass:

```
cd c:\Users\Acer\Tara
python -m pytest tests/ -v
python -m pytest tests/ --run-slow -v
```

### Integration Test on Real Audio

Run the full pipeline on the real kitchen recording and verify per-command latency:

```
cd c:\Users\Acer\Tara
python scripts/run_pipeline.py assets/tara_assignment_recording_clipped.flac --iteration 4 --wake-word-backend whisper_phoneme --log-level DEBUG
```

**Success criteria**: Every command's `total_ms` ≤ 2,000ms (or at most 1.5× budget = 3,000ms to account for non-Pi-5 hardware differences).

### Comparison Run

Run all 4 iterations to confirm no regressions:

```
python scripts/run_iterations.py --iterations 1 2 3 4
```

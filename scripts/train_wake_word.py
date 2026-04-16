"""
Custom wake word training for "Tara" and "Hey Tara" using openWakeWord.

Steps:
  1. Generate synthetic TTS audio (gTTS - free, no API key)
  2. Train openWakeWord verifier model
  3. Export to ONNX → models/tara.onnx, models/hey_tara.onnx

Usage:
    python scripts/train_wake_word.py
    python scripts/train_wake_word.py --phrase "hey tara" --samples 500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path("models")
TRAINING_DIR = Path("training_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom wake word model")
    parser.add_argument("--phrase", default="hey tara", help="Wake word phrase")
    parser.add_argument("--samples", type=int, default=500, help="Number of TTS samples")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--skip-generate", action="store_true", help="Skip TTS generation")
    return parser.parse_args()


def generate_tts_samples(phrase: str, n_samples: int, out_dir: Path) -> list[Path]:
    """Generate synthetic TTS audio samples using gTTS with variation."""
    try:
        from gtts import gTTS
    except ImportError:
        print("ERROR: Install gTTS: pip install gtts")
        sys.exit(1)

    try:
        from pydub import AudioSegment
    except ImportError:
        print("ERROR: Install pydub: pip install pydub")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Language/accent variations for robustness
    tld_variants = ["com", "co.in", "com.au", "co.uk", "ca"]
    # Speed variations via pydub
    speed_factors = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]

    print(f"Generating {n_samples} TTS samples for '{phrase}'...")
    t0 = time.perf_counter()

    for i in range(n_samples):
        tld = tld_variants[i % len(tld_variants)]
        speed = random.choice(speed_factors)

        out_path = out_dir / f"sample_{i:04d}.wav"
        if out_path.exists():
            generated.append(out_path)
            continue

        try:
            # Generate TTS
            mp3_path = out_dir / f"sample_{i:04d}.mp3"
            tts = gTTS(text=phrase, lang="en", tld=tld)
            tts.save(str(mp3_path))

            # Convert to 16kHz mono WAV + speed variation
            audio = AudioSegment.from_mp3(str(mp3_path))
            audio = audio.set_frame_rate(16000).set_channels(1)

            # Apply speed variation
            if speed != 1.0:
                audio = audio._spawn(
                    audio.raw_data,
                    overrides={"frame_rate": int(audio.frame_rate * speed)}
                ).set_frame_rate(16000)

            audio.export(str(out_path), format="wav")
            mp3_path.unlink(missing_ok=True)
            generated.append(out_path)

            if (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i+1}/{n_samples} samples | {elapsed:.1f}s")

        except Exception as e:
            print(f"  Sample {i} failed: {e} — skipping")

    print(f"Generated {len(generated)} samples in {time.perf_counter()-t0:.1f}s")
    return generated


def download_openwakeword_models() -> bool:
    """Download pretrained openWakeWord base models if missing."""
    try:
        import openwakeword
        from openwakeword import utils as oww_utils

        # Try official download utility
        if hasattr(oww_utils, "download_models"):
            print("Downloading openWakeWord pretrained models...")
            oww_utils.download_models()
            return True
        elif hasattr(oww_utils, "download_pretrained_models"):
            print("Downloading openWakeWord pretrained models...")
            oww_utils.download_pretrained_models()
            return True
        else:
            # Manual download via requests
            print("Auto-download not available. Downloading models manually...")
            return _manual_download_oww_models()
    except Exception as e:
        print(f"Model download failed: {e}")
        return False


def _manual_download_oww_models() -> bool:
    """Manually download minimum required openWakeWord model files."""
    import urllib.request
    import openwakeword

    oww_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    oww_dir.mkdir(parents=True, exist_ok=True)

    # Minimum required models from openWakeWord GitHub releases
    models = {
        "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
        "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
        "alexa_v0.1.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.onnx",
    }

    for fname, url in models.items():
        dest = oww_dir / fname
        if dest.exists():
            print(f"  {fname}: already present")
            continue
        print(f"  Downloading {fname}...")
        try:
            urllib.request.urlretrieve(url, str(dest))
            print(f"  {fname}: OK ({dest.stat().st_size // 1024}KB)")
        except Exception as e:
            print(f"  {fname}: FAILED ({e})")
            return False
    return True


def train_openwakeword_model(
    phrase: str,
    positive_dir: Path,
    output_dir: Path,
) -> Path | None:
    """Train openWakeWord verifier and export ONNX."""
    try:
        import openwakeword
    except ImportError:
        print("ERROR: pip install openwakeword")
        sys.exit(1)

    # Ensure base models are downloaded
    if not download_openwakeword_models():
        print("Could not download base models. Falling back to sklearn.")

    model_name = phrase.replace(" ", "_").lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}.onnx"

    print(f"\nTraining openWakeWord model for '{phrase}'...")
    print(f"Positive samples: {len(list(positive_dir.glob('*.wav')))}")
    print(f"Output: {output_path}")

    return _train_sklearn_fallback(phrase, positive_dir, output_dir, model_name)


def _extract_oww_scores(oww, audio_float32: "np.ndarray", sr: int = 16000) -> "np.ndarray | None":
    """
    Extract openWakeWord prediction scores as feature vector for a WAV clip.
    Uses the raw model prediction scores across all built-in keywords as features.
    This gives a meaningful embedding: audio that sounds like 'alexa' gets high
    alexa score, audio that sounds like 'jarvis' gets high jarvis score, etc.
    'tara' audio will likely score high on phonetically similar keywords.
    """
    import numpy as np
    audio_int16 = (np.clip(audio_float32, -1, 1) * 32767).astype(np.int16)
    chunk_size = 1280  # 80ms at 16kHz
    all_scores = []

    oww.reset()
    for i in range(0, len(audio_int16) - chunk_size + 1, chunk_size):
        chunk = audio_int16[i : i + chunk_size]
        pred = oww.predict(chunk)  # dict: {keyword: score}
        if pred:
            scores = list(pred.values())
            all_scores.append(scores)

    if not all_scores:
        # Audio too short — pad and try once
        padded = np.pad(audio_int16, (0, max(0, chunk_size - len(audio_int16))))
        pred = oww.predict(padded[:chunk_size])
        if pred:
            return np.array(list(pred.values()), dtype=np.float32)
        return None

    return np.mean(all_scores, axis=0).astype(np.float32)


def _generate_kitchen_negatives(n: int, flac_path: Path, out_dir: Path) -> list[Path]:
    """Extract random 1s chunks from kitchen FLAC as negative samples."""
    import numpy as np
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.wav"))
    if len(existing) >= n:
        print(f"  Using {len(existing)} existing kitchen negatives")
        return existing[:n]

    print(f"  Extracting {n} kitchen audio chunks as negatives...")
    try:
        audio, sr = sf.read(str(flac_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
    except Exception as e:
        print(f"  Could not load FLAC: {e}")
        return []

    chunk_len = sr  # 1 second
    generated = []
    for i in range(n):
        start = random.randint(0, max(0, len(audio) - chunk_len - 1))
        chunk = audio[start : start + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        out_path = out_dir / f"kitchen_neg_{i:04d}.wav"
        sf.write(str(out_path), chunk, sr)
        generated.append(out_path)

    print(f"  Generated {len(generated)} kitchen negatives")
    return generated


def _generate_speech_negatives(n: int, out_dir: Path) -> list[Path]:
    """
    Generate TTS speech negatives — non-Tara words/phrases.

    Three categories:
    1. Common kitchen commands (what users say that is NOT "Tara")
    2. Phonetically similar confusables ("terra", "tiara", "terror", "tarot")
    3. General English phrases (diverse speech, not noise)

    This teaches the classifier: "tara" speech ≠ all other speech.
    Previous failure: only kitchen noise as negatives → classifier learned speech vs noise.
    """
    try:
        from gtts import gTTS
        from pydub import AudioSegment
    except ImportError:
        print("ERROR: pip install gtts pydub")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.wav"))
    if len(existing) >= n:
        print(f"  Using {len(existing)} existing speech negatives")
        return existing[:n]

    # Diverse negative phrases — kitchen commands, confusables, general speech
    phrases = [
        # Kitchen commands (common context — NOT Tara commands)
        "okay", "yes", "no", "stop", "start", "go", "wait", "done",
        "hello", "hey there", "what time is it", "how long", "set timer",
        "turn off", "turn on", "open", "close", "next", "back",
        "add salt", "add pepper", "stir it", "check the oven",
        "how much", "what is", "tell me", "show me",
        # Phonetically similar to "tara" — key confusables
        "terra", "tiara", "terror", "terrace", "tarot", "Torah",
        "terra cotta", "tiara please", "the terror", "hey terra",
        # General English — diverse speech distribution
        "good morning", "thank you", "please help", "I need help",
        "what happened", "where is", "can you", "do you know",
        "the weather today", "play some music", "call my friend",
        "remind me later", "set an alarm", "what is the time",
        "turn down", "volume up", "skip this", "go back",
        "ingredients list", "recipe steps", "cooking time",
    ]

    tld_variants = ["com", "co.in", "com.au", "co.uk"]
    generated = []
    phrase_cycle = phrases * ((n // len(phrases)) + 2)

    for i in range(n):
        out_path = out_dir / f"speech_neg_{i:04d}.wav"
        if out_path.exists():
            generated.append(out_path)
            continue

        phrase = phrase_cycle[i]
        tld = tld_variants[i % len(tld_variants)]
        try:
            mp3_path = out_dir / f"speech_neg_{i:04d}.mp3"
            tts = gTTS(text=phrase, lang="en", tld=tld)
            tts.save(str(mp3_path))
            audio = AudioSegment.from_mp3(str(mp3_path))
            audio = audio.set_frame_rate(16000).set_channels(1)
            # Trim to 1s max — same window as wake word check
            audio = audio[:1000]
            audio.export(str(out_path), format="wav")
            mp3_path.unlink(missing_ok=True)
            generated.append(out_path)
        except Exception as e:
            print(f"  Speech neg {i} failed: {e}")

    print(f"  Generated {len(generated)} speech negative samples")
    return generated


def _train_sklearn_fallback(
    phrase: str,
    positive_dir: Path,
    output_dir: Path,
    model_name: str,
) -> Path | None:
    """
    Train binary classifier using openWakeWord prediction scores as features.

    Feature vector: openWakeWord scores for all built-in keywords on each audio clip.
    Positive: TTS audio of 'tara'/'hey tara'
    Negative: kitchen FLAC audio chunks (real noise) + random noise

    Classifier: LogisticRegression → export pickle (ONNX via skl2onnx if available)
    """
    print("\nTraining binary classifier on openWakeWord prediction scores...")

    try:
        import numpy as np
        import soundfile as sf
        from openwakeword.model import Model
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import pickle
    except ImportError as e:
        print(f"Missing dep: {e}  →  pip install scikit-learn soundfile")
        return None

    # Load openWakeWord (base models now downloaded)
    print("Loading openWakeWord model...")
    try:
        oww = Model(inference_framework="onnx")
        print(f"  Loaded. Keywords: {list(oww.models.keys())}")
    except Exception as e:
        print(f"Could not load openWakeWord: {e}")
        return None

    # ── Positive features ────────────────────────────────────────────────────
    pos_files = sorted(positive_dir.glob("*.wav"))[:300]
    print(f"Extracting features from {len(pos_files)} positive samples...")
    X_pos = []
    for f in pos_files:
        try:
            audio, sr = sf.read(str(f))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            feat = _extract_oww_scores(oww, audio.astype(np.float32), sr)
            if feat is not None:
                X_pos.append(feat)
        except Exception:
            pass
    print(f"  Extracted {len(X_pos)} positive feature vectors")

    if not X_pos:
        print("FAILED: Could not extract any positive features.")
        return None

    # ── Negative features ────────────────────────────────────────────────────
    # THREE types of negatives — critical for low false positive rate:
    # 1. Speech negatives (non-Tara words) — teaches "tara" vs other speech
    # 2. Kitchen noise chunks — teaches "tara" vs background noise
    # 3. Phonetically similar words — teaches "tara" vs "terra"/"tiara"/"terror"
    print("Extracting negative features (speech + noise + phonetic confusables)...")

    X_neg = []

    # --- 1. Speech negatives via gTTS ---
    speech_neg_dir = TRAINING_DIR / "negative" / "speech"
    speech_neg_files = _generate_speech_negatives(150, speech_neg_dir)
    for f in speech_neg_files:
        try:
            audio, sr = sf.read(str(f))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            feat = _extract_oww_scores(oww, audio.astype(np.float32), sr)
            if feat is not None:
                X_neg.append(feat)
        except Exception:
            pass
    print(f"  Speech negatives: {len(X_neg)} feature vectors")

    # --- 2. Kitchen noise negatives ---
    neg_dir = TRAINING_DIR / "negative" / "kitchen"
    flac_path = Path("assets/tara_assignment_recording_clipped.flac")
    kitchen_count_before = len(X_neg)
    neg_files = _generate_kitchen_negatives(100, flac_path, neg_dir)
    for f in neg_files:
        try:
            audio, sr = sf.read(str(f))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            feat = _extract_oww_scores(oww, audio.astype(np.float32), sr)
            if feat is not None:
                X_neg.append(feat)
        except Exception:
            pass
    print(f"  Kitchen negatives: {len(X_neg) - kitchen_count_before} feature vectors")

    # --- 3. Gaussian noise pad if needed ---
    while len(X_neg) < len(X_pos):
        noise = np.random.randn(16000).astype(np.float32) * 0.05
        feat = _extract_oww_scores(oww, noise)
        if feat is not None:
            X_neg.append(feat)

    print(f"  Total negatives: {len(X_neg)} feature vectors")

    if not X_neg:
        print("FAILED: Could not generate any negative features.")
        return None

    # ── Train ────────────────────────────────────────────────────────────────
    n = min(len(X_pos), len(X_neg))
    X = np.array(X_pos[:n] + X_neg[:n])
    y = np.array([1] * n + [0] * n)

    print(f"Training on {len(X)} samples ({n} pos + {n} neg) | features={X.shape[1]}...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"Training accuracy: {train_acc:.3f}")

    # ── Save model ───────────────────────────────────────────────────────────
    pkl_path = output_dir / f"{model_name}_clf.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"clf": clf, "phrase": phrase, "n_features": X.shape[1]}, f)
    print(f"Classifier saved: {pkl_path}")

    # Try ONNX export
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onnx_model = convert_sklearn(
            clf, initial_types=[("float_input", FloatTensorType([None, X.shape[1]]))]
        )
        onnx_path = output_dir / f"{model_name}.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved: {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"ONNX export failed ({e}) — using pickle classifier")
        return pkl_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phrases = ["tara", "hey tara"]

    for phrase in phrases:
        print(f"\n{'='*50}")
        print(f"Training: '{phrase}'")
        print(f"{'='*50}")

        safe_name = phrase.replace(" ", "_")
        pos_dir = TRAINING_DIR / "positive" / safe_name

        if not args.skip_generate:
            generate_tts_samples(phrase, args.samples, pos_dir)
        else:
            print(f"Skipping generation. Using existing: {pos_dir}")

        model_path = train_openwakeword_model(phrase, pos_dir, args.output_dir)

        if model_path:
            print(f"\nSUCCESS: Model at {model_path}")
            print("Pipeline will auto-detect this model on next run.")
        else:
            print(f"\nFAILED: Could not train model for '{phrase}'.")
            print("Use Porcupine backend instead: --wake-word-backend porcupine")

    print("\nDone. Run pipeline with:")
    print("  python scripts/run_pipeline.py assets/tara_assignment_recording_clipped.flac --iteration 4")


if __name__ == "__main__":
    main()

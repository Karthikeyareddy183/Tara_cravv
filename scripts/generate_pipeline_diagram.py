"""
Generate pipeline_flow.png — architecture flow diagram.
Uses matplotlib (no graphviz dependency).

Usage:
    python scripts/generate_pipeline_diagram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_diagram(output_path: Path = Path("docs/pipeline_flow.png")) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Title
    ax.text(5, 13.5, "Tara Voice Pipeline — Final Architecture (Iteration 4)",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # Stage definitions: (y_center, label, latency, why, color)
    stages = [
        (12.0, "Raw Audio Input\n(.flac / streaming chunks)", "", "", "#f0f0f0"),
        (10.2, "Stage 1: Noise Suppression\nModel: DeepFilterNet (ONNX)",
         "~150ms | Budget: 200ms",
         "Handles non-stationary kitchen noise\n(whistles, sizzle, chopping)\nnoisereduce too weak for impulsive noise\nPi 5: ONNX Runtime + AI HAT+ NPU",
         "#dbeafe"),
        (7.8, "Stage 2: VAD\nModel: Silero VAD",
         "~40ms | Budget: 100ms",
         "Gates audio chunks — STT only on voice\n1MB model, CPU-only, ONNX backend\nPi 5: trivially compatible, <50ms",
         "#dcfce7"),
        (5.4, "Stage 3: Wake Word\nModel: openWakeWord / Porcupine",
         "~200ms | Budget: 300ms",
         "Triggers ONLY at utterance START\n(first 1s buffer checked)\nPrevents mid-sentence Tara transcription\nPi 5: TFLite/ONNX CPU inference",
         "#fef9c3"),
        (3.0, "Stage 4: STT\nModel: faster-whisper tiny.en",
         "~280ms | Budget: 1000ms",
         "CTranslate2 int8 quantised — 4x faster\n150–300ms on CPU\nEnglish-only: smaller, faster\nCan run server-side (not Pi 5 required)",
         "#fce7f3"),
        (0.8, "Text Transcript\n(command addressed to Tara)", "", "", "#f0f0f0"),
    ]

    box_width = 5.5
    box_height = 1.2
    x_left = 2.25

    for i, (y, label, latency, why, color) in enumerate(stages):
        # Main box
        rect = mpatches.FancyBboxPatch(
            (x_left, y - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="#555",
            linewidth=1.5,
        )
        ax.add_patch(rect)

        # Label
        ax.text(5, y + 0.1, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#1a1a1a")

        # Latency badge
        if latency:
            ax.text(5, y - 0.28, latency, ha="center", va="center",
                    fontsize=7.5, color="#555", style="italic")

        # Why annotation on right
        if why:
            ax.text(8.1, y, why, ha="left", va="center",
                    fontsize=6.5, color="#333",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fafafa",
                              edgecolor="#ccc", alpha=0.8))

        # Arrow to next stage
        if i < len(stages) - 1:
            next_y = stages[i + 1][0]
            ax.annotate(
                "",
                xy=(5, next_y + box_height / 2 + 0.05),
                xytext=(5, y - box_height / 2 - 0.05),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
            )

    # VAD gate label
    ax.text(2.0, 6.6, "voice-active\nchunks only", ha="center", va="center",
            fontsize=7, color="#15803d", style="italic")

    # Wake word gate label
    ax.text(2.0, 4.2, "wake word\ndetected only", ha="center", va="center",
            fontsize=7, color="#92400e", style="italic")

    # Total latency box
    total_rect = mpatches.FancyBboxPatch(
        (0.3, 0.05), 3.5, 0.55,
        boxstyle="round,pad=0.05",
        facecolor="#ecfdf5",
        edgecolor="#16a34a",
        linewidth=2,
    )
    ax.add_patch(total_rect)
    ax.text(2.05, 0.32,
            "Target Total: ~670ms  |  Budget: 2000ms  |  Margin: 1330ms",
            ha="center", va="center", fontsize=7.5, color="#15803d", fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Diagram saved to: {output_path}")


if __name__ == "__main__":
    output = Path("docs/pipeline_flow.png")
    generate_diagram(output)

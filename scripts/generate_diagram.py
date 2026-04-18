"""Generate the Tara two-architecture flow diagram as PNG."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


BG = "#f7f8fb"
PANEL = "#ffffff"
BOX = "#ffffff"
TEXT = "#182033"
SUB = "#526070"
MUTED = "#738196"
BLUE = "#2563eb"
GREEN = "#14803c"
RED = "#c2410c"
YELLOW = "#b7791f"
PURPLE = "#7c3aed"
LINE = "#d8dee9"


def add_box(ax, x, y, w, h, title, subtitle, edge, fill=BOX, title_size=9.4):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.9,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=TEXT,
        fontfamily="DejaVu Sans",
    )
    ax.text(
        x + w / 2,
        y + h * 0.31,
        subtitle,
        ha="center",
        va="center",
        fontsize=7.3,
        color=SUB,
        fontfamily="DejaVu Sans",
    )


def arrow(ax, x1, y1, x2, y2, color=BLUE, lw=1.7, style="solid"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": lw,
            "linestyle": style,
            "shrinkA": 1,
            "shrinkB": 1,
        },
    )


def label(ax, x, y, text, color=MUTED, size=7.0, align="center"):
    ax.text(
        x,
        y,
        text,
        ha=align,
        va="center",
        fontsize=size,
        color=color,
        fontfamily="DejaVu Sans",
        style="italic",
    )


def main() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(15, 10.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    ax.text(
        7.5,
        10.05,
        "TARA Voice Pipeline - Two Architectures Tried",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=TEXT,
        fontfamily="DejaVu Sans",
    )
    ax.text(
        7.5,
        9.72,
        "Same noisy kitchen recording; local latency path vs cloud accuracy path",
        ha="center",
        va="center",
        fontsize=9.2,
        color=SUB,
        fontfamily="DejaVu Sans",
    )

    # Streaming note: keeps the production-oriented experiment visible without
    # overloading the main batch-file result path.
    add_box(
        ax,
        0.9,
        8.25,
        3.2,
        0.55,
        "STREAMING MIC PATH",
        "mic input -> 32ms chunks -> same stages",
        YELLOW,
        fill="#fff8e6",
        title_size=8.6,
    )
    arrow(ax, 4.1, 8.52, 5.0, 8.36, YELLOW, lw=1.3, style="dashed")
    ax.text(
        4.62,
        8.45,
        "replaces file input",
        ha="center",
        va="center",
        fontsize=7.6,
        color=YELLOW,
        fontfamily="DejaVu Sans",
        style="italic",
        bbox={"facecolor": BG, "edgecolor": "none", "pad": 0.8},
    )
    label(ax, 2.5, 8.06, "Iteration 4g real-time path", size=7.4)

    add_box(
        ax,
        11.0,
        8.16,
        3.15,
        0.68,
        "ORDER ALSO TESTED",
        "",
        MUTED,
        fill="#f2f4f8",
        title_size=8.3,
    )
    ax.text(11.48, 8.38, "Raw", ha="center", va="center", fontsize=7.2, color=SUB, fontfamily="DejaVu Sans")
    ax.text(12.58, 8.38, "DeepFilterNet", ha="center", va="center", fontsize=7.2, color=SUB, fontfamily="DejaVu Sans")
    ax.text(13.62, 8.38, "VAD", ha="center", va="center", fontsize=7.2, color=SUB, fontfamily="DejaVu Sans")
    arrow(ax, 11.72, 8.38, 12.03, 8.38, MUTED, lw=1.0)
    arrow(ax, 13.03, 8.38, 13.38, 8.38, MUTED, lw=1.0)
    label(ax, 12.58, 7.96, "Iterations 3 and early 4", size=7.4)

    # Shared preprocessing panel.
    shared_panel = FancyBboxPatch(
        (4.65, 6.12),
        5.7,
        2.95,
        boxstyle="round,pad=0.16,rounding_size=0.08",
        facecolor="#eef5ff",
        edgecolor=LINE,
        linewidth=1.4,
        linestyle="--",
    )
    ax.add_patch(shared_panel)
    ax.text(
        7.5,
        8.88,
        "LATEST COMMON PREPROCESSING",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=BLUE,
        fontfamily="DejaVu Sans",
    )

    add_box(
        ax,
        5.0,
        8.05,
        5.0,
        0.62,
        "RAW AUDIO INPUT",
        "143s FLAC or streaming PCM | 16kHz mono | kitchen noise + speech",
        YELLOW,
    )
    add_box(
        ax,
        5.0,
        7.15,
        5.0,
        0.62,
        "Silero VAD",
        "raw float32 PCM -> candidate speech segments | streaming ~2ms/chunk",
        BLUE,
    )
    add_box(
        ax,
        5.0,
        6.25,
        5.0,
        0.62,
        "DeepFilterNet3",
        "speech windows -> denoised segments | streaming ~10ms/chunk",
        BLUE,
    )

    arrow(ax, 7.5, 8.05, 7.5, 7.77)
    arrow(ax, 7.5, 7.15, 7.5, 6.87)

    # Split into two architectures.
    arrow(ax, 7.5, 6.25, 3.25, 5.38, GREEN)
    arrow(ax, 7.5, 6.25, 11.75, 5.38, PURPLE)
    label(ax, 4.65, 5.98, "denoised segments")
    label(ax, 10.35, 5.98, "denoised segments")

    add_box(
        ax,
        0.75,
        4.62,
        5.2,
        0.76,
        "Architecture 1: Local Low-Latency",
        "Porcupine recommended | OWW + whisper-phoneme also tested",
        GREEN,
        fill="#f0fdf4",
    )
    add_box(
        ax,
        0.75,
        3.58,
        5.2,
        0.76,
        "faster-whisper STT",
        "tiny.en int8 | command-only audio | warm avg can be under 1s",
        GREEN,
        fill="#f0fdf4",
    )
    add_box(
        ax,
        0.75,
        2.25,
        5.2,
        0.9,
        "Result",
        "Latency architecture validated; wake model needs low-SNR retraining",
        GREEN,
        fill="#e8f8ef",
    )
    arrow(ax, 3.35, 4.62, 3.35, 4.34, GREEN)
    label(ax, 4.78, 4.47, "wake trigger/reject")
    arrow(ax, 3.35, 3.58, 3.35, 3.15, GREEN)
    label(ax, 4.72, 3.36, "local transcript")
    ax.text(
        3.35,
        2.02,
        "Budget fit: promising for latency | Not ready until wake model is retrained",
        ha="center",
        va="center",
        fontsize=7.4,
        color=GREEN,
        fontfamily="DejaVu Sans",
    )

    add_box(
        ax,
        9.05,
        4.62,
        5.2,
        0.76,
        "Architecture 2: Cloud Accuracy",
        "Deepgram wake word | Nova-3 + keyterm=Tara | avg ~2498ms",
        PURPLE,
        fill="#f5f0ff",
    )
    add_box(
        ax,
        9.05,
        3.58,
        5.2,
        0.76,
        "Deepgram STT",
        "Nova-2/3 | post-wake command audio | avg ~2305ms",
        PURPLE,
        fill="#f5f0ff",
    )
    add_box(
        ax,
        9.05,
        2.25,
        5.2,
        0.9,
        "Result",
        '"Can you tell me what is the next step of the recipe?"',
        PURPLE,
        fill="#efe7ff",
    )
    arrow(ax, 11.65, 4.62, 11.65, 4.34, PURPLE)
    label(ax, 13.20, 4.47, "wake confidence + time")
    arrow(ax, 11.65, 3.58, 11.65, 3.15, PURPLE)
    label(ax, 13.07, 3.36, "transcript + confidence")
    ax.text(
        11.65,
        2.02,
        "Budget fit: NO | Best recovered command on assignment audio",
        ha="center",
        va="center",
        fontsize=7.4,
        color=RED,
        fontfamily="DejaVu Sans",
    )

    # Decision panel.
    panel = FancyBboxPatch(
        (1.2, 0.55),
        12.6,
        1.12,
        boxstyle="round,pad=0.13,rounding_size=0.08",
        facecolor=PANEL,
        edgecolor=YELLOW,
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(panel)
    ax.text(
        7.5,
        1.36,
        "FINAL DECISION",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=YELLOW,
        fontfamily="DejaVu Sans",
    )
    ax.text(
        7.5,
        1.07,
        "Use Deepgram path as the accuracy proof/fallback for this noisy recording.",
        ha="center",
        va="center",
        fontsize=8.2,
        color=TEXT,
        fontfamily="DejaVu Sans",
    )
    ax.text(
        7.5,
        0.82,
        "Use local Porcupine + faster-whisper for production latency after wake-word retraining and better mic capture.",
        ha="center",
        va="center",
        fontsize=8.2,
        color=TEXT,
        fontfamily="DejaVu Sans",
    )

    plt.tight_layout(pad=0.35)
    plt.savefig("docs/pipeline_flow.png", dpi=180, bbox_inches="tight", facecolor=BG)
    print("Saved: docs/pipeline_flow.png")


if __name__ == "__main__":
    main()

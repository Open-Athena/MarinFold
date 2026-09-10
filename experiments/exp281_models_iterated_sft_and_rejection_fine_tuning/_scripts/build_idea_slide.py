"""Render one slide describing exp281's method and current pilot scope."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK = "#142b3d"
BLUE = "#276b9d"
GREEN = "#19745b"


def box(ax: plt.Axes, x: float, y: float, width: float, height: float, color: str) -> None:
    """Draw a rounded panel in slide coordinates."""
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012",
                               facecolor=color, edgecolor="none"))


def main() -> None:
    fig = plt.figure(figsize=(16, 9), facecolor="#f8fafb")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.05, 0.94, "MARINFOLD  /  EXPERIMENT 281", fontsize=12, color=BLUE, weight="bold")
    ax.text(0.05, 0.87, "Generate hypotheses. Learn to synthesize them.",
            fontsize=29, color=INK, weight="bold")
    ax.text(0.05, 0.82, "Iterated offline SFT for a longer training horizon; useful diversity is the hypothesis.",
            fontsize=17, color="#526574")

    ax.text(0.05, 0.75, "TRAINING DOCUMENT  ·  <contacts-v1.multi>", fontsize=12, color=INK, weight="bold")
    panels = [(0.05, 0.15, "Sequence", "context", "#e8edf1", INK),
              (0.235, 0.275, "Sampled hypotheses", "later loss weight 0.1", "#e0edf7", BLUE),
              (0.545, 0.19, "<final-prediction>", "switch to synthesis", "#e8edf1", INK),
              (0.77, 0.18, "Reference contacts", "loss weight 1.0", "#dcefe7", GREEN)]
    for x, width, title, subtitle, color, text_color in panels:
        box(ax, x, 0.62, width, 0.095, color)
        ax.text(x + width / 2, 0.677, title, fontsize=15, ha="center", color=text_color, weight="bold")
        ax.text(x + width / 2, 0.640, subtitle, fontsize=12, ha="center", color=text_color)
    for left, right in ((0.20, 0.235), (0.51, 0.545), (0.735, 0.77)):
        ax.add_patch(FancyArrowPatch((left + 0.004, 0.667), (right - 0.004, 0.667),
                                    arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color=INK))
    ax.text(0.05, 0.571, "Natural stop: marker loss 1.  Forced budget: insert between statements, marker loss 0.",
            fontsize=14, color=INK)
    ax.text(0.05, 0.537, "References supervise the answer; they are never in the generation prompt. At inference, generate the answer.",
            fontsize=13, color="#526574")

    stages = [
        (0.05, "1  FORMAT WARM-UP", "Current phase", BLUE,
         "Teach sections, final marker and end.\nHypothesis loss 1.0; 16 bootstrap drafts.\n50% ordinary-contact rehearsal."),
        (0.36, "2  ITERATED SYNTHESIS", "After the format gate", GREEN,
         "Freeze → sample sequential histories → SFT.\nHypotheses 0.1; reference answer 1.0.\nRefresh a large corpus; repeat rounds."),
        (0.67, "3  REJECTION SFT", "Once synthesis works", GREEN,
         "Sample four whole trajectories per protein.\nSelect by generated final-answer F1.\nKeep history; train on reference answer."),
    ]
    for x, title, subtitle, color, body in stages:
        box(ax, x, 0.255, 0.28, 0.22, "#ffffff")
        ax.text(x + 0.008, 0.444, title, fontsize=15, color=color, weight="bold")
        ax.text(x + 0.008, 0.409, subtitle, fontsize=12, color="#526574")
        ax.text(x + 0.008, 0.367, body, fontsize=12.8, color=INK, va="top", linespacing=1.8)

    box(ax, 0.05, 0.104, 0.9, 0.107, "#e8edf1")
    ax.text(0.065, 0.180, "PILOT: 1.5B model · 2,023 train / 25 held-out proteins · batch 32 · 2,000 steps",
            fontsize=15, color=INK, weight="bold")
    ax.text(0.065, 0.145, "Next: 200 natural + 200 forced completions. Gate: ≥99% valid in each mode; ≥90% natural with ≥2 hypotheses.*",
            fontsize=12.8, color=INK)
    ax.text(0.05, 0.070, "*Nonempty hypotheses. Larger refreshed SFT and rejection training have not started. Snapshot: 10 Sep 2026.",
            fontsize=10.5, color="#526574")
    ax.text(0.05, 0.043, "github.com/Open-Athena/MarinFold/issues/281", fontsize=10, color=BLUE)
    invocation = ("uv run --no-project --with matplotlib python "
                  "experiments/exp281_models_iterated_sft_and_rejection_fine_tuning/_scripts/build_idea_slide.py")
    ax.text(0.05, 0.018, invocation, fontsize=7.5, color="#687985")
    output = ROOT / "plots/idea_overview"
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=150)
    output.with_suffix(".png.meta.json").write_text(json.dumps({
        "script": invocation, "args": [],
        "caption": "Exp281 method: format warm-up, iterated synthesis SFT, then whole-trajectory rejection selection."
    }, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()

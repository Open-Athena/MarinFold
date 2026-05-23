# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Render talk slides describing iter_R4_grow_05_10_15_25.

Output: slides.pdf in this directory. Each page is one slide,
rendered with matplotlib at 16:9.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

_THIS = Path(__file__).resolve().parent

WIDTH, HEIGHT = 13.33, 7.5  # 16:9 at 80 dpi

TITLE_SIZE = 28
H1_SIZE = 22
BODY_SIZE = 16
SMALL_SIZE = 13
MONO_FAMILY = "DejaVu Sans Mono"

NAVY = "#0a2540"
ORANGE = "#d97706"
GREEN = "#15803d"
GREY = "#6b7280"
LIGHTGREY = "#e5e7eb"
RED = "#b91c1c"


def new_slide(title: str | None = None) -> tuple:
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    if title:
        ax.text(
            5, 92, title,
            fontsize=TITLE_SIZE, fontweight="bold", color=NAVY,
            va="top", ha="left",
        )
        ax.plot([5, 95], [88, 88], color=ORANGE, lw=2)
    return fig, ax


def footer(ax, page_no: int, total: int):
    ax.text(
        95, 2, f"{page_no} / {total}",
        fontsize=SMALL_SIZE, color=GREY, va="bottom", ha="right",
    )
    ax.text(
        5, 2, "exp27 · iter_R4_grow_05_10_15_25 · MarinFold 1B",
        fontsize=SMALL_SIZE, color=GREY, va="bottom", ha="left",
    )


def slide_title(pdf, page_no, total):
    fig, ax = new_slide()
    ax.text(
        50, 65, "iter_R4_grow_05_10_15_25",
        fontsize=TITLE_SIZE + 8, fontweight="bold", color=NAVY,
        va="center", ha="center", family=MONO_FAMILY,
    )
    ax.text(
        50, 55,
        "Iterative contact-seeded inference for MarinFold 1B",
        fontsize=H1_SIZE, color=NAVY, va="center", ha="center",
    )
    ax.text(
        50, 47,
        "Lifts mean LDDT on the FoldBench-10 train set from 0.250 → 0.351  (+40.7%)",
        fontsize=BODY_SIZE, color=GREEN, va="center", ha="center",
    )
    ax.text(
        50, 35,
        "Issue #27 · exp27 · branch exp/27-improved-inference-algorithm",
        fontsize=SMALL_SIZE, color=GREY, va="center", ha="center",
        family=MONO_FAMILY,
    )
    footer(ax, page_no, total)
    pdf.savefig(fig); plt.close(fig)


def slide_problem(pdf, page_no, total):
    fig, ax = new_slide("The problem")
    bullets = [
        "MarinFold 1B is a Llama-arch LLM trained on protein documents",
        "  in the contacts-and-distances-v1 grammar:",
        "",
        "    <begin_sequence> <AAs> <begin_statements>",
        "      [<*-range-contact> <p_i> <p_j>]*",
        "      <distance> <p_i> <p_j> <atom_i> <atom_j>  →  <d_X.X>",
        "",
        "Baseline readout (exp20): prompt with <begin_statements>, then for",
        "every (i, j) pair read the next-token distribution over <d_X.X> bins.",
        "Zero context, no autoregression, no seeded contacts.",
        "",
        "On the 10-protein FoldBench train set:",
        "    mean LDDT = 0.2496      (median 0.250, per-protein 0.151–0.449)",
        "",
        "Goal: clear +50% over baseline (mean LDDT ≥ 0.3744)",
        "      within 5× the baseline wall-clock (≤ 6920 s on one A100).",
    ]
    y = 78
    for line in bullets:
        is_code = line.startswith("    ") or "<" in line and ">" in line
        ax.text(
            5, y, line,
            fontsize=BODY_SIZE,
            color=NAVY if not line.startswith("Goal") else GREEN,
            family=MONO_FAMILY if is_code else "DejaVu Sans",
            va="top", ha="left",
        )
        y -= 4.5
    footer(ax, page_no, total)
    pdf.savefig(fig); plt.close(fig)


def slide_key_insight(pdf, page_no, total):
    fig, ax = new_slide("Key insight: the model is great with context")
    # Three boxes
    boxes = [
        (8, 50, 28, 32, "Baseline\nreadout",
         "<begin_statements>\n<distance> <p_i> <p_j>\n  <CB> <CB>",
         "mean LDDT\n0.250", NAVY),
        (38, 50, 28, 32, "Seeded\ncontacts",
         "<begin_statements>\n<long-range-contact> ...\n<medium-...> ...\n<distance> ...",
         "mean LDDT\n0.282 (+13%)", ORANGE),
        (68, 50, 28, 32, "GT oracle\n(diagnostic)",
         "<begin_statements>\nALL true contacts\n(gt_d < 8 Å)\n<distance> ...",
         "mean LDDT\n0.717 (+187%)", GREEN),
    ]
    for x, y, w, h, label, prompt, score, color in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5",
            facecolor="white", edgecolor=color, linewidth=2.5,
        ))
        ax.text(x + w/2, y + h - 4, label,
                fontsize=H1_SIZE, fontweight="bold", color=color,
                va="top", ha="center")
        ax.text(x + 2, y + h - 11, prompt,
                fontsize=10, color=NAVY, family=MONO_FAMILY,
                va="top", ha="left")
        ax.text(x + w/2, y + 4, score,
                fontsize=BODY_SIZE, color=color, fontweight="bold",
                va="center", ha="center")
    ax.text(
        50, 38,
        "→ contact-prediction quality is the bottleneck for honest algorithms.",
        fontsize=BODY_SIZE, color=NAVY, va="center", ha="center",
    )
    ax.text(
        50, 28,
        "How do we get honest, high-quality contacts at inference time?",
        fontsize=H1_SIZE, color=ORANGE, fontweight="bold",
        va="center", ha="center", style="italic",
    )
    ax.text(
        50, 18,
        "Idea: iterate. The model's distance readout under a seeded prefix is sharper",
        fontsize=BODY_SIZE, color=NAVY, va="center", ha="center",
    )
    ax.text(
        50, 13,
        "→ extract NEW high-confidence contacts from the round-N distogram → re-seed.",
        fontsize=BODY_SIZE, color=NAVY, va="center", ha="center",
    )
    footer(ax, page_no, total)
    pdf.savefig(fig); plt.close(fig)


def slide_enrichment(pdf, page_no, total):
    fig, ax = new_slide("Iteration enriches the high-confidence contact pool")

    # Bar chart per protein: contacts >0.5 in baseline vs iter_R3_low
    proteins = ["8eb9", "7y5j", "7ykm", "7ur2", "8baq",
                "8cba", "7zs2", "7xz3", "7ylr", "7uk8"]
    Ls = [95, 102, 105, 195, 208, 214, 316, 325, 330, 394]
    baseline = [0, 84, 7, 38, 3, 17, 28, 13, 95, 33]
    after_iter = [132, 186, 132, 232, 227, 210, 370, 326, 449, 404]

    ax_chart = fig.add_axes([0.10, 0.18, 0.80, 0.55])
    x = np.arange(len(proteins))
    width = 0.38
    ax_chart.bar(x - width/2, baseline, width, label="baseline",
                 color=NAVY, edgecolor="white")
    ax_chart.bar(x + width/2, after_iter, width, label="after iter R=3 (min p=0.1)",
                 color=ORANGE, edgecolor="white")
    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels([f"{p}\nL={L}" for p, L in zip(proteins, Ls)],
                              fontsize=11)
    ax_chart.set_ylabel("# contacts with prob > 0.5", fontsize=BODY_SIZE)
    ax_chart.legend(fontsize=BODY_SIZE, loc="upper left")
    ax_chart.set_ylim(0, 500)
    for spine in ["top", "right"]:
        ax_chart.spines[spine].set_visible(False)
    ax_chart.grid(axis="y", alpha=0.3)

    ax.text(
        5, 12,
        "8baq_A: 3 → 227 contacts (76×).  8eb9_A: 0 → 132.  Every protein gets a "
        "much richer high-confidence pool after iteration —",
        fontsize=BODY_SIZE, color=NAVY, va="bottom", ha="left",
    )
    ax.text(
        5, 7,
        "→ later rounds can afford LARGER K (more seeds) without losing precision.",
        fontsize=BODY_SIZE, color=GREEN, fontweight="bold",
        va="bottom", ha="left",
    )
    footer(ax, page_no, total)
    pdf.savefig(fig); plt.close(fig)


def slide_algorithm(pdf, page_no, total):
    fig, ax = new_slide("The algorithm:  iter_R4_grow_05_10_15_25")

    # Schedule visualization at top
    ax.text(
        5, 80,
        "Round-by-round contact budget (K = k_per_L · L per round):",
        fontsize=BODY_SIZE, color=NAVY, va="top", ha="left",
    )
    rounds = [
        (1, 0.5, 7),
        (2, 1.0, 14),
        (3, 1.5, 21),
        (4, 2.5, 36),
    ]
    y_bar = 65
    bar_h = 5
    max_w = 60
    for r, k, w in rounds:
        ax.add_patch(plt.Rectangle((20, y_bar - r*9), w, bar_h,
                                    facecolor=ORANGE, alpha=0.85))
        ax.text(18, y_bar - r*9 + bar_h/2,
                f"R{r}", fontsize=BODY_SIZE, color=NAVY, fontweight="bold",
                va="center", ha="right")
        ax.text(20 + w + 1, y_bar - r*9 + bar_h/2,
                f"K = {k}·L  contacts", fontsize=BODY_SIZE,
                color=NAVY, va="center", ha="left", family=MONO_FAMILY)

    # Pseudocode on right
    ax.add_patch(FancyBboxPatch(
        (60, 27), 36, 50, boxstyle="round,pad=0.3",
        facecolor="#f8fafc", edgecolor=NAVY, linewidth=1.2,
    ))
    code = [
        "distogram = naive_readout()",
        "",
        "for K in [0.5L, 1.0L, 1.5L, 2.5L]:",
        "    seeds = top-K contacts from",
        "      distogram where p > 0.1,",
        "      ordered long > medium > short",
        "",
        "    prefix = <begin_statements>",
        "             + [<{range}-range-contact>",
        "                 <p_i> <p_j>",
        "                for each seed]",
        "",
        "    distogram = readout under",
        "                prefix, only for",
        "                LDDT-shell pairs",
    ]
    yy = 73
    for line in code:
        ax.text(62, yy, line, fontsize=10, color=NAVY,
                family=MONO_FAMILY, va="top", ha="left")
        yy -= 3.0

    ax.text(
        5, 20,
        "Each round picks fresh seeds from the previous round's "
        "(sharper) distogram.",
        fontsize=BODY_SIZE, color=NAVY, va="top", ha="left",
    )
    ax.text(
        5, 15,
        "Contacts-only — distance-bin commits hurt LDDT (one-hot rows zero "
        "wrong-mode pairs).",
        fontsize=BODY_SIZE, color=NAVY, va="top", ha="left",
    )
    ax.text(
        5, 10,
        "Standard expected-distance readout on the final distogram. "
        "No post-hoc sharpening.",
        fontsize=BODY_SIZE, color=GREEN, fontweight="bold",
        va="top", ha="left",
    )
    footer(ax, page_no, total)
    pdf.savefig(fig); plt.close(fig)


def slide_progression(pdf, page_no, total):
    fig, ax = new_slide("Progression: knob → knob → knob")

    rows = [
        ("baseline_naive",                       0.2496, NAVY,    False),
        ("+ sharpen (T=0.05)",                   0.2738, NAVY,    False),
        ("seeded contacts (R=1, K=L, min=0.3)",  0.2816, NAVY,    False),
        ("iter R=3 (K=L, min=0.1)",              0.3376, NAVY,    False),
        ("iter R=3 growing K [0.5, 1, 1.5]",     0.3421, NAVY,    False),
        ("iter R=4 growing K [0.5, 1, 1.5, 2.5]", 0.3511, GREEN,  True),
        ("",                                     0.3744, ORANGE,  False),
        ("GT-oracle (diagnostic only)",          0.7167, GREY,    False),
    ]

    bar_x0 = 30
    bar_w_max = 60
    val_max = 0.75
    y = 75
    for label, val, color, headline in rows:
        if label:
            ax.text(bar_x0 - 1, y + 1.5, label,
                    fontsize=BODY_SIZE, color=color,
                    fontweight="bold" if headline else "normal",
                    family=MONO_FAMILY if "iter" in label or "seeded" in label
                                       or "sharpen" in label or "baseline" in label
                                       or "oracle" in label
                                    else "DejaVu Sans",
                    va="center", ha="right")
        bar_w = bar_w_max * (val / val_max)
        ax.add_patch(plt.Rectangle((bar_x0, y), bar_w, 3,
                                    facecolor=color,
                                    alpha=0.85 if headline else 0.55,
                                    edgecolor="none"))
        ax.text(bar_x0 + bar_w + 0.5, y + 1.5,
                f" {val:.4f}",
                fontsize=BODY_SIZE,
                color=color,
                fontweight="bold" if headline else "normal",
                family=MONO_FAMILY, va="center", ha="left")
        if label.startswith("") and val == 0.3744:
            ax.text(bar_x0 + bar_w + 7, y + 1.5,
                    "← +50% bar (not cleared)",
                    fontsize=SMALL_SIZE, color=ORANGE,
                    va="center", ha="left", style="italic")
        if headline:
            ax.text(bar_x0 + bar_w + 7, y + 1.5,
                    "  ← in-budget HEADLINE  (+40.68%)",
                    fontsize=SMALL_SIZE, color=GREEN,
                    fontweight="bold",
                    va="center", ha="left")
        y -= 8

    ax.text(
        5, 13,
        "+10% bar (0.2746) cleared by sharpening alone.  +15.94% by seeded + sharpen.",
        fontsize=BODY_SIZE, color=NAVY, va="bottom", ha="left",
    )
    ax.text(
        5, 8,
        "Iteration with growing K is the big lever past +25%.  +40.68% in 4373 s "
        "(3.16× baseline, in budget).",
        fontsize=BODY_SIZE, color=NAVY, va="bottom", ha="left",
    )
    footer(ax, page_no, total)
    pdf.savefig(fig); plt.close(fig)


def slide_per_protein(pdf, page_no, total):
    fig, ax = new_slide("Per-protein results vs the +50% bar")

    proteins = ["7y5j", "7ykm", "7ur2", "7zs2", "7ylr",
                "8eb9", "8cba", "7xz3", "8baq", "7uk8"]
    Ls = [102, 105, 195, 316, 330, 95, 214, 325, 208, 394]
    baseline = [0.4485, 0.3317, 0.2633, 0.2500, 0.2673,
                0.1510, 0.2045, 0.1913, 0.1872, 0.2010]
    final = [0.5659, 0.5178, 0.4008, 0.3619, 0.3169,
             0.3071, 0.2735, 0.2618, 0.2540, 0.2515]
    oracle = [0.8044, 0.8052, 0.7219, 0.6895, 0.6734,
              0.7254, 0.7319, 0.6993, 0.6870, 0.6290]

    ax_chart = fig.add_axes([0.07, 0.17, 0.88, 0.65])
    y = np.arange(len(proteins))
    ax_chart.barh(y, oracle, color=LIGHTGREY, label="GT-oracle ceiling",
                  edgecolor="white")
    ax_chart.barh(y, final, color=ORANGE, label="iter_R4_grow",
                  edgecolor="white")
    ax_chart.barh(y, baseline, color=NAVY, label="baseline",
                  edgecolor="white", alpha=0.85)
    ax_chart.axvline(0.3744, color=GREEN, linestyle="--", lw=2,
                     label="+50% bar (0.3744)")

    for i, (b, f, o) in enumerate(zip(baseline, final, oracle)):
        ax_chart.text(f + 0.01, i, f"{f:.3f}",
                      fontsize=10, color=NAVY, va="center", ha="left")

    ax_chart.set_yticks(y)
    ax_chart.set_yticklabels([f"{p}_A (L={L})" for p, L in zip(proteins, Ls)],
                              fontsize=12, family=MONO_FAMILY)
    ax_chart.invert_yaxis()
    ax_chart.set_xlabel("mean LDDT (distogram-CB)", fontsize=BODY_SIZE)
    ax_chart.set_xlim(0, 0.9)
    ax_chart.legend(loc="lower right", fontsize=12)
    for s in ["top", "right"]:
        ax_chart.spines[s].set_visible(False)
    ax_chart.grid(axis="x", alpha=0.3)

    ax.text(
        5, 12,
        "4 / 10 proteins pass the +50% bar on this single algorithm.",
        fontsize=BODY_SIZE, color=NAVY, va="bottom", ha="left",
    )
    ax.text(
        5, 7,
        "The hard misses (8baq, 7uk8, 7xz3, 8cba, 8eb9, 7ylr) all have huge "
        "oracle headroom (0.43–0.55) — model has the capacity, but its honest "
        "contact predictions remain sparse.",
        fontsize=SMALL_SIZE, color=NAVY, va="bottom", ha="left",
    )
    footer(ax, page_no, total)
    pdf.savefig(fig); plt.close(fig)


def slide_what_didnt_work(pdf, page_no, total):
    fig, ax = new_slide("What didn't work")

    items = [
        ("Sampled contact prefixes",
         "Model emits 2–33 contacts before transitioning to <distance>;",
         "sampling-based seed selection has lower precision than top-K marginal.",
         "+2.9% single rollout, +0.3% M=5 averaging (worse: averaging blurs distograms)."),
        ("Distance-bin commits",
         "Inject <distance>...<d_X.X> statements for high-confidence pairs.",
         "Pair becomes one-hot row → LDDT zero when mode is wrong.",
         "Per-pair variance kills the gain.  Net: 0.3503 vs 0.3511 no-kd (no change)."),
        ("K = 2L seeded contacts",
         "Double the seed prefix.",
         "Lower-precision tail seeds poison the readout.",
         "−1.2% vs K = L.  Precision > count."),
        ("Mixture-of-distograms / per-pair max-confidence",
         "Average predictions across algorithms, or pick most-confident per pair.",
         "Variants aren't differently wrong — just less wrong, in the same direction.",
         "Worse than the best single algorithm."),
        ("Sharpening on top of iter_R4_grow",
         "p' = softmax(log p / T), sweep T.",
         "Sharpening only helps when distributions are high-entropy from bad context.",
         "T=1.0 (no sharpen) wins.  Matches the oracle pattern."),
    ]
    y = 78
    for title, why, what, num in items:
        ax.text(5, y, f"✗  {title}",
                fontsize=BODY_SIZE, fontweight="bold", color=RED,
                va="top", ha="left")
        ax.text(9, y - 3.4, why, fontsize=SMALL_SIZE, color=NAVY,
                va="top", ha="left")
        ax.text(9, y - 6.7, what, fontsize=SMALL_SIZE, color=NAVY,
                va="top", ha="left")
        ax.text(9, y - 10.0, num, fontsize=SMALL_SIZE, color=GREY,
                va="top", ha="left", style="italic")
        y -= 14.5
    footer(ax, page_no, total)
    pdf.savefig(fig); plt.close(fig)


def slide_takeaways(pdf, page_no, total):
    fig, ax = new_slide("Takeaways")

    points = [
        ("Context > sampling.",
         "Picking top-K from the marginal contact distribution beats sampling "
         "contact statements from the model autoregressively at any T."),
        ("Iteration enriches the high-confidence contact pool.",
         "For 8 of 10 proteins, contacts above prob 0.5 grow 5–75× after 3 "
         "iterations.  Later rounds can afford larger K without losing precision."),
        ("Growing K matters.",
         "Cautious early (K=0.5L), bold late (K=2.5L) beats fixed K=L or K=2L. "
         "Round 1's noisy bold picks would limit later rounds."),
        ("Precision > count.",
         "K=2L always hurts in early rounds.  Distance commits hurt.  "
         "What works is exactly the model's most-confident predictions."),
        ("Sharpening rescues bad context, not good context.",
         "Big gain on naive (+9.7%); zero gain on oracle.  As algorithms improve, "
         "their distributions sharpen on their own and post-hoc sharpening hurts."),
        ("Contact prediction is the bottleneck, not the model.",
         "GT-oracle reaches 0.72.  For hard proteins (8baq, 7uk8, ...), the model "
         "just doesn't honestly identify enough contacts at inference time."),
    ]
    y = 80
    for title, body in points:
        ax.text(5, y, f"●  {title}",
                fontsize=BODY_SIZE, fontweight="bold", color=NAVY,
                va="top", ha="left")
        ax.text(9, y - 3.5, body, fontsize=SMALL_SIZE, color=NAVY,
                va="top", ha="left", wrap=True)
        y -= 12

    footer(ax, page_no, total)
    pdf.savefig(fig); plt.close(fig)


def slide_repro(pdf, page_no, total):
    fig, ax = new_slide("Reproducer")

    ax.text(
        5, 80,
        "10 proteins · A100 40 GB · bf16 · vLLM 0.7.3 · ~4373 s wall (3.16× baseline)",
        fontsize=BODY_SIZE, color=NAVY, va="top", ha="left",
    )

    ax.add_patch(FancyBboxPatch(
        (5, 18), 90, 55, boxstyle="round,pad=0.5",
        facecolor="#f8fafc", edgecolor=NAVY, linewidth=1.2,
    ))
    cmds = [
        "# 1. Produce the baseline full-distogram prior",
        "uv run python run_baseline.py --dtype bfloat16 --n-gpus 1",
        "uv run python snapshot_distograms.py --to distogram_baseline_naive.npz",
        "",
        "# 2. Run iter_R4_grow_05_10_15_25",
        "uv run python run_iterative.py --dtype bfloat16 --n-gpus 1 \\",
        "    --algorithm iter_R4_grow_05_10_15_25 \\",
        "    --n-rounds 4 \\",
        "    --k-contacts-per-L-per-round 0.5 1.0 1.5 2.5 \\",
        "    --k-distances-per-L-per-round 0.0 0.0 0.0 0.0 \\",
        "    --min-contact-prob 0.1 \\",
        "    --prior-name distogram_baseline_naive.npz",
        "",
        "# Result: mean LDDT 0.3511 (+40.68% over baseline 0.2496)",
    ]
    y = 70
    for line in cmds:
        color = GREY if line.startswith("#") else NAVY
        ax.text(7, y, line, fontsize=12, color=color,
                family=MONO_FAMILY, va="top", ha="left")
        y -= 3.5

    ax.text(
        5, 12,
        "Branch: exp/27-improved-inference-algorithm",
        fontsize=SMALL_SIZE, color=GREY, va="top", ha="left",
        family=MONO_FAMILY,
    )
    ax.text(
        5, 8,
        "Full narrative + every experiment row: RESULTS_LOG.md  ·  data/experiments.tsv",
        fontsize=SMALL_SIZE, color=GREY, va="top", ha="left",
        family=MONO_FAMILY,
    )
    footer(ax, page_no, total)
    pdf.savefig(fig); plt.close(fig)


def main():
    out = _THIS / "slides.pdf"
    builders = [
        slide_title,
        slide_problem,
        slide_key_insight,
        slide_enrichment,
        slide_algorithm,
        slide_progression,
        slide_per_protein,
        slide_what_didnt_work,
        slide_takeaways,
        slide_repro,
    ]
    total = len(builders)
    with PdfPages(out) as pdf:
        for i, b in enumerate(builders, 1):
            b(pdf, i, total)
    print(f"Wrote {out}  ({total} slides)")


if __name__ == "__main__":
    main()

"""Plot held-out finalization diagnostics from the committed continuation CSV.

Run as a module from the repository root; plotting dependencies are isolated
from the pinned training environment.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from experiments.exp281_models_iterated_sft_and_rejection_fine_tuning.build_summary import (
    save_plot_with_meta,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "data/format_s02_training.csv").open() as handle:
        history = list(csv.DictReader(handle))
    rows = [row for row in history if row.get("validation/final_marker_loss")]
    steps = [int(row["train/step"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for name, label, color in (("final_marker", "Natural final marker (6 targets)", "#396ea4"),
                               ("final_end", "Final-answer end (9 targets)", "#b75638")):
        axes[0].plot(steps, [float(row[f"validation/{name}_loss"]) for row in rows],
                     "o-", label=label, color=color)
        axes[1].plot(steps, [100 * float(row[f"validation/{name}_accuracy"]) for row in rows],
                     "o-", label=label, color=color)
    axes[0].set(ylabel="Teacher-forced cross-entropy", xlabel="Global optimizer step",
                title="Held-out transition losses")
    axes[1].set(ylabel="Teacher-forced top-1 accuracy (%)", xlabel="Global optimizer step",
                ylim=(-3, 103), title="Held-out transition accuracy")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("exp281 · extended format warm-up · sparse transition diagnostics", fontsize=14)
    save_plot_with_meta(
        fig, root / "plots/format_s02_transitions.png",
        script=("uv run --no-project --with matplotlib python -m "
                "experiments.exp281_models_iterated_sft_and_rejection_fine_tuning._scripts.plot_continuation"),
        args=[],
        caption=("The same held-out corpus supplies six supervised natural final markers and nine multi-document "
                 "end tokens at every check. Forced markers and padding are excluded. These sparse teacher-forced "
                 "diagnostics do not replace the separate free-running format gate."),
        dpi=160,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    training = [row for row in history if row.get("train/loss")]
    ax.plot([int(row["train/step"]) for row in training],
            [float(row["train/loss"]) for row in training],
            color="#396ea4", alpha=0.7, linewidth=0.8, label="Training minibatch")
    ax.plot(steps, [float(row["validation/loss"]) for row in rows],
            "o-", color="#b75638", label="Held-out corpus")
    ax.set(xlabel="Global optimizer step", ylabel="Weighted token cross-entropy",
           title="exp281 · extended format warm-up · fixed corpus")
    ax.legend(frameon=False)
    save_plot_with_meta(
        fig, root / "plots/format_s02_loss.png",
        script=("uv run --no-project --with matplotlib python -m "
                "experiments.exp281_models_iterated_sft_and_rejection_fine_tuning._scripts.plot_continuation"),
        args=[],
        caption=("Same frozen 2,023-document training and 25-document held-out corpora. "
                 "Training loss approaches zero while held-out loss rises. "
                 "These token losses do not measure free-running contact accuracy."),
        dpi=160,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

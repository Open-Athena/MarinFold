"""Plot the completed format trial from committed CSV/JSON results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-slug", default="trial_s03")
    args = parser.parse_args()
    root = Path(__file__).parent
    with (root / f"data/{args.run_slug}_training.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    gate = json.loads((root / f"data/{args.run_slug}_format_gate.json").read_text())
    validation = [row for row in rows if row.get("validation/loss")]
    training = [row for row in rows if row.get("train/loss")]
    final_step = max(int(row["train/step"]) for row in rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    ax = axes[0]
    ax.plot([int(r["train/step"]) for r in training], [float(r["train/loss"]) for r in training],
            alpha=0.45, color="#3974ad", label="Training minibatch")
    ax.plot([int(r["train/step"]) for r in validation], [float(r["validation/loss"]) for r in validation],
            "o-", color="#172d45", label="Internal validation")
    ax.set(xlabel="Optimizer step", ylabel="Weighted token cross-entropy",
           title="Teacher-forced loss decreases", xlim=(min(int(row["train/step"]) for row in rows), final_step))
    ax.legend(frameon=False)
    ax = axes[1]
    modes = ["natural", "forced"]
    values = [100 * gate["modes"][mode]["valid_fraction"] for mode in modes]
    ax.bar(["Natural finalization", "Forced finalization"], values, color="#bb5a3c", width=0.55)
    ax.axhline(99, linestyle="--", color="#555555", label="Preregistered validity gate: 99%")
    for i, mode in enumerate(modes):
        ax.text(i, values[i] + 3, f"{gate['modes'][mode]['valid']}/200", ha="center", fontsize=12)
    ax.set(ylabel="Valid complete trajectories (%)", ylim=(0, 112),
           title="Finalization has not been acquired")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.suptitle(f"exp281 · 1.5B format warm-up · step {final_step} · 8×H100", fontsize=15)
    save_plot_with_meta(
        fig, root / f"plots/{args.run_slug}.png",
        caption=("Loss improves but the fixed format gate fails on 25 held-out proteins, eight samples each. "
                 "Invalid outputs remain in the denominator. Training CSV uses the successful resumed trajectory; "
                 "duplicate optimizer steps from interrupted attempts are excluded. No protein-accuracy improvement is established."),
        dpi=160,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

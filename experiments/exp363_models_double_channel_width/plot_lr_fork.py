"""Plot the saved October 26 full-validation snapshot without API access."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta


HERE = Path(__file__).resolve().parent
DEFAULT = "contacts-v1-exp277-m2-p06-full-epoch-1.5B"
ORIGINAL = "contacts-v1-exp363-m2-p06-full-epoch-5.86B"
FORK = ORIGINAL + "-fork74178-lr7p07e-4"


def main() -> None:
    """Render original divergence and the fork's matched default comparison."""
    curves: dict[str, dict[int, float]] = {}
    with (HERE / "data/lr_fork_validation_20261026.csv").open() as handle:
        for row in csv.DictReader(handle):
            curve = curves.setdefault(row["run_id"], {})
            step, loss = int(row["global_step"]), float(row["validation_loss"])
            if step in curve or not math.isfinite(loss):
                raise ValueError(f"Invalid validation observation: {row}")
            curve[step] = loss
    matched = sorted(curves[FORK].keys() & curves[DEFAULT].keys() & curves[ORIGINAL].keys())
    if len(matched) != len(curves[FORK]):
        raise ValueError("Every fork evaluation must have exact-step controls")
    progress = json.loads((HERE / "data/progress_20261026.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    for run, label, color in [
        (ORIGINAL, "5.862B original: LR 0.001", "#c25138"),
        (FORK, "5.862B fork: LR 0.000707", "#16836d"),
        (DEFAULT, "1.471B default", "#2864a5"),
    ]:
        steps = sorted(curves[run])
        left.plot([s / 1000 for s in steps], [curves[run][s] for s in steps],
                  "o-", markersize=3, color=color, label=label)
        if run != ORIGINAL:
            right.plot([s / 1000 for s in matched], [curves[run][s] for s in matched],
                       "o-", markersize=3, color=color, label=label)
    left.set_yscale("log")
    left.set_yticks([3, 5, 10, 20, 40], labels=["3", "5", "10", "20", "40"])
    left.set(title="Original LR run diverged", ylabel="Full validation loss (log scale)")
    left.axvline(74.178, color="#777", linestyle="--", linewidth=1)
    left.text(0.02, 0.96, "Fork source: step 74,178", va="top", transform=left.transAxes, fontsize=10)
    left.legend(frameon=False, fontsize=9, loc="center right")
    right.set(title="Fork beats default at all 19 matched evaluations",
              ylabel="Full validation loss (nats/token)")
    right.legend(frameon=False, fontsize=9, loc="upper right")
    latest = matched[-1]
    gap = curves[FORK][latest] - curves[DEFAULT][latest]
    right.text(0.04, 0.04, f"Step {latest:,}: fork − default = {gap:+.5f}\n"
               f"Perplexity change: {100 * math.expm1(gap):+.2f}%",
               transform=right.transAxes, fontsize=10)
    for axis in (left, right):
        axis.set_xlabel("Optimizer step (thousands)")
        axis.grid(alpha=0.18)
    fig.suptitle("October 26, 2026 · same native + MPNN corpus and validation · one seed", fontsize=12)
    save_plot_with_meta(
        fig, HERE / "plots/lr_fork_comparison.png", dpi=180,
        caption=(
            f"W&B snapshot {progress['observed_at']}. Original high-LR divergence is shown "
            "on a log scale at left; right compares fork/default at exact matching steps. "
            "No smoothing or interpolation. LR reaches 0.000707 at step 76,293. "
            "The small validation advantage does not establish contact accuracy."
        ),
    )
    plt.close(fig)
    print(f"Matched {len(matched)} evaluations; latest fork-minus-default gap {gap:+.8f}")


if __name__ == "__main__":
    main()

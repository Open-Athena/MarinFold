# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Locate the mid-training loss transition in each 1.5B run.

Every contacts-v1 1.5B run drops ~0.09 nats over about 1,500 steps somewhere in
the middle of training. It is not the augmentation ramp (linear, and it makes
the data harder), not the learning rate, and not the token budget — all three
are identical across these runs at every step checked.

It is a learning transition, and the point of this figure is that its timing is
NOT reproducible: the two RoPE runs — exp262's control and exp232's reference,
the same architecture on the same data at the same seed — transition about
6,000 steps apart. (Review later found they also differed in embedding
initialisation; a smaller perturbation than architecture, but not identical.) That run-to-run spread is worth up to ~0.09 nats at a fixed
step, an order of magnitude more than the architecture difference exp262 is
trying to measure, so mid-run comparisons between single runs cannot resolve it.
"""

import argparse
from pathlib import Path

import matplotlib
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

RUNS = {
    "exp262 control (RoPE)": ("prot-exp262-cw-cv1-arch-full-control-p06", "#2b6cb0"),
    "exp262 NoPE + smear": ("prot-exp262-cw-cv1-arch-full-nope-smear-p06", "#dd6b20"),
    "exp232 reference (RoPE)": ("prot-exp232-cw-cv1-decontam-s02-m2-p06-aug", "#718096"),
}
BIN = 500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--low", type=int, default=10000)
    parser.add_argument("--high", type=int, default=30000)
    arguments = parser.parse_args()

    api = wandb.Api()
    figure, axes = plt.subplots(figsize=(9.5, 5.2))
    # Stagger the annotations; the NoPE and exp232 transitions are ~1k apart and
    # their labels collide at a shared height.
    heights = iter((3.335, 3.335, 3.295))
    for label, (name, colour) in RUNS.items():
        run = api.run(f"open-athena/MarinFold/{name}")
        history = run.history(keys=["train/loss"], pandas=True, samples=20000).dropna(subset=["train/loss"])
        window = history[(history["_step"] >= arguments.low) & (history["_step"] <= arguments.high)]
        if window.empty:
            continue
        binned = window.groupby((window["_step"] // BIN) * BIN)["train/loss"].mean()
        axes.plot(binned.index, binned.values, color=colour, linewidth=1.8, label=label)
        # The transition is the step with the largest fall between the 1,500
        # steps before it and the 1,500 after. A plain diff of a short rolling
        # mean is not enough: exp232 has a loss spike near 13k whose *recovery*
        # is a steeper single-bin drop than its real transition.
        span = 3
        best_step, best_drop = None, 0.0
        values = binned.to_list()
        steps = list(binned.index)
        for i in range(span, len(values) - span):
            before = sum(values[i - span:i]) / span
            after = sum(values[i:i + span]) / span
            if after - before < best_drop:
                best_drop, best_step = after - before, steps[i]
        step = int(best_step)
        axes.axvline(step, color=colour, linestyle=":", linewidth=1.4)
        axes.annotate(f"{label.split()[0]}\n~{step // 1000}k ({best_drop:+.03f})",
                      (step, next(heights)), color=colour, fontsize=8, ha="center")

    axes.set_xlabel("step (of 145,200)")
    axes.set_ylabel(f"train loss, {BIN}-step bins")
    axes.set_title("the same transition, at different steps in every run\n"
                   "two of these are the SAME architecture on the same data and seeds")
    axes.legend(fontsize=9, loc="upper right")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    save_plot_with_meta(
        figure, arguments.plots_dir / "loss_transition.png",
        caption=(
            "Each 1.5B run drops ~0.09 nats over ~1,500 steps somewhere mid-training. exp262's "
            "control transitions near 15.5k, its NoPE arm near 21k, exp232's reference near 21.5k "
            "— control and exp232 share architecture, data and seed. The timing is "
            "not reproducible, and its ~6,000-step spread is worth far more loss at a fixed step "
            "than the architecture difference under test."
        ),
        dpi=150,
    )
    print(f"wrote {arguments.plots_dir / 'loss_transition.png'}")


if __name__ == "__main__":
    main()

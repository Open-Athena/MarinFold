# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the two full-budget exp262 runs against exp232's reference curve.

Three things on one page: whether the control is reproducing the usual setup
(it underwrites everything else), the eval-loss gap between the arms, and the
smoothed train-loss gap, which has ten times the resolution of the eval points
while the run is young.
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

RUNS = {
    "control": ("prot-exp262-cw-cv1-arch-full-control-p06", "#4a5568"),
    "nope-smear": ("prot-exp262-cw-cv1-arch-full-nope-smear-p06", "#dd6b20"),
    "exp232 reference": ("prot-exp232-cw-cv1-decontam-s02-m2-p06-aug", "#a0aec0"),
}
GENERATION_NATS = 0.053  # the whole #75 -> #117 model generation, for scale


def fetch(entity_project: str = "open-athena/MarinFold") -> dict:
    api = wandb.Api()
    out = {}
    for label, (name, colour) in RUNS.items():
        run = api.run(f"{entity_project}/{name}")
        train = run.history(keys=["train/loss"], pandas=True, samples=6000).dropna(subset=["train/loss"])
        evaluation = run.history(keys=["eval/loss"], pandas=True, samples=2000).dropna(subset=["eval/loss"])
        out[label] = {
            "colour": colour,
            "train": train.set_index("_step")["train/loss"].sort_index(),
            "eval": evaluation.set_index("_step")["eval/loss"].sort_index(),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    arguments = parser.parse_args()
    data = fetch()

    figure, (curves, gap) = plt.subplots(1, 2, figsize=(13, 4.8))
    for label, payload in data.items():
        smoothed = payload["train"].rolling(50, min_periods=10).mean()
        curves.plot(smoothed.index, smoothed.values, color=payload["colour"], linewidth=1.1,
                    alpha=0.55 if label == "exp232 reference" else 1.0, label=f"{label} (train)")
        if len(payload["eval"]):
            curves.plot(payload["eval"].index, payload["eval"].values, "o--", color=payload["colour"],
                        markersize=4, linewidth=1.0, alpha=0.9, label=f"{label} (eval)")
    # Clip the initial plunge; every difference worth seeing is below 3.6.
    curves.set_ylim(2.8, 3.7)
    curves.set_xlabel("step (of 145,200)")
    curves.set_ylabel("loss (nats/token)")
    curves.set_title("exp262 full-budget runs, 1.5B on decontaminated contacts-v1")
    curves.legend(fontsize=7)
    curves.grid(alpha=0.3)

    control, proposal = data["control"], data["nope-smear"]
    common = control["train"].index.intersection(proposal["train"].index)
    delta = (proposal["train"].loc[common] - control["train"].loc[common]).rolling(200, min_periods=50).mean()
    gap.plot(delta.index, delta.values, color="#2b6cb0", linewidth=1.3, label="train (200-step mean)")
    shared_eval = control["eval"].index.intersection(proposal["eval"].index)
    if len(shared_eval):
        gap.plot(shared_eval, (proposal["eval"].loc[shared_eval] - control["eval"].loc[shared_eval]).values,
                 "o-", color="#c53030", markersize=7, label="eval")
    gap.axhline(0, color="black", linewidth=0.9)
    gap.axhspan(-GENERATION_NATS, GENERATION_NATS, color="gray", alpha=0.18,
                label=f"±{GENERATION_NATS} nats = one model generation")
    gap.set_xlabel("step (of 145,200)")
    gap.set_ylabel("Δ loss, NoPE+smear − control")
    gap.set_title("below zero = NoPE + smear is winning")
    gap.legend(fontsize=8)
    gap.grid(alpha=0.3)

    figure.tight_layout()
    save_plot_with_meta(
        figure, arguments.plots_dir / "full_run_progress.png",
        caption=(
            "Left: both full-budget arms with exp232's run behind them in grey — the control is "
            "tracking it to ~0.01 nats, which is what licenses reading the gap at all. Right: the "
            "arm gap, against the 0.053 nats the whole #75 to #117 generation was worth. The early "
            "positive excursion is LR warmup; the crossover is near step 7,000."
        ),
        dpi=150,
    )
    print(f"wrote {arguments.plots_dir / 'full_run_progress.png'}")


if __name__ == "__main__":
    main()

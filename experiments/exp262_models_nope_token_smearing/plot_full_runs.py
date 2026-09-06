# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the two full-budget exp262 runs against exp232's completed reference.

exp232's ``s02-m2-p06-aug`` ran this exact recipe to completion, so it works as
a **common baseline**: each arm is plotted as its difference from exp232 at the
same step. That does two things at once. It shows whether the control is
reproducing the usual setup — the thing that licenses reading any of this — and
it lets the two arms be compared even while they sit at different points in the
schedule, which they now do, because the control lost ~11 hours to a preemption
that took its whole gang down.

A direct arm-vs-arm panel is kept for the step range where both have data, since
that is the comparison of record; the exp232-relative view is what extends past it.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

REFERENCE = "exp232 s02-m2-p06-aug"
RUNS = {
    "control (RoPE)": ("prot-exp262-cw-cv1-arch-full-control-p06", "#2b6cb0"),
    "NoPE + smear": ("prot-exp262-cw-cv1-arch-full-nope-smear-p06", "#dd6b20"),
    REFERENCE: ("prot-exp232-cw-cv1-decontam-s02-m2-p06-aug", "#a0aec0"),
}
GENERATION_NATS = 0.053  # the whole #75 -> #117 model generation, for scale


def fetch(entity_project: str = "open-athena/MarinFold") -> dict:
    api = wandb.Api()
    out = {}
    for label, (name, colour) in RUNS.items():
        run = api.run(f"{entity_project}/{name}")
        train = run.history(keys=["train/loss"], pandas=True, samples=8000).dropna(subset=["train/loss"])
        evaluation = run.history(keys=["eval/loss"], pandas=True, samples=2000).dropna(subset=["eval/loss"])
        out[label] = {
            "colour": colour,
            "train": train.set_index("_step")["train/loss"].sort_index(),
            "eval": evaluation.set_index("_step")["eval/loss"].sort_index(),
        }
    return out


BIN = 500


def binned(series, bin_size: int = BIN):
    """Mean loss per ``bin_size``-step bin, keyed by bin centre.

    Both series must be binned before differencing. Per-step training loss is
    dominated by batch-to-batch noise, and subtracting two raw noisy series
    doubles it — the first version of this plot was unreadable for that reason.
    """
    if series.empty:
        return series
    grouped = series.groupby((series.index // bin_size) * bin_size + bin_size // 2).mean()
    return grouped[grouped.index >= 0]


def minus_reference(series, reference) -> tuple[np.ndarray, np.ndarray]:
    """``series`` minus exp232, both binned onto the same grid first."""
    left, right = binned(series), binned(reference)
    shared = left.index.intersection(right.index)
    return shared.to_numpy(dtype=float), (left.loc[shared] - right.loc[shared]).to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    arguments = parser.parse_args()
    data = fetch()
    reference_train = data[REFERENCE]["train"]
    reference_eval = data[REFERENCE]["eval"]

    figure, (curves, direct, versus) = plt.subplots(
        1, 3, figsize=(17.5, 4.9), width_ratios=[1.0, 1.25, 1.0]
    )

    for label, payload in data.items():
        smoothed = payload["train"].rolling(60, min_periods=15).mean()
        curves.plot(smoothed.index, smoothed.values, color=payload["colour"], linewidth=1.1,
                    alpha=0.6 if label == REFERENCE else 1.0, label=f"{label} — train")
        if len(payload["eval"]):
            curves.plot(payload["eval"].index, payload["eval"].values, "o--", color=payload["colour"],
                        markersize=3.5, linewidth=0.9, alpha=0.9, label=f"{label} — eval")
    curves.set_ylim(2.8, 3.6)
    curves.set_xlabel("step (of 145,200)")
    curves.set_ylabel("loss (nats/token)")
    curves.set_title("all three runs, same recipe")
    curves.legend(fontsize=6.5, loc="upper right")
    curves.grid(alpha=0.3)

    for label in ("control (RoPE)", "NoPE + smear"):
        payload = data[label]
        steps, delta = minus_reference(payload["train"], reference_train)
        versus.plot(steps, delta, color=payload["colour"], linewidth=1.2, label=f"{label} — train")
        if len(payload["eval"]):
            e_steps, e_delta = minus_reference(payload["eval"], reference_eval)
            versus.plot(e_steps, e_delta, "o", color=payload["colour"], markersize=7,
                        markeredgecolor="black", markeredgewidth=0.6, label=f"{label} — eval")
    # Every run in this comparison has a step-change in its loss somewhere in
    # 14k-24k, and they do not line up: exp232's is near 22k, the control's near
    # 15k, the NoPE arm's near 21k. Differencing across that offset measures
    # WHEN each run hit the feature, not how well it is training, so the
    # exp232-relative view is not to be read inside this band.
    versus.axvspan(14000, 24000, color="#c53030", alpha=0.10)
    versus.annotate("step-change region:\nexp232-relative\ncomparison unreliable",
                    (19000, -0.14), fontsize=6.5, ha="center", color="#c53030")
    versus.axhline(0, color="black", linewidth=1.0)
    versus.axhspan(-GENERATION_NATS, GENERATION_NATS, color="gray", alpha=0.18,
                   label=f"±{GENERATION_NATS} = one model generation")
    versus.set_ylim(-0.16, 0.16)
    versus.set_xlabel("step (of 145,200)")
    versus.set_ylabel(f"Δ loss vs {REFERENCE}")
    versus.set_title("each arm against exp232 at the same step\n(control ≈ 0 means the setup reproduces)")
    versus.legend(fontsize=6.5, loc="upper right")
    versus.grid(alpha=0.3)

    control, proposal = data["control (RoPE)"], data["NoPE + smear"]
    # Deliberately eval-only. The arms' train-loss difference is contaminated by
    # the step-change timing offset (the control's is near 15k, the NoPE arm's
    # near 21k), so plotting it beside the matched evals invites reading a
    # scheduling artifact as an architecture effect.
    matched = control["eval"].index.intersection(proposal["eval"].index)
    deltas = (proposal["eval"].loc[matched] - control["eval"].loc[matched])
    if len(matched):
        direct.plot(matched, deltas.values, "o-", color="#c53030", markersize=10,
                    linewidth=2.0, zorder=3, label="eval (the comparison of record)")
        # Label sparsely: with a full run's worth of evals every point labelled
        # is an unreadable pile.
        marks = list(zip(matched, deltas.values))
        for index, (step, value) in enumerate(marks):
            if index % 4 and index != len(marks) - 1:
                continue
            direct.annotate(f"{value:+.4f}", (step, value), textcoords="offset points",
                            xytext=(0, 14 if value < 0 else -20), ha="center", fontsize=8.5)
        mean = float(deltas.mean())
        direct.axhline(mean, color="#c53030", linestyle="--", linewidth=1.2,
                       label=f"mean of {len(deltas)} matched evals = {mean:+.4f}")
    direct.axhline(0, color="black", linewidth=1.2)
    direct.axhspan(-GENERATION_NATS, GENERATION_NATS, color="gray", alpha=0.18,
                   label=f"±{GENERATION_NATS} = one model generation")
    direct.set_ylim(-0.09, 0.09)
    direct.set_xlabel("step (of 145,200)")
    direct.set_ylabel("Δ loss, NoPE+smear − control")
    direct.set_title("head to head at matched steps\nbelow zero = NoPE + smear winning")
    direct.legend(fontsize=7, loc="lower right")
    direct.grid(alpha=0.3)

    figure.tight_layout()
    save_plot_with_meta(
        figure, arguments.plots_dir / "full_run_progress.png",
        caption=(
            "exp232's completed run is the common baseline. Middle panel: each arm minus exp232 at "
            "the same step — the control sits on zero, which is what shows the newer marin pin did "
            "not move the loss scale, and it is why the NoPE curve below zero can be read as "
            "architecture. Right: head to head where both have data; the control was preempted at "
            "step 21,535 and is catching up. Grey band is the 0.053 nats the #75 to #117 generation "
            "was worth. Middle is the comparison of record — both arms scored at identical "
            "steps on identical data; the small-scale pilot predicted about -0.16 and the mean "
            "of the matched evals is nowhere near it. Right: the red band is where all three "
            "runs have a step-change at different steps, so differencing across it measures "
            "timing rather than quality."
        ),
        dpi=150,
    )
    print(f"wrote {arguments.plots_dir / 'full_run_progress.png'}")


if __name__ == "__main__":
    main()

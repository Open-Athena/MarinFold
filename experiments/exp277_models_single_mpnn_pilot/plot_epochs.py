"""Build exp277's second-epoch figures from saved results.

Run: uv run --no-project --with pandas --with matplotlib python plot_epochs.py
Reads only committed tables; no predictor or training run is performed here.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EPOCHS = DATA / "eval_rollout_v2_epochs"
PLOTS = HERE / "plots"

EPOCH1 = "#b83c54"
EPOCH2 = "#2a6f97"
TIE = 0.005
RESTORE_STEP = 213_072
EPOCH1_COOLDOWN = 213_076
EPOCH2_COOLDOWN = 213_073 + int(0.8 * 266_345)
SEED = 277
DRAWS = 10_000
SETS = (
    ("legacy_554", "legacy 554"),
    ("eval-val", "eval-val (97)"),
    ("eval-denovo", "eval-denovo (19)"),
)


def save(fig: plt.Figure, name: str, caption: str) -> None:
    """Write raster/vector figures and their reproduction sidecars."""
    save_plot_with_meta(
        fig,
        PLOTS / f"{name}.png",
        caption=caption,
        script="plot_epochs.py",
        args=[],
        dpi=180,
    )
    svg = PLOTS / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text("".join(l.rstrip() + "\n" for l in svg.read_text().splitlines()))
    plt.close(fig)


def validation_figure() -> None:
    """Both runs on one absolute-step axis, with the tail magnified.

    The second epoch restores full trainer state at step 213,072, so the two
    curves share every step before that. Plotting them on absolute steps is
    what makes the comparison legible: the continuation's plateau sits a hair
    below the first epoch's, and both cooldowns fall to nearly the same place.
    """
    one = pd.read_csv(DATA / "epoch_validation_progress.csv")
    two = pd.read_csv(DATA / "epoch2_validation_progress.csv")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.4),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.55, 1]},
    )

    ax = axes[0]
    ax.plot(
        one.global_step / 1000,
        one.validation_loss,
        color=EPOCH1,
        lw=1.6,
        label="Epoch 1 (from scratch)",
    )
    ax.plot(
        two.global_step / 1000,
        two.validation_loss,
        color=EPOCH2,
        lw=1.6,
        label="Epoch 2 (restored at step 213,072, reshuffled)",
    )
    ax.axvline(RESTORE_STEP / 1000, color="#555", ls=":", lw=1.2)
    ax.annotate(
        "restore point",
        xy=(RESTORE_STEP / 1000, 3.62),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color="#555",
    )
    ax.set(
        xlabel="Absolute optimizer step (thousands)",
        ylabel="LM validation loss",
        title="Validation loss, both epochs",
    )
    ax.legend(fontsize=9, frameon=False)

    ax = axes[1]
    for frame, colour, label in ((one, EPOCH1, "Epoch 1"), (two, EPOCH2, "Epoch 2")):
        tail = frame[frame.global_step >= RESTORE_STEP]
        ax.plot(
            tail.global_step / 1000,
            tail.validation_loss,
            color=colour,
            lw=1.6,
            label=label,
        )
    # Label the cooldown onsets along the bottom, where the panel is empty; a
    # rotated label at curve height collides with both the plateau and the legend.
    for step, colour, text, align in (
        (EPOCH1_COOLDOWN, EPOCH1, "epoch 1 cooldown", "left"),
        (EPOCH2_COOLDOWN, EPOCH2, "epoch 2 cooldown", "right"),
    ):
        ax.axvline(step / 1000, color=colour, ls="--", lw=1.1, alpha=0.7)
        ax.annotate(
            text,
            xy=(step / 1000, 2.9605),
            xytext=(4 if align == "left" else -4, 0),
            textcoords="offset points",
            fontsize=8,
            color=colour,
            ha=align,
        )
    best1 = one.loc[one.validation_loss.idxmin()]
    best2 = two.loc[two.validation_loss.idxmin()]
    for row, colour in ((best1, EPOCH1), (best2, EPOCH2)):
        ax.plot(row.global_step / 1000, row.validation_loss, "o", color=colour, ms=5)
        ax.annotate(
            f"{row.validation_loss:.5f}",
            xy=(row.global_step / 1000, row.validation_loss),
            xytext=(-4, -14),
            textcoords="offset points",
            fontsize=9,
            color=colour,
            ha="right",
        )
    ax.set(
        xlabel="Absolute optimizer step (thousands)",
        ylabel="LM validation loss",
        title="From the restore point onward",  # y-range covers the 3.0886 spike
        ylim=(2.955, 3.10),
    )
    ax.legend(fontsize=9, frameon=False, loc="center left")
    save(
        fig,
        "epoch2_validation_loss",
        "LM validation loss for both exp277 epochs on absolute steps. "
        f"Best {best1.validation_loss:.5f} (epoch 1) vs {best2.validation_loss:.5f} "
        "(epoch 2): a second full epoch is worth 0.00175 nats.",
    )


def contact_figures() -> None:
    """Paired contact-accuracy effect of the second epoch."""
    headline = pd.read_csv(EPOCHS / "epoch_comparison.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    ax.axvspan(-TIE, TIE, color="#bbb", alpha=0.25, lw=0)
    ax.axvline(0, color="#444", lw=1)
    labels, offsets = [], []
    for i, (key, pretty) in enumerate(SETS):
        for j, (rng, marker) in enumerate((("all", "o"), ("long", "s"))):
            row = headline[(headline.subset == key) & (headline["range"] == rng)].iloc[
                0
            ]
            y = -(i * 2.4 + j * 0.9)
            ax.errorbar(
                row.delta,
                y,
                xerr=[[row.delta - row.ci_low], [row.ci_high - row.delta]],
                fmt=marker,
                color=EPOCH2,
                ms=6,
                capsize=3,
                lw=1.5,
            )
            labels.append(f"{pretty}\n{rng}-range" if j == 0 else f"{rng}-range")
            offsets.append(y)
    ax.set_yticks(offsets, labels, fontsize=9)
    ax.set(
        xlabel="Paired R-precision change, epoch 2 − epoch 1",
        title="Effect of the second epoch on contact accuracy",
    )
    # Label the band inside the axes; an annotation below the lowest row falls
    # outside the figure and silently disappears.
    ax.set_ylim(min(offsets) - 1.1, max(offsets) + 1.5)
    ax.annotate(
        "tie band (±0.005)",
        xy=(TIE, max(offsets) + 1.0),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=8.5,
        color="#666",
        ha="left",
        va="center",
    )
    save(
        fig,
        "epoch2_contact_delta",
        "Paired per-protein R-precision change, second epoch minus first, with "
        "95% bootstrap intervals (seed 277, 10,000 resamples). Both epochs were "
        "scored in one job. Natural-protein deltas sit inside the tie band.",
    )

    precision = pd.read_csv(EPOCHS / "subset_aggregate_metrics.csv")
    sub = precision[
        precision.cut.eq("R")
        & precision["range"].eq("all")
        & precision.subset.isin([k for k, _ in SETS])
    ]
    wide = sub.pivot_table(index="subset", columns="model", values="precision")
    fig, ax = plt.subplots(figsize=(7.4, 4.0), layout="constrained")
    x = np.arange(len(SETS))
    e1 = [
        wide.loc[k][[c for c in wide.columns if "epoch2" not in c][0]] for k, _ in SETS
    ]
    e2 = [wide.loc[k][[c for c in wide.columns if "epoch2" in c][0]] for k, _ in SETS]
    ax.bar(x - 0.19, e1, 0.36, label="Epoch 1", color=EPOCH1)
    ax.bar(x + 0.19, e2, 0.36, label="Epoch 2", color=EPOCH2)
    for xi, (a, b) in enumerate(zip(e1, e2)):
        for dx, v in ((-0.19, a), (0.19, b)):
            ax.annotate(
                f"{v:.3f}",
                xy=(xi + dx, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x, [p for _, p in SETS])
    ax.set(
        ylabel="R-precision (all ranges)",
        ylim=(0, 0.83),
        title="Contact accuracy by evaluation set",
    )
    ax.legend(fontsize=9, frameon=False)
    save(
        fig,
        "epoch2_contact_by_set",
        "All-range R-precision for both exp277 epochs, scored together in one "
        "job over 670 units. eval-test was not read.",
    )


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    validation_figure()
    contact_figures()
    sources = [
        DATA / "epoch_validation_progress.csv",
        DATA / "epoch2_validation_progress.csv",
        EPOCHS / "epoch_comparison.csv",
        EPOCHS / "subset_aggregate_metrics.csv",
    ]
    (EPOCHS / "figure_provenance.json").write_text(
        json.dumps(
            {
                "sources": {
                    str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources
                },
                "bootstrap_seed": SEED,
                "bootstrap_replicates": DRAWS,
                "eval_job": "/bizon/exp277-eval-v2-02",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

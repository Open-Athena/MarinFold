# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 8 -- two figures: the scoreboard, and what eval-val was over-reporting.

``eval_sets_scoreboard.png``
    All-range R-precision for every predictor on the three sets, with 95 %
    bootstrap intervals over proteins. The three MarinFold checkpoints are
    coloured; baselines are grey, so the comparison a reader makes first is the
    one that matters.

``val_vs_test.png``
    Each predictor's eval-val score joined to its eval-test score. A baseline's
    slope is the sample difference between the two protein sets; a
    decontaminated checkpoint's should look like a baseline's; the contaminated
    reference's excess slope over them is the contamination estimate.

    uv run python plot_results.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import upstream as U  # noqa: E402

DATA = U.DATA
PLOTS = U.HERE / "plots"

CHECKPOINT_COLORS = {
    "#232 m2-p06 (decontaminated)": "#d55e00",
    "#232 m1-p02 (decontaminated)": "#e69f00",
    "#199 cooldown (contaminated)": "#0072b2",
}
BASELINE_COLOR = "#8f8b86"
SETS = ("eval-val", "eval-test", "eval-denovo")
SET_LABELS = {
    "eval-val": "eval-val\n(97 natural, seen before)",
    "eval-test": "eval-test\n(217 natural, held out)",
    "eval-denovo": "eval-denovo\n(19 designs)",
}
BOOTSTRAP_DRAWS = 4_000
SEED = 245


def interval(values: np.ndarray) -> tuple[float, float]:
    """95 % bootstrap interval of the mean."""
    generator = np.random.default_rng(SEED)
    index = generator.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stamp(path: Path, sources: dict[str, Path]) -> None:
    """Write the sidecar every committed plot in this repo carries."""
    meta = {
        "plot": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": {
            name: {"path": str(source.relative_to(U.REPO)),
                   "sha256": U.sha256(source)}
            for name, source in sources.items()
        },
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")


def load() -> tuple[pd.DataFrame, list[str]]:
    per_protein = pd.read_csv(DATA / "per_protein.csv.gz")
    sets = pd.read_csv(DATA / "eval_sets.csv")
    sets = sets[sets.scorable == 1]
    frame = per_protein[(per_protein["range"] == "all") & (per_protein["cut"] == "R")]
    frame = frame.merge(sets[["stem", "eval_set", "is_viral"]], on="stem")
    order = [p for p in CHECKPOINT_COLORS if p in set(frame.predictor)]
    order += sorted(set(frame.predictor) - set(order),
                    key=lambda p: -frame.loc[frame.predictor == p, "precision"].mean())
    return frame, order


def scoreboard(frame: pd.DataFrame, order: list[str], out: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
    for axis, eval_set in zip(axes, SETS, strict=True):
        subset = frame[frame.eval_set == eval_set]
        positions, heights, lows, highs, colors = [], [], [], [], []
        for index, predictor in enumerate(order):
            values = subset.loc[subset.predictor == predictor, "precision"].to_numpy()
            if not len(values):
                continue
            low, high = interval(values)
            positions.append(index)
            heights.append(values.mean())
            lows.append(values.mean() - low)
            highs.append(high - values.mean())
            colors.append(CHECKPOINT_COLORS.get(predictor, BASELINE_COLOR))
        axis.bar(positions, heights, color=colors, width=0.72)
        axis.errorbar(positions, heights, yerr=[lows, highs], fmt="none",
                      ecolor="#33312e", elinewidth=1.1, capsize=3)
        for position, height in zip(positions, heights, strict=True):
            axis.text(position, height + 0.012, f"{height:.3f}", ha="center",
                      fontsize=8.5, color="#33312e")
        axis.set_title(SET_LABELS[eval_set], fontsize=10.5)
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels(order, rotation=40, ha="right", fontsize=8.5)
        axis.grid(axis="y", color="#dddad6", linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("R-precision (all ranges)")
    axes[0].set_ylim(0, max(0.9, frame.groupby(["eval_set", "predictor"])
                            .precision.mean().max() * 1.18))
    figure.suptitle(
        "Contact R-precision on FoldBench's monomers, split into what we had "
        "already scored and what we had not", fontsize=12.5)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out, dpi=200)
    plt.close(figure)


def val_versus_test(frame: pd.DataFrame, order: list[str], out: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.6))
    for predictor in order:
        subset = frame[frame.predictor == predictor]
        val = subset.loc[subset.eval_set == "eval-val", "precision"].mean()
        test = subset.loc[subset.eval_set == "eval-test", "precision"].mean()
        color = CHECKPOINT_COLORS.get(predictor, BASELINE_COLOR)
        width = 2.4 if predictor in CHECKPOINT_COLORS else 1.4
        axis.plot([0, 1], [val, test], color=color, linewidth=width,
                  marker="o", markersize=5.5, zorder=3 if color != BASELINE_COLOR else 2)
        axis.annotate(f"{predictor}  {test - val:+.3f}", (1, test),
                      textcoords="offset points", xytext=(8, 0), va="center",
                      fontsize=9, color=color)
    axis.set_xlim(-0.08, 1.75)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["eval-val\n(97 natural, scored before)",
                          "eval-test\n(217 natural, never scored)"])
    axis.set_ylabel("Mean R-precision (all ranges)")
    axis.set_title("Every predictor scores lower on the held-out monomers.\n"
                   "The question is whether the contaminated model falls further.",
                   fontsize=11.5)
    axis.grid(axis="y", color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    frame, order = load()

    sources = {"per_protein": DATA / "per_protein.csv.gz",
               "eval_sets": DATA / "eval_sets.csv"}
    board = PLOTS / "eval_sets_scoreboard.png"
    scoreboard(frame, order, board)
    stamp(board, sources)
    gap = PLOTS / "val_vs_test.png"
    val_versus_test(frame, order, gap)
    stamp(gap, sources)
    print(f"[plots] -> {board}\n[plots] -> {gap}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

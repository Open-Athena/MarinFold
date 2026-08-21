# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 -- the eval-val figure.

Left panel: all-range R-precision for every predictor on eval-val's 97 natural
FoldBench monomers -- this experiment's four MarinFold readouts and the pairwise
readout in colour, #245's published predictors in grey. The two oracle
best-of-100 bars are hatched, because they are headroom diagnostics that need
the ground truth to pick a rollout and are not recipes anyone can run.

Middle panel: the paired per-protein differences the experiment preregistered --
``seeded - iid`` for consensus and for oracle best-of-100. Points are means with
95 % bootstrap intervals over proteins; the shaded band is #204's 0.005 tie
threshold, which is the only reason a small difference here is readable at all.

Right panel: the decomposition that says *why*. The rollout index is the
pairwise rank of the seed it was given, so seed accuracy can be read against
rollout quality along the same axis -- accuracy falls 33 points from the top of
the ranking to the bottom and rollout R-precision does not move.

    uv run python plot.py --data data --out plots
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import EVAL_SETS_CSV, EXP245_DATA  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BASELINE_PER_PROTEIN = EXP245_DATA / "per_protein.csv.gz"

ARM_COLORS = {
    "iid consensus": "#0072b2",
    "seeded consensus": "#d55e00",
    "seeded consensus (seed vote removed)": "#e69f00",
    "iid oracle best-of-N": "#56b4e9",
    "seeded oracle best-of-N": "#cc79a7",
    "pairwise": "#009e73",
}
ORACLE_ARMS = ("iid oracle best-of-N", "seeded oracle best-of-N")
BASELINE_COLOR = "#8f8b86"
#: #204's four evaluations of one unchanged checkpoint span this much.
TIE_THRESHOLD = 0.005
BOOTSTRAP_DRAWS = 10_000
SEED = 254


def interval(values: np.ndarray) -> tuple[float, float]:
    """95 % bootstrap interval of the mean."""
    generator = np.random.default_rng(SEED)
    index = generator.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stamp(path: Path, sources: dict[str, Path], caption: str) -> None:
    """Provenance sidecar: which script, which inputs, which bytes."""
    meta = {
        "script": Path(sys.argv[0]).name,
        "args": sys.argv[1:],
        "caption": caption,
        "plot": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": {
            name: {
                "path": str(source.resolve().relative_to(REPO)),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            for name, source in sources.items()
        },
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")


def load(data_dir: Path) -> pd.DataFrame:
    """This experiment's arms plus #245's baselines, restricted to eval-val."""
    mine = pd.read_csv(data_dir / "exp254_per_protein.csv.gz")
    mine = mine[(mine["range"] == "all") & (mine["cut"] == "R")]
    mine = mine[["stem", "predictor", "precision"]]

    sets = pd.read_csv(EVAL_SETS_CSV, usecols=["stem", "eval_set", "scorable"])
    val_stems = set(sets[(sets.eval_set == "eval-val") & (sets.scorable == 1)].stem)

    baselines = pd.read_csv(BASELINE_PER_PROTEIN)
    baselines = baselines[(baselines["range"] == "all") & (baselines["cut"] == "R")
                          & baselines["stem"].isin(val_stems)]
    baselines = baselines[["stem", "predictor", "precision"]]
    # #245 already publishes m2-p06 under the exp82 recipe; keeping it in the
    # figure makes the sanity gate visible -- the `iid consensus` bar beside it
    # is this run's reproduction of that number.
    frame = pd.concat([mine, baselines], ignore_index=True)
    assert set(frame[frame.predictor == "iid consensus"].stem) == val_stems, (
        "the iid arm does not cover exactly eval-val"
    )
    return frame


def scoreboard(axis, frame: pd.DataFrame) -> None:
    means = frame.groupby("predictor")["precision"].mean().sort_values()
    for position, predictor in enumerate(means.index):
        values = frame.loc[frame.predictor == predictor, "precision"].to_numpy()
        low, high = interval(values)
        mean = values.mean()
        color = ARM_COLORS.get(predictor, BASELINE_COLOR)
        axis.barh(position, mean, color=color, height=0.72,
                  hatch="///" if predictor in ORACLE_ARMS else None,
                  edgecolor="white" if predictor in ORACLE_ARMS else color)
        axis.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt="none",
                      ecolor="#33312e", elinewidth=1.1, capsize=3)
        axis.text(high + 0.012, position, f"{mean:.3f}", va="center",
                  fontsize=8.5, color="#33312e")
    axis.set_yticks(range(len(means)))
    axis.set_yticklabels(means.index, fontsize=8.5)
    axis.set_xlabel("R-precision (all ranges), mean over 97 proteins")
    axis.set_xlim(0, min(1.0, means.max() * 1.25))
    axis.grid(axis="x", color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_title("eval-val (97 natural FoldBench monomers), #232 m2-p06\n"
                   "hatched = oracle, needs the answer to pick a rollout",
                   fontsize=10.5)


def deltas(axis, frame: pd.DataFrame) -> None:
    pairs = [
        ("seeded consensus", "iid consensus", "consensus\nseeded - i.i.d."),
        ("seeded consensus (seed vote removed)", "iid consensus",
         "consensus, seed vote removed\nseeded - i.i.d."),
        ("seeded oracle best-of-N", "iid oracle best-of-N",
         "oracle best-of-100\nseeded - i.i.d."),
    ]
    wide = frame.pivot_table(index="stem", columns="predictor", values="precision")
    axis.axvspan(-TIE_THRESHOLD, TIE_THRESHOLD, color="#dddad6", alpha=0.55, zorder=0)
    axis.axvline(0, color="#33312e", linewidth=0.9, zorder=1)
    # A handful of proteins swing far enough to flatten the tie band out of
    # existence if the axis is left to autoscale. The axis is clipped instead
    # and the clipped points are drawn on the edge as carets and counted, so
    # nothing is silently dropped from the figure.
    limit = 0.12
    labels, n_clipped = [], 0
    jitter = np.random.default_rng(SEED)
    for position, (a, b, label) in enumerate(pairs):
        d = (wide[a] - wide[b]).dropna().to_numpy()
        low, high = interval(d)
        mean = d.mean()
        offsets = jitter.uniform(-0.16, 0.16, len(d))
        inside = np.abs(d) <= limit
        n_clipped += int((~inside).sum())
        axis.scatter(d[inside], position + offsets[inside], s=9,
                     color=ARM_COLORS[a], alpha=0.35, zorder=2)
        axis.scatter(np.clip(d[~inside], -limit, limit), position + offsets[~inside],
                     s=26, marker="|", color=ARM_COLORS[a], alpha=0.9, zorder=2)
        axis.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt="o",
                      color="#33312e", markersize=6, elinewidth=1.6, capsize=4,
                      zorder=3)
        axis.text(mean, position + 0.32, f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]",
                  ha="center", fontsize=8.5, color="#33312e")
        labels.append(label)
    axis.set_yticks(range(len(pairs)))
    axis.set_yticklabels(labels, fontsize=8.5)
    axis.set_ylim(-0.7, len(pairs) - 0.15)
    axis.set_xlim(-limit, limit)
    axis.set_xlabel(f"Per-protein difference in R-precision (all ranges)\n"
                    f"{n_clipped} points beyond ±{limit:g} drawn as ticks on the edge")
    axis.grid(axis="x", color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.set_title("Paired per-protein differences\n"
                   "shaded band = the 0.005 tie threshold (#204)", fontsize=10.5)


def conditioning_panel(axis, by_rank: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Seed accuracy falls steeply down the pairwise ranking; the rollouts do not.

    The rollout index *is* the pairwise rank of the seed it was handed, so this
    is a free dose-response curve. Seeds drawn from ranks 1-10 are true contacts
    79 % of the time and seeds from ranks 71-100 only 46 %, a 33-point swing --
    and the R-precision of the rollouts they produced is flat across the whole
    range, within 0.004 of each other and of the unseeded arm.

    A pooled true-seed-versus-false-seed split is deliberately NOT plotted here.
    It reads +0.18, almost all of which is protein difficulty: a protein the
    model handles well supplies both more correct seeds and better rollouts. The
    within-protein contrast, quoted in the corner, is the honest number.
    """
    positions = np.arange(len(by_rank))
    axis.bar(positions, by_rank["seed_accuracy"], color="#009e73", width=0.62,
             label="seed is a true contact")
    for position, value in zip(positions, by_rank["seed_accuracy"]):
        axis.text(position, value + 0.012, f"{value:.2f}", ha="center", fontsize=8,
                  color="#33312e")
    axis.plot(positions, by_rank["rollout_precision"], color="#d55e00", marker="o",
              markersize=6, linewidth=2, label="R-precision of those rollouts")
    axis.axhline(by_rank["iid_rollout_precision"].iloc[0], color="#0072b2",
                 linewidth=1.6, linestyle="--", label="unseeded rollout")
    axis.set_xticks(positions)
    axis.set_xticklabels(by_rank["seed_rank_bucket"], fontsize=8.5)
    quantities = summary.set_index("quantity")["value"]
    axis.set_xlabel(
        "pairwise rank of the seed handed to the rollout\n"
        "within a protein, true seed - false seed = "
        f"{quantities['within-protein difference (true - false)']:+.3f}")
    axis.set_ylim(0, 1.0)
    axis.legend(fontsize=8, loc="upper right", frameon=False)
    axis.grid(axis="y", color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_title("The seed barely moves the rollout\n"
                   "accuracy swings 33 points; quality does not follow",
                   fontsize=10.5)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("plots"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frame = load(args.data)
    by_rank = pd.read_csv(args.data / "exp254_seed_rank.csv")
    conditioning_summary = pd.read_csv(args.data / "exp254_seed_conditioning_summary.csv")

    figure, axes = plt.subplots(1, 3, figsize=(19.5, 6.4),
                                gridspec_kw={"width_ratios": [1.45, 1.1, 0.95]})
    scoreboard(axes[0], frame)
    deltas(axes[1], frame)
    conditioning_panel(axes[2], by_rank, conditioning_summary)
    figure.suptitle(
        "Seeding each rollout with a top-ranked pairwise contact, versus i.i.d. "
        "sampling", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    dest = args.out / "seeded_vs_iid_eval_val.png"
    figure.savefig(dest, dpi=200)
    plt.close(figure)
    stamp(dest, {"exp254_per_protein": args.data / "exp254_per_protein.csv.gz",
                 "exp254_seed_rank": args.data / "exp254_seed_rank.csv",
                 "exp254_seed_conditioning_summary":
                     args.data / "exp254_seed_conditioning_summary.csv",
                 "exp245_per_protein": BASELINE_PER_PROTEIN,
                 "eval_sets": EVAL_SETS_CSV},
          "All-range contact R-precision on eval-val for the #232 m2-p06 "
          "checkpoint under five MarinFold readouts and #245's published "
          "predictors, the paired seeded-minus-i.i.d. differences, and seed "
          "accuracy against rollout quality along the pairwise ranking.")
    print(f"[plot] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

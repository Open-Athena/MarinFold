# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 8b -- the scoreboard again, split viral vs non-viral.

#241 found that viral eval proteins and non-viral ones rank predictors
differently, so every headline here carries the split. This draws it: the same
predictors and the same metric as ``plot_results.py``'s scoreboard, with each
bar broken into its two strata.

**The honest caveat is the sample size.** FoldBench's monomers are only 5.7 %
viral: 6 of eval-val's 97 and 13 of eval-test's 217, and none of the 19 designs.
A six-protein mean has a bootstrap interval around +/-0.15, so the per-set viral
bars are indicative at best. The third panel pools the two natural sets -- 19
viral against 295 non-viral -- which is the only cell here with enough proteins
to carry an argument, and it is still thin.

    uv run python plot_viral_split.py
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

import upstream as U  # noqa: E402

DATA = U.DATA
PLOTS = U.HERE / "plots"
OUT = PLOTS / "viral_split_scoreboard.png"
TABLE = DATA / "viral_split.csv"

CHECKPOINT_COLORS = {
    "#232 m2-p06 (decontaminated)": "#d55e00",
    "#232 m1-p02 (decontaminated)": "#e69f00",
    "#199 cooldown (contaminated)": "#0072b2",
}
BASELINE_COLOR = "#8f8b86"
#: Non-viral is the solid bar, viral the hatched one drawn beside it.
VIRAL_HATCH = "//"
BOOTSTRAP_DRAWS = 4_000
SEED = 245

PANELS = (
    ("eval-val", "eval-val\n(91 non-viral · 6 viral)"),
    ("eval-test", "eval-test\n(204 non-viral · 13 viral)"),
    ("natural pooled", "eval-val + eval-test\n(295 non-viral · 19 viral)"),
)


def interval(values: np.ndarray) -> tuple[float, float]:
    """95 % bootstrap interval of the mean; degenerate for n = 1."""
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    generator = np.random.default_rng(SEED)
    index = generator.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load() -> tuple[pd.DataFrame, list[str]]:
    per_protein = pd.read_csv(DATA / "per_protein.csv.gz")
    sets = pd.read_csv(DATA / "eval_sets.csv")
    sets = sets[sets.scorable == 1]
    frame = per_protein[(per_protein["range"] == "all") & (per_protein["cut"] == "R")]
    frame = frame.merge(sets[["stem", "eval_set", "is_viral"]], on="stem")
    natural = frame[frame.eval_set != "eval-denovo"].copy()
    natural["eval_set"] = "natural pooled"
    frame = pd.concat([frame, natural], ignore_index=True)
    order = [p for p in CHECKPOINT_COLORS if p in set(frame.predictor)]
    pooled = frame[frame.eval_set == "natural pooled"]
    order += sorted(set(frame.predictor) - set(order),
                    key=lambda p: -pooled.loc[pooled.predictor == p, "precision"].mean())
    return frame, order


def rows(frame: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Tidy mean + interval + n per (set, predictor, stratum)."""
    records = []
    for eval_set, _ in PANELS:
        for predictor in order:
            subset = frame[(frame.eval_set == eval_set)
                           & (frame.predictor == predictor)]
            for stratum, mask in (("non-viral", subset.is_viral == 0),
                                  ("viral", subset.is_viral == 1)):
                values = subset.loc[mask, "precision"].to_numpy()
                if not len(values):
                    continue
                low, high = interval(values)
                records.append({
                    "eval_set": eval_set, "predictor": predictor,
                    "stratum": stratum, "n": len(values),
                    "value": float(values.mean()), "ci_low": low, "ci_high": high,
                })
    return pd.DataFrame(records)


def draw(table: pd.DataFrame, order: list[str], out: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 6.4), sharey=True)
    width = 0.38
    for axis, (eval_set, title) in zip(axes, PANELS, strict=True):
        panel = table[table.eval_set == eval_set]
        for offset, stratum in ((-width / 2, "non-viral"), (width / 2, "viral")):
            part = panel[panel.stratum == stratum].set_index("predictor")
            positions, heights, lows, highs, colors = [], [], [], [], []
            for index, predictor in enumerate(order):
                if predictor not in part.index:
                    continue
                row = part.loc[predictor]
                positions.append(index + offset)
                heights.append(row.value)
                lows.append(max(0.0, row.value - row.ci_low))
                highs.append(max(0.0, row.ci_high - row.value))
                colors.append(CHECKPOINT_COLORS.get(predictor, BASELINE_COLOR))
            axis.bar(positions, heights, width=width, color=colors,
                     hatch=None if stratum == "non-viral" else VIRAL_HATCH,
                     edgecolor="white", linewidth=0.6,
                     alpha=1.0 if stratum == "non-viral" else 0.78)
            axis.errorbar(positions, heights, yerr=[lows, highs], fmt="none",
                          ecolor="#33312e", elinewidth=1.0, capsize=2.5)
            # Values are printed only on the pooled panel: 54 labels across
            # three panels collides with its own error bars, and the exact
            # numbers live in data/viral_split.csv.
            if eval_set == "natural pooled":
                for position, height, high in zip(positions, heights, highs,
                                                  strict=True):
                    axis.text(position, height + high + 0.015, f"{height:.2f}",
                              ha="center", fontsize=7.6, color="#33312e")
        axis.set_title(title, fontsize=10.5)
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels(order, rotation=40, ha="right", fontsize=8.5)
        axis.grid(axis="y", color="#dddad6", linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("R-precision (all ranges)")
    axes[0].set_ylim(0, 1.02)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE_COLOR, edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE_COLOR, edgecolor="white",
                      hatch=VIRAL_HATCH, alpha=0.78),
    ]
    axes[2].legend(handles, ["non-viral", "viral"], frameon=False,
                   loc="upper right", fontsize=9)
    figure.suptitle(
        "Viral proteins are harder for every predictor that depends on homology — "
        "and not for Protenix + MSA", fontsize=12.5)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out, dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    frame, order = load()
    table = rows(frame, order)
    table.to_csv(TABLE, index=False)
    draw(table, order, OUT)

    sources = {"per_protein": DATA / "per_protein.csv.gz",
               "eval_sets": DATA / "eval_sets.csv"}
    meta = {
        "script": Path(sys.argv[0]).name,
        "args": sys.argv[1:],
        "caption": (
            "All-range R-precision per predictor, split viral vs non-viral, on "
            "each natural eval set and on the two pooled. Hatched bars are "
            "viral. Error bars are 95 % bootstrap intervals; the viral cells "
            "hold 6, 13 and 19 proteins, so their intervals are wide by "
            "construction. eval-denovo is omitted -- none of the 19 designs is "
            "viral."
        ),
        "plot": OUT.name,
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "sources": {name: {"path": str(path.relative_to(U.REPO)),
                           "sha256": U.sha256(path)}
                    for name, path in sources.items()},
    }
    OUT.with_suffix(OUT.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")

    wide = table.pivot_table(index="predictor", columns=["eval_set", "stratum"],
                             values="value")
    print(wide.round(3).to_string())
    print(f"\n[plots] -> {OUT}\n[plots] table -> {TABLE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

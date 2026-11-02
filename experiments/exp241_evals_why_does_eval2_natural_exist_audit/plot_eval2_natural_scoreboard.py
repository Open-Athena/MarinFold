# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 7 — the eval2-natural scoreboard, on the audited n=63 split.

Where MarinFold actually stands against the standard structure predictors on the
decontaminated *natural* set, after the 15 de novo designs #226 mislabelled are
moved to the designed side.

Two panels, because the second is the one the audit added:

* **left** — R-precision on eval2-natural (n=63), six predictors, with a
  bootstrap CI of the mean and a caret marking each predictor's pre-correction
  value on the published n=78. The correction moves the baselines further than
  it moves MarinFold.
* **right** — the same six on the viral (27) and non-viral (36) halves. 43 % of
  eval2-natural is viral and the predictors do not rank the same way on the two.

**Which MarinFold.** The bars are `contacts-v1-exp199-1.5B` (CoreWeave p06), the
checkpoint every baseline in `eval2_per_protein.csv.gz` was scored beside. The
current default is the **p06 cooldown**, which scores 0.3579 against p06's 0.3372
on the *published* n=78 — the two are directly comparable on that subset, and the
cooldown's per-protein eval2 rows live on CoreWeave S3 (unreachable from the
workstation), so it cannot be re-cut to n=63 without an in-cluster job. It is
drawn as a reference line on its own n, labelled as such, never as an n=63 bar.

    uv run python plot_eval2_natural_scoreboard.py
"""
import argparse
import csv
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import upstream as U  # noqa: E402
from build_summary import save_plot_with_meta  # noqa: E402

sys.path.insert(0, str(U.EXP226_DIR))
from build_eval2_scores import MARINFOLD, ORDER  # noqa: E402

DATA = U.HERE / "data"
PLOTS = U.HERE / "plots"
PER_PROTEIN = U.EXP226_DIR / "data" / "eval2_per_protein.csv.gz"

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0

#: Display names, short enough to sit on an axis.
LABELS = {
    MARINFOLD: "MarinFold #199\n(1.5B, sequence only)",
    "Protenix-v2 single-seq": "Protenix-v2\nsingle-seq",
    "ESMFold": "ESMFold",
    "ESMFold2": "ESMFold2",
    "Protenix-v2 + MSA": "Protenix-v2\n+ MSA",
    "seq-KNN k=10 (null)": "seq-KNN\n(null)",
}

BLUE = "#2980b9"        # MarinFold — the subject of the chart
GREY = "#95a5a6"        # the structure predictors
LIGHT = "#d5dbdb"       # the null
RED = "#c0392b"
ORANGE = "#d68910"

#: The current default checkpoint, scored on the *published* 78 — see the module
#: docstring for why it is not an n=63 bar.
COOLDOWN_LABEL = "contacts-v1-exp199-cooldown-1.5B"
COOLDOWN_R_N78 = 0.357937
P06_R_N78 = 0.3372

CAPTION = (
    "R-precision on eval2-natural after moving exp226's 15 mislabelled de novo "
    "designs to the designed side (n=63). Left: all six predictors, bootstrap CI "
    "of the mean, carets mark the pre-correction n=78 value. Right: the viral and "
    "non-viral halves scored separately."
)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, matching exp226's resample count."""
    idx = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def load() -> pd.DataFrame:
    """Per-protein all-range R-precision joined to the audited designed flag."""
    manifest = pd.read_csv(DATA / "eval2_manifest_v2.csv")
    wide = pd.read_csv(PER_PROTEIN)
    wide = wide[(wide["range"] == "all") & (wide["cut"] == "R")]
    wide = wide.drop(columns=["designed_any"]).merge(
        manifest[["dataset", "stem", "designed_any", "is_viral"]],
        on=["dataset", "stem"], how="left", validate="many_to_one")
    if wide["designed_any"].isna().any():
        raise SystemExit("some scored proteins did not join to the v2 manifest")
    return wide[wide["designed_any"] == 0]


def plot(natural: pd.DataFrame, out, argv) -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    published = {r["predictor"]: r["published"] for r in
                 csv.DictReader((DATA / "correction_effect.csv").open())
                 if r["metric"] == "R (all)"
                 and r["subset"] == "eval2 natural (audited)"}

    stats = []
    for predictor in ORDER:
        values = natural[predictor].dropna().to_numpy()
        lo, hi = bootstrap_ci(values, rng)
        stats.append({"predictor": predictor, "mean": values.mean(),
                      "lo": lo, "hi": hi, "n": len(values),
                      "before": float(published[predictor])})
    stats.sort(key=lambda s: s["mean"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.4),
                                  gridspec_kw={"width_ratios": [1.25, 1]})

    y = np.arange(len(stats))
    colors = [BLUE if s["predictor"] == MARINFOLD
              else LIGHT if "null" in s["predictor"] else GREY for s in stats]
    ax.barh(y, [s["mean"] for s in stats], 0.62, color=colors)
    ax.errorbar([s["mean"] for s in stats], y,
                xerr=[[s["mean"] - s["lo"] for s in stats],
                      [s["hi"] - s["mean"] for s in stats]],
                fmt="none", ecolor="#2c3e50", elinewidth=1.4, capsize=3)
    # The pre-correction value on the published n=78, so the effect of moving the
    # 15 designs is visible without a second chart.
    ax.scatter([s["before"] for s in stats], y, marker="|", s=260,
               color=RED, linewidths=2, zorder=5, label="before correction (n=78)")
    for yi, s in zip(y, stats):
        ax.text(max(s["hi"], s["before"]) + 0.018, yi, f"{s['mean']:.3f}",
                va="center", fontsize=10, fontweight="bold", color="#2c3e50")

    ax.set_yticks(y, [LABELS[s["predictor"]] for s in stats], fontsize=9)
    ax.set_xlim(0, 0.86)
    ax.set_ylim(-0.6, len(stats) - 0.2)
    ax.set_xlabel("R-precision (all ranges)")
    ax.set_title(f"eval2-natural, audited (n={len(natural)})\n"
                 "MarinFold is a sequence-only LM; the rest fold structure",
                 fontsize=11)
    # Anchored above the axes: inside the plot the caret glyph reads as a data
    # point on whichever row it lands in.
    ax.legend(frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=1, handletextpad=0.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    # --- right: the viral / non-viral split --------------------------------
    order = [s["predictor"] for s in reversed(stats)]
    halves = [("viral", natural[natural["is_viral"] == 1], ORANGE),
              ("non-viral", natural[natural["is_viral"] == 0], BLUE)]
    x = np.arange(len(order))
    width = 0.38
    for i, (label, subset, color) in enumerate(halves):
        means = [subset[p].dropna().mean() for p in order]
        offset = (i - 0.5) * width
        ax2.bar(x + offset, means, width * 0.92, color=color,
                label=f"{label} (n={len(subset)})")
        for xi, value in zip(x, means):
            ax2.text(xi + offset, value + 0.015, f"{value:.2f}", ha="center",
                     fontsize=8, color="#2c3e50")
    ax2.set_xticks(x, [LABELS[p].replace("\n", " ").replace(
        " (1.5B, sequence only)", "") for p in order],
        rotation=32, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 0.86)
    ax2.set_ylabel("R-precision (all ranges)")
    ax2.set_title("43 % of eval2-natural is viral —\nand the halves do not rank alike",
                  fontsize=11)
    ax2.legend(frameon=False, fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax2.set_axisbelow(True)

    fig.text(0.012, 0.015,
             f"MarinFold bars = contacts-v1-exp199-1.5B (CoreWeave p06), the "
             f"checkpoint the baselines were scored beside. The current default "
             f"{COOLDOWN_LABEL} scores {COOLDOWN_R_N78:.3f} vs p06's "
             f"{P06_R_N78:.3f} on the published n=78; its per-protein eval2 rows "
             f"are on CoreWeave S3 and need an in-cluster job to re-cut to n=63.",
             fontsize=7.2, color="#5d6d7e", wrap=True)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    save_plot_with_meta(fig, out, caption=CAPTION, args=argv)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    PLOTS.mkdir(exist_ok=True)
    plot(load(), PLOTS / "eval2_natural_scoreboard.png", list(sys.argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

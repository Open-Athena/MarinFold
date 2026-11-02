# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the exp200 result: a real per-candidate gain, cancelled by lost spread.

matplotlib and pandas are NOT in this experiment's lock, deliberately — the lock
pins marin 0.2.76 and two vLLM fork SHAs, and relocking it for a plot risks
disturbing an environment that took several attempts to resolve. Run with an
ephemeral environment instead::

    uv run --with matplotlib --with pandas python plot_results.py

Reads `data/eval_lr1em06_vs_armF_per_protein.csv` (paired per protein, n=554)
and writes `plots/effect_sizes.png` + `plots/quality_vs_spread.png` with the
sidecars `build_summary.py` needs.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_summary import save_plot_with_meta  # noqa: E402

QUALITY = "#B4552F"   # precision rose
SPREAD = "#37707F"    # diversity fell
FLAT = "#6B7180"      # the primary metric, unmoved

# (label, csv stem, which force it belongs to)
METRICS = [
    ("best-of-N F1  (primary)", "best_f1_mine", "flat"),
    ("first-candidate F1", "first_f1_mine", "quality"),
    ("per-contact precision", "precision", "quality"),
    ("last-candidate F1", "last_f1_mine", "quality"),
    ("candidates per generation", "n_sections_mine", "spread"),
    ("inter-candidate Jaccard", "mean_jaccard", "spread"),
]
COLORS = {"quality": QUALITY, "spread": SPREAD, "flat": FLAT}


def paired(df: pd.DataFrame, stem: str):
    d = (df[f"{stem}_t"] - df[f"{stem}_b"]).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, d.mean() / se, (d > 0).mean()


def main() -> int:
    df = pd.read_csv("data/eval_lr1em06_vs_armF_per_protein.csv")

    # --- effect sizes, in sigma, signed by direction of the underlying force ---
    rows = [(label, *paired(df, stem), force) for label, stem, force in METRICS]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    y = np.arange(len(rows))[::-1]
    for yi, (label, mean, se, sigma, win, force) in zip(y, rows):
        # Jaccard rising and sections falling both mean LESS spread; plot the
        # loss of spread as negative so the two forces read as opposed.
        plotted = -abs(sigma) if force == "spread" else sigma
        ax.barh(yi, plotted, height=0.58, color=COLORS[force],
                alpha=.85 if force != "flat" else .55)
        ax.text(plotted + (0.18 if plotted >= 0 else -0.18), yi,
                f"{mean:+.4f}  ({sigma:+.1f}σ)" if force != "spread"
                else f"{mean:+.4f}  ({sigma:+.1f}σ)",
                va="center", ha="left" if plotted >= 0 else "right",
                fontsize=8.6, color="#33383F")
    ax.axvline(0, color="#33383F", lw=.9)
    for t in (3, -3):
        ax.axvline(t, color="#9AA1B2", lw=.7, ls=(0, (3, 3)))
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9.4)
    ax.set_xlabel("paired effect size (σ), n = 554 proteins\n"
                  "spread metrics plotted as loss of spread; dashed lines = ±3σ", fontsize=8.8)
    ax.set_xlim(-11.5, 9.5)
    ax.set_title("Candidate quality rose. Spread fell. The product did not move.",
                 fontsize=11.4, loc="left", pad=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_plot_with_meta(
        fig, "plots/effect_sizes.png", dpi=190,
        # Captions over two lines silently overlap the plot; keep under ~200 chars.
        caption=("Paired per-protein effect sizes, 1e-6 arm vs #163 arm F. Quality rose at "
                 "4.6-5.1 sigma, spread fell at 7.5-7.7; their product, best-of-N F1, sits at "
                 "+0.0008."),
    )
    plt.close(fig)

    # --- the trade, per protein ---
    dq = df["first_f1_mine_t"] - df["first_f1_mine_b"]
    ds = df["n_sections_mine_t"] - df["n_sections_mine_b"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(ds, dq, s=13, alpha=.5, color=SPREAD, edgecolor="none")
    ax.axhline(0, color="#33383F", lw=.8)
    ax.axvline(0, color="#33383F", lw=.8)
    ax.scatter([ds.mean()], [dq.mean()], s=150, marker="+", color=QUALITY, lw=2.2,
               zorder=5, label=f"mean ({ds.mean():+.2f}, {dq.mean():+.4f})")
    ax.set_xlabel("Δ candidates per generation", fontsize=9.4)
    ax.set_ylabel("Δ first-candidate F1", fontsize=9.4)
    # Be exact rather than flattering: up-and-left is the plurality quadrant at
    # 41.0%, not a majority, and the anticorrelation is weak (r = -0.20).
    ul = 100 * ((dq > 0) & (ds < 0)).mean()
    r = dq.corr(ds)
    ax.set_title(f"Fewer candidates, slightly better ones — {ul:.0f}% of proteins (r = {r:.2f})",
                 fontsize=10.6, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.6, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save_plot_with_meta(
        fig, "plots/quality_vs_spread.png", dpi=190,
        caption=("Per protein: change in first-candidate F1 vs candidate count. Better-and-fewer "
                 "is the plurality quadrant at 41%, and the trade is weak per protein (r = -0.20) "
                 "even though both means shift."),
    )
    plt.close(fig)

    print("wrote plots/effect_sizes.png and plots/quality_vs_spread.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

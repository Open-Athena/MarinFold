# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp82 single-panel, high-resolution cut of the predictor-comparison boxplot.

``plot_comparison.py`` renders a 1x4 grid (all / short / medium / long) sized for
a slide. This script renders **one range only** (default ``all``) as a standalone
figure at publication resolution, reusing that module's ``CONFIGS``, palette and
box styling verbatim so the panel is identical to its 4-panel counterpart --
only larger, and with the range folded into the title.

Emits a raster ``.png`` (``--dpi``, default 600) and a vector ``.pdf`` next to it;
the PDF is resolution-independent and is the one to drop into a paper or slide.

Reads the same table as ``plot_comparison.py`` (``build_comparison_table.py``'s
output). Run in a venv with pandas + seaborn::

    <exp82-venv>/bin/python plot_all_only.py \
        --precision-csv _scratch/contact_precision_with_rollout.csv --out plots
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from build_summary import save_plot_with_meta
from plot_comparison import BOXNOTE, CONFIGS, CUT_FILE, _axis_label, _title, _vals

# The "all" band is every pair at sep >= 6 (plot_comparison.RANGES); spell the
# band out in the title since a lone panel has no siblings to contrast against.
RANGE_BLURB = {"all": "all contacts (sep ≥ 6)", "short": "short range (sep 6–11)",
               "medium": "medium range (sep 12–23)", "long": "long range (sep ≥ 24)"}


def plot_single_range(df, out_base: Path, *, cut: str, rng: str, dpi: int, script_args) -> list[Path]:
    sub = df[(df["cut"] == cut) & (df["range"] == rng)]
    if sub.empty:
        raise SystemExit(f"no rows for cut={cut!r} range={rng!r} -- is this the merged table?")

    labels = [c[3] for c in CONFIGS]
    palette = {c[3]: c[4] for c in CONFIGS}
    rows, means = [], {}
    for model, mode, pred, disp, _ in CONFIGS:
        v = _vals(sub, model, mode, pred)
        v = v[np.isfinite(v)]
        rows += [(disp, x) for x in v]
        means[disp] = float(v.mean()) if v.size else float("nan")
    bdf = pd.DataFrame(rows, columns=["cfg", "precision"])
    # The eval unit is (dataset, stem), not stem: 7ur7_A and 8ah9_A each appear in
    # two datasets, so counting stems alone would report 552 where exp82 says 554.
    n_prot = int(sub.drop_duplicates(["dataset", "stem"]).shape[0])

    # Same box styling as plot_comparison.plot_by_config_and_range; marker/font
    # sizes nudged up because one panel gets the space four used to share.
    meanprops = dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6)
    flierprops = dict(marker=".", markersize=3, markerfacecolor="0.4", markeredgecolor="none", alpha=0.35)
    fig, ax = plt.subplots(figsize=(8, 6.4))
    sns.boxplot(data=bdf, x="cfg", y="precision", order=labels, hue="cfg", palette=palette,
                ax=ax, width=0.62, legend=False, showmeans=True, meanprops=meanprops,
                flierprops=flierprops, medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(alpha=0.85), linewidth=1.1)
    for xi, disp in enumerate(labels):
        if not np.isnan(means[disp]):
            ax.text(xi, 1.03, f"{means[disp]:.2f}", ha="center", va="bottom", fontsize=9.5)
    if cut == "AUC":
        ax.axhline(0.5, ls="--", lw=1, color="grey", zorder=0)
    for t in ax.get_xticklabels():
        t.set_rotation(30)
        t.set_horizontalalignment("right")
        t.set_fontsize(10)
    ax.tick_params(axis="y", labelsize=10)
    # plot_comparison hangs this note off fig.text because four panels share one
    # copy; as this panel's xlabel it tracks the rotated tick labels instead of
    # stranding a band of whitespace at the canvas bottom. It is written as one
    # slide-width line, so fold it in half to fit a lone panel's narrower axes.
    parts = BOXNOTE.replace("n=554 proteins", f"n={n_prot} proteins").split("   ·   ")
    half = (len(parts) + 1) // 2
    note = "   ·   ".join(parts[:half]) + "\n" + "   ·   ".join(parts[half:])
    ax.set_xlabel(note, fontsize=9, color="0.3", labelpad=12)
    ax.set_ylabel(_axis_label(cut), fontsize=12)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(axis="y", alpha=0.3)

    head = (f"{_title(cut)}: MarinFold rollout+resample vs pairwise / "
            f"Protenix-v2 / ESMFold / ESMFold2")
    ax.set_title(f"{textwrap.fill(head, 58)}\n{RANGE_BLURB.get(rng, rng)}  ·  n={n_prot}",
                 fontsize=12.5)
    fig.tight_layout()

    caption = (f"{_axis_label(cut)} vs pyconfind contacts (degree>=0.001, sep>=6), per predictor, "
               f"{RANGE_BLURB.get(rng, rng)} only, over {n_prot} proteins. Single-panel high-resolution "
               f"cut of cmp_contacts_at_{CUT_FILE[cut]}_by_config_and_range.png (same data and styling). "
               f"Boxplots: box = median & IQR, whiskers = 1.5x IQR, points = outliers, white diamond = "
               f"mean (labelled). rollout+resample ranks pairs by sampled-completion vote frequency; "
               f"pairwise by contact-statement log-prob; structure models by pyconfind degree on the "
               f"predicted structure. Metrics computed by exp89's exact code.")

    written = []
    for suffix, kw in ((".png", {"dpi": dpi}), (".pdf", {})):
        path = out_base.with_suffix(suffix)
        save_plot_with_meta(fig, path, caption=caption, script="plot_all_only.py",
                            args=script_args, **kw)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--precision-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("plots"))
    ap.add_argument("--cut", default="R", choices=list(CUT_FILE))
    ap.add_argument("--range", dest="rng", default="all", choices=list(RANGE_BLURB))
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    df = pd.read_csv(args.precision_csv)
    sa = ["--precision-csv", str(args.precision_csv), "--out", str(args.out),
          "--cut", args.cut, "--range", args.rng, "--dpi", str(args.dpi)]
    base = args.out / f"cmp_contacts_at_{CUT_FILE[args.cut]}_{args.rng}_highres"
    for p in plot_single_range(df, base, cut=args.cut, rng=args.rng, dpi=args.dpi, script_args=sa):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

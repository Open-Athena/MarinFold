# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Project-README "where we stand" figure — contact R-precision (n=554).

A single aggregate boxplot panel (range = all, sep>=6) putting MarinFold's best
inference — **rollout+resample+tiebreak**, labelled "MarinFold #61 n=100 rollouts"
— next to the structure predictors. Replaces exp89's pairwise/ensemble version.
Reuses exp89 `plot.py`'s `plot_single_panel` style; reads the rollout-extended
`contact_precision_with_rollout.csv` (`build_comparison_table.py`).

    uv run python plot_where_we_stand.py --precision-csv <csv> --out <png>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# One MarinFold bar (the rollout recipe) + the structure predictors.
CONFIGS = [
    ("marinfold-cv1-rollout-resample-tiebreak", "single_seq", "lm", "MarinFold #61 n=100 rollouts", "#7f2704"),
    ("protenix-v2", "single_seq", "structure", "Protenix-v2 · SS", "#9ecae1"),
    ("protenix-v2", "msa", "structure", "Protenix-v2 · MSA", "#2171b5"),
    ("esmfold", "single_seq", "structure", "ESMFold", "#74c476"),
    ("esmfold2", "single_seq", "structure", "ESMFold2", "#238b45"),
]


def _vals(df, model, mode, predictor):
    s = df[(df["model"] == model) & (df["mode"] == mode) & (df["predictor"] == predictor)]
    return s["precision"].to_numpy(dtype=float)


def main(csv: Path, out: Path) -> None:
    df = pd.read_csv(csv)
    sub = df[(df["cut"] == "R") & (df["range"] == "all")]
    labels = [c[3] for c in CONFIGS]
    palette = {c[3]: c[4] for c in CONFIGS}
    rows, means = [], {}
    for model, mode, pred, disp, _ in CONFIGS:
        v = _vals(sub, model, mode, pred)
        v = v[np.isfinite(v)]
        rows += [(disp, x) for x in v]
        means[disp] = float(v.mean()) if v.size else float("nan")
    bdf = pd.DataFrame(rows, columns=["cfg", "precision"])

    fig, ax = plt.subplots(figsize=(9, 5.6))
    sns.boxplot(data=bdf, x="cfg", y="precision", order=labels, hue="cfg", palette=palette, ax=ax,
                width=0.6, legend=False, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
                flierprops=dict(marker=".", markersize=2, markerfacecolor="0.4", markeredgecolor="none", alpha=0.35),
                medianprops=dict(color="black", linewidth=1.4), boxprops=dict(alpha=0.85), linewidth=1.0)
    for xi, disp in enumerate(labels):
        if not np.isnan(means[disp]):
            ax.text(xi, 1.03, f"{means[disp]:.2f}", ha="center", va="bottom", fontsize=9)
    for t in ax.get_xticklabels():
        t.set_rotation(22); t.set_horizontalalignment("right"); t.set_fontsize(9)
    ax.set_xlabel("")
    ax.set_ylabel("R-precision", fontsize=11)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Contact R-precision  (n=554)", fontsize=13)
    fig.text(0.5, 0.005, "box = median & IQR · whiskers = 1.5×IQR · ◆ = mean · points = outliers",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(
        {"script": "plot_where_we_stand.py", "args": ["--precision-csv", str(csv)],
         "caption": ("Contact R-precision (all sep>=6, n=554): MarinFold #61 n=100 rollouts "
                     "(rollout+resample+tiebreak) vs Protenix-v2 / ESMFold / ESMFold2.")}, indent=2))
    print(f"wrote {out}  means: " + ", ".join(f"{k}={v:.3f}" for k, v in means.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    main(args.precision_csv, args.out)

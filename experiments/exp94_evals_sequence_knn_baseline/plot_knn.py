# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 5 — the plots the issue asks for.

* ``plots/headline_rprecision.png`` — long-range R-precision bars: MarinFold #61
  rollout+resample (n=100) next to the seq-KNN sweep and the standard predictors
  (Protenix-v2 single-seq/MSA, ESMFold, ESMFold2). The issue's success criterion.
* ``plots/k_sweep.png`` — long-range R-precision vs k for seq-KNN, self-included
  vs self-excluded.
* ``plots/memorization_scatter.png`` — per-protein long-range R-precision,
  seq-KNN k=10 vs MarinFold rollout+resample. High correlation ⇒ the LM tracks the
  copy-nearest-neighbor signal (the core memorization read).

Reads ``data/knn_comparison.csv`` (aggregated) + ``data/knn_precision_all.csv``
(per-protein KNN) + the base per-protein CSV (for MarinFold per-protein).

    uv run python plot_knn.py --base-csv <exp82>/_scratch/contact_precision_with_rollout.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MARINFOLD = "marinfold-cv1-rollout-resample-tiebreak"  # "#61 with n=100 rollouts"
# (display label, model, mode, predictor) for the headline bars, in reference
# order. protenix-v2 carries both a `distogram` and a `structure` predictor under
# one model name; we show the structure variant (contacts from the predicted
# structure, like ESMFold) to match the issue's "standard comparison predictors".
HEADLINE = [
    ("seq-KNN k=1", "seq-knn-k1", "single_seq", "knn"),
    ("seq-KNN k=10", "seq-knn-k10", "single_seq", "knn"),
    ("seq-KNN k=50", "seq-knn-k50", "single_seq", "knn"),
    ("MarinFold #61\n(rollout x100)", MARINFOLD, "single_seq", "lm"),
    ("Protenix-v2\n(single-seq)", "protenix-v2", "single_seq", "structure"),
    ("Protenix-v2\n(MSA)", "protenix-v2", "msa", "structure"),
    ("ESMFold", "esmfold", "single_seq", "structure"),
    ("ESMFold2", "esmfold2", "single_seq", "structure"),
]


def lookup(agg: pd.DataFrame, model: str, mode: str, rng: str, cut: str,
           predictor: str | None = None) -> float:
    m = agg[(agg.model == model) & (agg["mode"] == mode)
            & (agg.range == rng) & (agg.cut == cut)]
    if predictor is not None:
        m = m[m.predictor == predictor]
    return float(m["mean_precision"].iloc[0]) if len(m) else float("nan")


def plot_headline(agg: pd.DataFrame, out: Path) -> None:
    labels = [h[0] for h in HEADLINE]
    vals = [lookup(agg, h[1], h[2], "long", "R", h[3]) for h in HEADLINE]
    colors = ["#888" if "KNN" in lbl else ("#2a7" if "MarinFold" in lbl else "#48c")
              for lbl in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("long-range R-precision")
    ax.set_title("Contact prediction: sequence-KNN null model vs MarinFold and structure predictors")
    ax.set_ylim(0, max(v for v in vals if v == v) * 1.18)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_k_sweep(agg: pd.DataFrame, out: Path, ks=(1, 5, 10, 25, 50)) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode_suffix, lbl, style in [("", "self-included", "-o"), ("-noself", "self-excluded", "--s")]:
        ys = [lookup(agg, f"seq-knn-k{k}{mode_suffix}", "single_seq", "long", "R") for k in ks]
        ax.plot(ks, ys, style, label=lbl)
    mf = lookup(agg, MARINFOLD, "single_seq", "long", "R")
    ax.axhline(mf, color="#2a7", ls=":", label=f"MarinFold #61 ({mf:.3f})")
    ax.set_xlabel("k (nearest training neighbors)")
    ax.set_ylabel("long-range R-precision")
    ax.set_title("seq-KNN: R-precision vs k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_scatter(knn: pd.DataFrame, base: pd.DataFrame, out: Path, k: int = 10) -> None:
    sel = lambda df, model: df[(df.model == model) & (df.range == "long") & (df.cut == "R")][
        ["dataset", "stem", "precision"]]
    kn = sel(knn, f"seq-knn-k{k}").rename(columns={"precision": "knn"})
    mf = sel(base, MARINFOLD).rename(columns={"precision": "marinfold"})
    m = kn.merge(mf, on=["dataset", "stem"]).dropna()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m["knn"], m["marinfold"], s=12, alpha=0.45, edgecolor="none")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    r = m["knn"].corr(m["marinfold"]) if len(m) > 2 else float("nan")
    ax.set_xlabel(f"seq-KNN k={k}  long-range R-precision")
    ax.set_ylabel("MarinFold #61  long-range R-precision")
    ax.set_title(f"Per-protein agreement (n={len(m)}, Pearson r={r:.2f})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}  (r={r:.3f})")


def plot_stratified(knn: pd.DataFrame, base: pd.DataFrame, summary: pd.DataFrame,
                    out: Path, k: int = 10) -> None:
    """R-precision binned by the eval protein's best train-hit sequence identity.

    The direct memorization read: as the nearest training neighbor gets closer,
    does MarinFold's accuracy track the copy-the-neighbor baseline? `query` in the
    hit summary is `{dataset}__{stem}`; split it to join on (dataset, stem).
    """
    sel = lambda df, model: df[(df.model == model) & (df.range == "long") & (df.cut == "R")][
        ["dataset", "stem", "precision"]]
    kn = sel(knn, f"seq-knn-k{k}").rename(columns={"precision": "knn"})
    mf = sel(base, MARINFOLD).rename(columns={"precision": "marinfold"})
    s = summary.copy()
    s[["dataset", "stem"]] = s["query"].str.split("__", n=1, expand=True)
    m = kn.merge(mf, on=["dataset", "stem"]).merge(s[["dataset", "stem", "best_fident", "n_hits"]],
                                                   on=["dataset", "stem"])
    m["ident"] = m["best_fident"].where(m["n_hits"] > 0, 0.0)

    edges = [(-0.01, 0.001, "no hit"), (0.001, 0.3, "0-30%"), (0.3, 0.5, "30-50%"),
             (0.5, 0.7, "50-70%"), (0.7, 1.01, "70-100%")]
    labels, knn_means, mf_means, counts = [], [], [], []
    for lo, hi, lbl in edges:
        b = m[(m["ident"] > lo) & (m["ident"] <= hi)]
        if len(b) == 0:
            continue
        labels.append(f"{lbl}\n(n={len(b)})")
        knn_means.append(b["knn"].mean())
        mf_means.append(b["marinfold"].mean())
        counts.append(len(b))
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - 0.2 for i in x], knn_means, 0.4, label=f"seq-KNN k={k}", color="#888")
    ax.bar([i + 0.2 for i in x], mf_means, 0.4, label="MarinFold #61", color="#2a7")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("best training-set sequence identity (nearest neighbor)")
    ax.set_ylabel("long-range R-precision")
    ax.set_title("Contact accuracy vs nearest-training-neighbor identity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


IDENT_BINS = [(-0.01, 0.001, "no hit"), (0.001, 0.3, "0-30%"), (0.3, 0.5, "30-50%"),
              (0.5, 0.7, "50-70%"), (0.7, 1.01, "70-100%")]
# (label, model, mode, predictor, source) — source picks the per-protein frame.
STRAT_PREDICTORS = [
    ("seq-KNN k=10", "seq-knn-k10", "single_seq", "knn", "knn", "#888", "-o"),
    ("MarinFold #61", MARINFOLD, "single_seq", "lm", "base", "#2a7", "-o"),
    ("Protenix-v2 (single-seq)", "protenix-v2", "single_seq", "structure", "base", "#48c", "--s"),
    ("Protenix-v2 (MSA)", "protenix-v2", "msa", "structure", "base", "#15396b", "--s"),
    ("ESMFold", "esmfold", "single_seq", "structure", "base", "#e8a", "--^"),
    ("ESMFold2", "esmfold2", "single_seq", "structure", "base", "#a26", "--^"),
]


def _per_protein_long_r(df, model, mode, predictor):
    sub = df[(df.model == model) & (df["mode"] == mode) & (df.predictor == predictor)
             & (df.range == "long") & (df.cut == "R")]
    return sub[["dataset", "stem", "precision"]].dropna()


def plot_stratified_all(knn, base, summary, out, k: int = 10) -> None:
    """Long-range R-precision vs nearest-neighbor identity, for every predictor.

    Same identity bins as `plot_stratified`, but as a line per predictor so the
    contrast is visible: seq-KNN (copy baseline) rises with identity while the
    folding predictors are flat — their accuracy doesn't depend on whether a
    training homolog exists.
    """
    s = summary.copy()
    s[["dataset", "stem"]] = s["query"].str.split("__", n=1, expand=True)
    s["ident"] = s["best_fident"].where(s["n_hits"] > 0, 0.0)
    s = s[["dataset", "stem", "ident"]]

    x = range(len(IDENT_BINS))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, model, mode, pred, src, color, style in STRAT_PREDICTORS:
        frame = knn if src == "knn" else base
        pp = _per_protein_long_r(frame, model, mode, pred).merge(s, on=["dataset", "stem"])
        ys, xs = [], []
        for i, (lo, hi, _) in enumerate(IDENT_BINS):
            b = pp[(pp["ident"] > lo) & (pp["ident"] <= hi)]
            if len(b):
                xs.append(i)
                ys.append(b["precision"].mean())
        ax.plot(xs, ys, style, color=color, label=label, lw=1.8, ms=6)

    # bin counts (predictor-independent: full eval set)
    counts = [int(((s["ident"] > lo) & (s["ident"] <= hi)).sum()) for lo, hi, _ in IDENT_BINS]
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{lbl}\n(n={c})" for (_, _, lbl), c in zip(IDENT_BINS, counts)], fontsize=8)
    ax.set_xlabel("best training-set sequence identity (nearest neighbor)")
    ax.set_ylabel("long-range R-precision")
    ax.set_title("Contact accuracy vs nearest-training-neighbor identity — all predictors")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, default=Path("data/knn_comparison.csv"))
    ap.add_argument("--knn-per-protein", type=Path, default=Path("data/knn_precision_all.csv"))
    ap.add_argument("--hit-summary", type=Path, default=Path("data/knn_hit_summary.csv"))
    ap.add_argument("--base-csv", type=Path, required=True)
    ap.add_argument("--plots", type=Path, default=Path("plots"))
    args = ap.parse_args()

    agg = pd.read_csv(args.comparison)
    knn = pd.read_csv(args.knn_per_protein)
    base = pd.read_csv(args.base_csv)
    summary = pd.read_csv(args.hit_summary)
    args.plots.mkdir(parents=True, exist_ok=True)

    plot_headline(agg, args.plots / "headline_rprecision.png")
    plot_k_sweep(agg, args.plots / "k_sweep.png")
    plot_scatter(knn, base, args.plots / "memorization_scatter.png")
    plot_stratified(knn, base, summary, args.plots / "rprecision_vs_identity.png")
    plot_stratified_all(knn, base, summary, args.plots / "rprecision_vs_identity_all.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

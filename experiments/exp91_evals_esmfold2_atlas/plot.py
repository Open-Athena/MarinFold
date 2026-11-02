# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the exp91 characterization figures from the CSVs in ``data/``.

Reads the sample parquet + the summary CSVs written by ``characterize.py`` and
``novelty.py`` and writes PNGs to ``plots/``:

- ``composition.png`` — length / mean-pLDDT / pTM distributions, with the
  paper's pLDDT > 0.7 high-confidence line and a pLDDT-vs-length hexbin.
- ``pfam_darkmatter.png`` — distribution of cluster Pfam characterization and
  the naming-tier counts.
- ``source_composition.png`` — source-DB split of the 6.8 B sequences.
- ``novelty.png`` — nearest-afdb-rep TM-score histogram + verdict counts
  (only if ``data/atlas_vs_afdb_reps_similarity.csv`` exists).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
PLDDT_HIGH = 0.70


def plot_composition() -> None:
    meta = pd.read_parquet(DATA / "atlas_sample.parquet")
    frac_hi = (meta["mean_plddt"] > PLDDT_HIGH).mean()
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].hist(meta["seq_len"], bins=60, color="#4477aa")
    ax[0].set(xlabel="sequence length (aa)", ylabel="count",
              title=f"length (median {meta['seq_len'].median():.0f})")
    ax[1].hist(meta["mean_plddt"], bins=60, color="#66ccee")
    ax[1].axvline(PLDDT_HIGH, color="crimson", ls="--",
                  label=f">0.7: {frac_hi:.0%}")
    ax[1].set(xlabel="mean pLDDT", title="mean pLDDT"); ax[1].legend()
    ax[2].hist(meta["ptm"], bins=60, color="#228833")
    ax[2].set(xlabel="pTM", title=f"pTM (median {meta['ptm'].median():.2f})")
    hb = ax[3].hexbin(meta["seq_len"], meta["mean_plddt"], gridsize=40,
                      cmap="viridis", bins="log")
    ax[3].axhline(PLDDT_HIGH, color="crimson", ls="--")
    ax[3].set(xlabel="length (aa)", ylabel="mean pLDDT", title="pLDDT vs length")
    fig.colorbar(hb, ax=ax[3], label="log10(count)")
    fig.suptitle(f"ESM Atlas folds — sample of {len(meta):,} of 1.1B (monomers)")
    fig.tight_layout()
    fig.savefig(PLOTS / "composition.png", dpi=120)
    plt.close(fig)


def plot_pfam() -> None:
    hist = pd.read_csv(DATA / "pfam_pct_hist.csv")
    tiers = pd.read_csv(DATA / "naming_tier_counts.csv")
    summ = pd.read_csv(DATA / "pfam_characterization.csv").iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(hist["bin_left"], hist["count"], width=4.5, align="edge",
              color="#aa3377")
    ax[0].set(xlabel="% of cluster members with a characterized Pfam domain",
              ylabel="clusters",
              title=f"Pfam dark matter — {summ['frac_fully_dark_pct0']:.0%} "
                    f"fully dark (~{summ['n_fully_dark_pct0'] / 1e6:.1f}M of "
                    f"{summ['n_clusters_ge50'] / 1e6:.1f}M)")
    tcol = tiers.columns[0]
    ax[1].bar(tiers[tcol].astype(str), tiers["count"], color="#ccbb44")
    ax[1].set(xlabel="naming_tier", ylabel="clusters",
              title="cluster naming tier (0 = best-characterized name)")
    fig.tight_layout()
    fig.savefig(PLOTS / "pfam_darkmatter.png", dpi=120)
    plt.close(fig)


def plot_source() -> None:
    src = DATA / "source_composition.csv"
    if not src.exists():
        print("[plot] no source_composition.csv; skipping")
        return
    df = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["source"], df["fraction"], color="#4477aa")
    meta_frac = df.loc[df["source"].isin(["spire", "mgy"]), "fraction"].sum()
    ax.set(ylabel="fraction of sequences",
           title=f"ESM Atlas source DBs (6.8B seqs) — "
                 f"metagenomic SPIRE+MGnify = {meta_frac:.0%}")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, r["fraction"], f"{r['fraction']:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOTS / "source_composition.png", dpi=120)
    plt.close(fig)


def plot_afdb_coverage() -> None:
    p = DATA / "afdb_uniref_coverage.csv"
    if not p.exists():
        return
    r = pd.read_csv(p).iloc[0]
    ts = [30, 50, 70, 90, 95]
    fracs = [r[f"frac_uniref_id_ge_{t}"] for t in ts]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ts, fracs, "o-", color="#4477aa")
    ax.axhline(1 - r["frac_no_uniprot_match"], color="grey", ls=":",
               label=f"any UniRef match: {1 - r['frac_no_uniprot_match']:.0%}")
    ax.set(xlabel="UniRef sequence identity threshold (%)",
           ylabel="fraction of Atlas clusters ≥ threshold",
           ylim=(0, 1),
           title=f"Atlas → AFDB sequence coverage — only "
                 f"{r['frac_uniref_id_ge_90']:.0%} are ≥90% id to UniProt "
                 f"({r['frac_no_uniprot_match']:.0%} have no match)")
    for t, f in zip(ts, fracs):
        ax.text(t, f + 0.02, f"{f:.0%}", ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "afdb_coverage.png", dpi=120)
    plt.close(fig)


def plot_pdb() -> None:
    p = DATA / "atlas_vs_pdb.csv"
    if not p.exists():
        print("[plot] no PDB CSV yet; skipping")
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(df["qtmscore"].fillna(0.0), bins=40, color="#aa3377")
    ax.axvline(0.5, color="crimson", ls="--", label="TM 0.5 (fold match)")
    ax.set(xlabel="best PDB TM-score", ylabel="Atlas structures",
           title=f"Atlas → PDB structural overlap (n={len(df)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "atlas_vs_pdb.png", dpi=120)
    plt.close(fig)


def plot_leakage() -> None:
    p = DATA / "atlas_leakage.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    g = df.groupby("source").agg(exact=("exact_dup", "mean"),
                                 near=("near_dup", "mean"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(g))
    ax.bar([i - 0.2 for i in x], g["exact"], width=0.4, label="exact dup",
           color="#cc3311")
    ax.bar([i + 0.2 for i in x], g["near"], width=0.4,
           label="near dup (≥30% id)", color="#ee9988")
    ax.set_xticks(list(x)); ax.set_xticklabels(g.index)
    ax.set(ylabel="fraction of eval proteins",
           title=f"exp65 eval-set leakage into the Atlas "
                 f"({df['exact_dup'].mean():.0%} exact / {df['near_dup'].mean():.0%} near overall)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "leakage.png", dpi=120)
    plt.close(fig)


def plot_novelty() -> None:
    p = DATA / "atlas_vs_afdb_reps_similarity.csv"
    if not p.exists():
        print("[plot] no novelty CSV yet; skipping")
        return
    df = pd.read_csv(p)
    tm_col = "best_train_qtmscore" if "best_train_qtmscore" in df else "best_qtmscore"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(df[tm_col].fillna(0.0), bins=40, color="#4477aa")
    ax[0].axvline(0.5, color="crimson", ls="--", label="TM 0.5 (same fold)")
    ax[0].axvline(0.9, color="darkorange", ls="--", label="TM 0.9 (redundant)")
    ax[0].set(xlabel="nearest afdb-24M-rep TM-score", ylabel="Atlas structures",
              title="structural novelty vs our training reps")
    ax[0].legend()
    vc = df["verdict"].value_counts()
    ax[1].bar(vc.index, vc.values, color="#228833")
    ax[1].set(ylabel="count", title="verdict")
    for i, v in enumerate(vc.values):
        ax[1].text(i, v, f"{v}\n{v / len(df):.0%}", ha="center", va="bottom")
    fig.suptitle(f"Atlas structural novelty — {len(df):,} sampled structures "
                 f"vs 1.33M afdb-24M reps")
    fig.tight_layout()
    fig.savefig(PLOTS / "novelty.png", dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_composition()
    plot_pfam()
    plot_source()
    plot_afdb_coverage()
    plot_pdb()
    plot_leakage()
    plot_novelty()
    print(f"[plot] wrote PNGs to {PLOTS}")


if __name__ == "__main__":
    main()

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster-representative selection — stage 2 of the ESM Atlas distillation pipeline.

The funnel (``funnel.py``) keeps a large, permissive, quality-and-novelty-filtered
set (~250M with the default "current" params). This stage **clusters that set and
picks one (or a few) representative(s) per cluster**, which is where the final size
and diversity are controlled — following ESMFold2's own recipe (Biohub 2026, App.
A.2.9.1): cluster, then per cluster take the longest sequence and, among those, the
structure with the *smallest pLDDT standard deviation* (favours uniformly-confident
structures over high-mean-but-ragged ones).

This module is **clustering-method-agnostic**: it consumes a per-structure frame
that already carries a ``cluster_id`` and selects reps from it. The clustering
itself happens upstream (AWS-side, on the full survivor set) by one of:

- **Reuse the Atlas SAE clusters** (cheapest — precomputed): 7.7M clusters with
  ≥50 members, ~230M with ≥5 members. The member-size tier is the size dial.
- **MMseqs2 sequence clustering** of survivors at 40%/70% id (ESMFold2's choice;
  best for a sequence-diverse set).
- **Foldseek structural clustering** (most principled for a structure model; most
  compute).

`select_representatives` is a pure function so the same selection logic runs on the
sample (for validation) and on the full production frame.

Outputs: ``data/selected_reps.csv`` (chosen ``protein_hash``s + cluster + metrics)
and prints cluster/selection statistics.
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Selection orderings: (sort columns, ascending flags). The first row per cluster
# after sorting is the representative; reps_per_cluster takes the first N.
SELECT_BY = {
    # ESMFold2: longest sequence, then smallest per-residue pLDDT std-dev.
    "esmfold2": (["seq_len", "plddt_std"], [False, True]),
    "plddt_std": (["plddt_std"], [True]),          # most uniform confidence first
    "mean_plddt": (["mean_plddt"], [False]),       # highest mean confidence first
    "length": (["seq_len"], [False]),              # longest first
}


def select_representatives(
    frame: pd.DataFrame,
    *,
    reps_per_cluster: int = 1,
    select_by: str = "esmfold2",
    min_cluster_size: int = 1,
) -> pd.DataFrame:
    """Pick up to ``reps_per_cluster`` representatives from each cluster.

    ``frame`` must carry ``cluster_id`` plus whatever columns ``select_by`` needs
    (``seq_len``/``plddt_std``/``mean_plddt``). Clusters with fewer than
    ``min_cluster_size`` members are dropped entirely (singleton noise control).
    Returns the selected rows with a ``cluster_size`` column.
    """
    if select_by not in SELECT_BY:
        raise ValueError(f"select_by must be one of {sorted(SELECT_BY)}")
    cols, asc = SELECT_BY[select_by]
    df = frame.copy()
    sizes = df.groupby("cluster_id")["protein_hash"].transform("size")
    df["cluster_size"] = sizes
    df = df[df["cluster_size"] >= min_cluster_size]
    ranked = df.sort_values(cols, ascending=asc)
    selected = ranked.groupby("cluster_id", as_index=False, sort=False).head(
        reps_per_cluster)
    return selected.reset_index(drop=True)


def load_survivors(data_dir: Path, cluster_map: Path | None) -> pd.DataFrame:
    """Load funnel survivors and attach their cluster assignment.

    Survivors come from ``funnel_decisions.csv`` (``kept == True``). ``cluster_map``
    is a CSV with at least ``protein_hash,cluster_id`` (produced upstream by the
    chosen clustering). Without it, the module can't select — it errors with a
    pointer to how to build one.
    """
    dec = pd.read_csv(data_dir / "funnel_decisions.csv")
    surv = dec[dec["kept"]].copy()
    if cluster_map is None or not Path(cluster_map).exists():
        raise SystemExit(
            "No cluster_map. Selection needs a protein_hash->cluster_id CSV from "
            "the upstream clustering (Atlas SAE clusters, MMseqs2, or foldseek). "
            "Pass --cluster-map; see module docstring.")
    cm = pd.read_csv(cluster_map)[["protein_hash", "cluster_id"]]
    return surv.merge(cm, on="protein_hash", how="inner")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cluster-map", type=Path, default=None,
                    help="CSV: protein_hash,cluster_id from upstream clustering")
    ap.add_argument("--reps-per-cluster", type=int, default=1)
    ap.add_argument("--select-by", choices=sorted(SELECT_BY), default="esmfold2")
    ap.add_argument("--min-cluster-size", type=int, default=1,
                    help="drop clusters with fewer than this many survivors")
    args = ap.parse_args(argv)

    frame = load_survivors(DATA, args.cluster_map)
    selected = select_representatives(
        frame, reps_per_cluster=args.reps_per_cluster,
        select_by=args.select_by, min_cluster_size=args.min_cluster_size)
    selected.to_csv(DATA / "selected_reps.csv", index=False)

    n_clusters = frame["cluster_id"].nunique()
    print(f"survivors: {len(frame)} in {n_clusters} clusters")
    print(f"selected:  {len(selected)} reps "
          f"(<= {args.reps_per_cluster}/cluster, by {args.select_by}, "
          f"min_cluster_size {args.min_cluster_size})")
    print(f"wrote {DATA / 'selected_reps.csv'}")


if __name__ == "__main__":
    main()

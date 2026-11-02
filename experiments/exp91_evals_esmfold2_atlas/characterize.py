# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Characterize the ESM Atlas from the streamed sample + small metadata files.

Answers issue #91 Q1 ("what's in it") and the pfam dark-matter angle (Q2 prep):

- **Fold composition** (from ``data/atlas_sample.parquet`` +
  ``data/atlas_struct_sample.csv``): sequence-length / mean-pLDDT / pTM
  distributions, the high-confidence fraction (mean pLDDT > 0.7, expected
  ≈ 0.38 from the paper), and an explicit monomer check.
- **Pfam characterization** (from the 684 MB
  ``v1/clusters/data/representative_proteins.parquet``, 2 narrow columns):
  how many of the 7.7 M ≥50-member SAE clusters carry *no* characterized Pfam
  domain ("dark matter"), plus the ``naming_tier`` distribution.
- **Source-DB composition** (from one ``v1/shared_indexes/provenance_*.parquet``
  shard, ``source`` column only): the SPIRE/MGnify/UniParc/… split over the
  6.8 B sequences (one shard is hash-uniform, hence representative).

Writes CSV summaries to ``data/`` consumed by ``plot.py``. The two remote reads
(rep table ≈ 8 s, provenance source aggregate ≈ 110 s) are cached; pass
``--refresh`` to recompute.
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

import atlas_io

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

REP_TABLE_KEY = f"{atlas_io.ATLAS_BUCKET}/v1/clusters/data/representative_proteins.parquet"
PROVENANCE_URL = (
    f"https://{atlas_io.ATLAS_BUCKET}.s3.{atlas_io.ATLAS_REGION}.amazonaws.com"
    "/v1/shared_indexes/provenance_0.parquet"
)
PLDDT_HIGH = 0.70  # paper's "high confidence" threshold


def _quantiles(s: pd.Series, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)) -> dict:
    return {f"q{int(q * 100):02d}": float(s.quantile(q)) for q in qs}


def fold_composition() -> pd.DataFrame:
    """Length / pLDDT / pTM distributions + monomer + high-confidence fraction."""
    meta = pd.read_parquet(DATA / "atlas_sample.parquet")
    struct = pd.read_csv(DATA / "atlas_struct_sample.csv")
    rows = []
    for col in ["seq_len", "mean_plddt", "ptm"]:
        rows.append({"metric": col, "n": len(meta), "mean": float(meta[col].mean()),
                     **_quantiles(meta[col])})
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA / "fold_distributions.csv", index=False)

    headline = pd.DataFrame([{
        "n_meta_sample": len(meta),
        "n_struct_sample": len(struct),
        "frac_plddt_gt_0.7": float((meta["mean_plddt"] > PLDDT_HIGH).mean()),
        "frac_plddt_gt_0.7_paper": 418.5 / 1095.5,  # 418.5M of 1.0955B
        "median_seq_len": float(meta["seq_len"].median()),
        "median_mean_plddt": float(meta["mean_plddt"].median()),
        "median_ptm": float(meta["ptm"].median()),
        "monomer_fraction": float(struct["is_monomer"].mean()),
        "max_n_chains": int(struct["n_chains"].max()),
    }])
    headline.to_csv(DATA / "fold_headline.csv", index=False)
    print("[fold] headline:\n", headline.T.to_string(header=False))
    return headline


def pfam_characterization(refresh: bool) -> pd.DataFrame:
    """Pfam dark-matter + naming-tier distribution over the 7.7 M SAE clusters."""
    out = DATA / "pfam_characterization.csv"
    hist_out = DATA / "pfam_pct_hist.csv"
    if out.exists() and hist_out.exists() and not refresh:
        print("[pfam] cached")
        return pd.read_csv(out)
    s3 = pafs.S3FileSystem(anonymous=True, region=atlas_io.ATLAS_REGION)
    df = pq.read_table(
        REP_TABLE_KEY, columns=["cluster_pct_characterized", "naming_tier"],
        filesystem=s3,
    ).to_pandas()
    n = len(df)
    pct = df["cluster_pct_characterized"]
    summary = pd.DataFrame([{
        "n_clusters_ge50": n,
        "frac_fully_dark_pct0": float((pct == 0).mean()),
        "n_fully_dark_pct0": int((pct == 0).sum()),
        "frac_mostly_dark_lt50": float((pct < 50).mean()),
        "median_pct_characterized": float(pct.median()),
        "mean_pct_characterized": float(pct.mean()),
    }])
    summary.to_csv(out, index=False)
    # Histogram of pct_characterized (0..100) and naming_tier counts.
    hist, edges = np.histogram(pct, bins=20, range=(0, 100))
    pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist}
                 ).to_csv(hist_out, index=False)
    df["naming_tier"].value_counts().sort_index().rename("count").to_csv(
        DATA / "naming_tier_counts.csv")
    print("[pfam] summary:\n", summary.T.to_string(header=False))
    return summary


def source_composition(refresh: bool) -> pd.DataFrame:
    """SPIRE/MGnify/UniParc/… split over the 6.8 B sequences (one prov. shard)."""
    out = DATA / "source_composition.csv"
    if out.exists() and not refresh:
        print("[source] cached")
        return pd.read_csv(out)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    df = con.execute(
        f"SELECT source, count(*) AS n FROM read_parquet('{PROVENANCE_URL}') "
        "GROUP BY source ORDER BY n DESC"
    ).df()
    df["fraction"] = df["n"] / df["n"].sum()
    # SPIRE + MGnify are the metagenomic bulk.
    meta_frac = df.loc[df["source"].isin(["spire", "mgy"]), "fraction"].sum()
    df.attrs["metagenomic_fraction"] = meta_frac
    df.to_csv(out, index=False)
    print(f"[source] metagenomic (spire+mgy) fraction = {meta_frac:.3f}")
    print(df.to_string(index=False))
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true",
                    help="recompute cached remote reads (rep table, provenance)")
    ap.add_argument("--skip-source", action="store_true",
                    help="skip the ~110s provenance source aggregate")
    args = ap.parse_args(argv)

    DATA.mkdir(parents=True, exist_ok=True)
    fold_composition()
    pfam_characterization(args.refresh)
    if not args.skip_source:
        source_composition(args.refresh)


if __name__ == "__main__":
    main()

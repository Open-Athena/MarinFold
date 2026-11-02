# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""How much of the ESM Atlas overlaps AFDB and the PDB (issue #91 Q2).

The headline question is structural/sequence novelty of the Atlas against the
two reference universes we care about — AlphaFold DB and the experimental PDB —
not our bespoke eval set. Three complementary measurements:

1. **Atlas → AFDB, sequence.** AFDB folds UniProt, so a UniProt/UniRef match is
   a proxy for "this is already in AFDB". The Atlas cluster table carries a
   ``uniref_match_identity`` per ≥50-member SAE cluster; we bucket those to show
   how much of the Atlas is near-identical vs only a distant homolog vs has *no*
   UniProt match at all (genuinely novel sequence space).
2. **Atlas → AFDB, structure.** Foldseek TM of the decoded Atlas structure
   sample against the 1.33 M afdb-24M cluster representatives (exp41's published
   DB) — a fold-space representation of AFDB. (Run separately by ``novelty.py``;
   summarized here for the combined table.)
3. **Atlas → PDB, structure.** Foldseek TM of the Atlas sample against Foldseek's
   prebuilt **PDB** database (``foldseek databases PDB``): what fraction of
   sampled Atlas folds have a near-match among experimentally solved structures.

Writes ``data/afdb_uniref_coverage.csv``, ``data/atlas_vs_pdb.csv``,
``data/atlas_vs_pdb_summary.csv``, and a combined ``data/overlap_summary.csv``.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

import atlas_io

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP41 = HERE.parent / "exp41_evals_foldseek_train_similarity"
sys.path.insert(0, str(EXP41))  # reuse the Foldseek binary wrapper + m8 parser
import query_similarity as qs  # noqa: E402

REP_TABLE_KEY = f"{atlas_io.ATLAS_BUCKET}/v1/clusters/data/representative_proteins.parquet"


def afdb_sequence_coverage() -> pd.DataFrame:
    """Atlas → UniProt/AFDB sequence coverage from the cluster uniref match."""
    s3 = pafs.S3FileSystem(anonymous=True, region=atlas_io.ATLAS_REGION)
    df = pq.read_table(
        REP_TABLE_KEY,
        columns=["uniref_match_accession", "uniref_match_identity"], filesystem=s3,
    ).to_pandas()
    n = len(df)
    has = df["uniref_match_accession"].notna() & (
        df["uniref_match_accession"].astype(str) != "")
    idt = df["uniref_match_identity"]  # 0–100
    rows = [{
        "n_clusters_ge50": n,
        "frac_no_uniprot_match": float((~has).mean()),
        "frac_uniref_id_ge_30": float((idt >= 30).mean()),
        "frac_uniref_id_ge_50": float((idt >= 50).mean()),
        "frac_uniref_id_ge_70": float((idt >= 70).mean()),
        "frac_uniref_id_ge_90": float((idt >= 90).mean()),
        "frac_uniref_id_ge_95": float((idt >= 95).mean()),
        "median_uniref_id_where_matched": float(idt[has].median()),
    }]
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "afdb_uniref_coverage.csv", index=False)
    print("[afdb-seq] ", out.T.to_string(header=False))
    return out


def atlas_vs_pdb(candidate_dir: Path, pdb_db: Path, fold_tm: float,
                 redundant_tm: float) -> pd.DataFrame:
    """Foldseek the Atlas structure sample against the prebuilt PDB DB."""
    out_m8 = DATA / "atlas_vs_pdb.m8"
    qs.easy_search(candidate_dir, pdb_db, out_m8, DATA / "_tmp_pdb",
                   alignment_type=1, max_seqs=50)
    hits = qs.parse_m8(out_m8)
    # Best hit per query candidate by qtmscore.
    hits["stem"] = hits["query"].map(qs._strip_struct_ext if hasattr(qs, "_strip_struct_ext")
                                     else (lambda s: str(s).rsplit(".", 1)[0]))
    best = (hits.sort_values("qtmscore", ascending=False)
                .groupby("stem", as_index=False).first())
    best.to_csv(DATA / "atlas_vs_pdb.csv", index=False)
    n_cand = len(list(candidate_dir.glob("*.cif"))) + len(list(candidate_dir.glob("*.pdb")))
    summary = pd.DataFrame([{
        "n_candidates": n_cand,
        "n_with_pdb_hit": len(best),
        "frac_pdb_fold_match_tm_ge_0.5": float((best["qtmscore"] >= fold_tm).sum()) / max(1, n_cand),
        "frac_pdb_near_identical_tm_ge_0.9": float((best["qtmscore"] >= redundant_tm).sum()) / max(1, n_cand),
        "median_best_pdb_tm": float(best["qtmscore"].median()),
        "median_best_pdb_fident": float(best["fident"].median()),
    }])
    summary.to_csv(DATA / "atlas_vs_pdb_summary.csv", index=False)
    print("[atlas-vs-pdb] ", summary.T.to_string(header=False))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate-dir", type=Path, default=DATA / "struct_sample")
    ap.add_argument("--pdb-db", type=Path, default=HERE / "pdb_db" / "pdb")
    ap.add_argument("--fold-tm", type=float, default=0.5)
    ap.add_argument("--redundant-tm", type=float, default=0.9)
    ap.add_argument("--skip-pdb", action="store_true")
    args = ap.parse_args(argv)

    DATA.mkdir(parents=True, exist_ok=True)
    afdb_sequence_coverage()
    if not args.skip_pdb:
        if not args.pdb_db.with_suffix(".index").exists() and not args.pdb_db.exists():
            sys.exit(f"PDB DB missing at {args.pdb_db}; run: foldseek databases PDB "
                     f"{args.pdb_db} tmp")
        atlas_vs_pdb(args.candidate_dir, args.pdb_db, args.fold_tm, args.redundant_tm)


if __name__ == "__main__":
    main()

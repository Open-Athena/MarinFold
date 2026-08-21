# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 2b — the structural drop list for the AFDB arm, via Foldseek.

This is the axis #41 said was the one that matters and nobody ever filtered on.
Of the 99 FoldBench monomers with a same-fold-or-closer training match, **65 sit
below 30 % sequence identity** — a sequence-only filter clears them as novel
while structurally they are near-duplicates. #213 confirms it from the other
side: of its 231 sequence-novel eval proteins, 194 are still ``same_fold`` or
``redundant`` against the AFDB training folds.

For the AFDB arm this is nearly free, which is the whole reason to do it first:
``afdb-24M`` is already Foldseek-clustered, every corpus row carries its
``struct_cluster_id``, and #41 built and published a Foldseek database of the
1,331,330 cluster representatives with a rep → split manifest. So the query is
554 structures against 1.33 M, and a hit purges a **whole cluster** — which is
exp65's ``eval-strategy-summary`` Step 3, and the only version that actually
makes train and eval fold-disjoint rather than merely sequence-disjoint.

Two properties of that shortcut have to be stated rather than assumed:

* **The search is at representative granularity.** We have structures for the
  1.33 M representatives, not for all 4.13 M documents, so a cluster is judged
  by its representative. A non-representative member could in principle exceed
  the threshold when its representative does not. AFDB clusters *are* Foldseek
  clusters, so members are structurally close to their representative by
  construction, but this is an approximation and not an exact per-row TM.
* **Purging is at cluster granularity.** Both Tier B and Tier C remove entire
  clusters here. That is deliberate for Tier C (fold disjointness is a
  statement about clusters) and conservative for Tier B.

``qtmscore`` — TM normalised by the *eval* structure — drives the verdict, the
same convention exp41/exp65 used for the ``fold_verdict`` labels this experiment
is measured against. ``max(qtmscore, ttmscore)`` is carried alongside as the
conservative sensitivity column, since a short eval protein embedded in a long
training fold scores low on one normalisation and high on the other.

    uv run python structure_droplist.py --work /data/exp225_decontam
    uv run python structure_droplist.py --work /data/exp225_decontam --skip-search
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from decontam_lib import (
    ARM_AFDB,
    CORPORA,
    REFERENCE_VERSION,
    STRUCT_FOLD_TM,
    STRUCT_REDUNDANT_TM,
    TIER_B,
    TIER_C,
    ensure_foldseek,
    run,
)

HERE = Path(__file__).resolve().parent

#: exp41's published representative DB — 1,331,330 AFDB cluster representatives
#: plus a ``representative_id -> split`` manifest, where ``representative_id``
#: *is* the ``struct_cluster_id`` the corpus rows carry.
#: ``hf buckets sync hf://buckets/silterra/afdb-24M-foldseek-train-reps <dir>``
DEFAULT_REPS_DIR = Path("/data/exp225_decontam/afdb_reps_db")

FORMAT = "query,target,alntmscore,qtmscore,ttmscore,lddt,fident,alnlen,evalue"
FIELDS = FORMAT.split(",")

STRUCT_EXTENSIONS = (".cif", ".mmcif", ".pdb", ".ent")


def normalize_query(name: str, known: set[str]) -> str:
    """Map a Foldseek query entry name back to a reference key.

    Foldseek names an entry after the file and appends ``_<chain>`` when the
    structure has named chains — *after* the extension, so the raw name looks
    like ``denovo_pdb__1mj0.cif_A``. Our reference keys already contain
    underscores and frequently end in ``_<chain>`` themselves
    (``foldbench100__5sbj_A``), so a trailing token is dropped only while the
    result is not yet a known key, never speculatively: over-stripping
    ``foldbench100__5sbj_A`` to ``foldbench100__5sbj`` would make that protein
    vanish from the drop list. An unresolvable name raises for the same reason.
    """
    candidate = name
    while True:
        base = candidate
        for suffix in STRUCT_EXTENSIONS:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base in known:
            return base
        if "_" not in candidate:
            raise ValueError(f"Foldseek query {name!r} does not resolve to a reference key")
        candidate = candidate.rsplit("_", 1)[0]


def search(
    work: Path, structures: Path, target_db: Path, *, max_seqs: int, evalue: float,
    threads: int,
) -> Path:
    """``foldseek easy-search`` the 554 against the representative DB (TM mode)."""
    foldseek = ensure_foldseek()
    print(f"[foldseek] binary: {foldseek}", flush=True)
    out = work / "structure_hits_afdb.tsv"
    tmp = work / "foldseek_tmp"
    t0 = time.time()
    run(
        [
            foldseek, "easy-search", structures, target_db, out, tmp,
            "--alignment-type", 1,          # TM-align mode
            "--format-output", FORMAT,
            "--max-seqs", max_seqs,
            "-e", evalue,
            "--threads", threads,
        ]
    )
    print(f"[foldseek] {out} ({out.stat().st_size / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f}s)", flush=True)
    return out


def reduce_hits(
    hits_path: Path, reps_manifest: Path, known: set[str], *, chunksize: int = 5_000_000
) -> tuple[pd.DataFrame, dict]:
    """One row per training cluster: its best TM to any eval structure.

    Streamed in chunks and folded into a per-target dict rather than sorted
    whole. Foldseek is asked for every alignment the prefilter survives here,
    which is tens of millions of rows over 1.33 M targets; materialising that
    as one frame and sorting it would cost tens of GB for a reduction that
    needs one record per target.

    Only ``split == train`` representatives survive: the corpus is built from
    ``afdb-24M``'s train split, so a val/test representative is not something
    that can be purged from it (and #41's clusters are split-consistent, so a
    train cluster never hides behind a val representative).
    """
    manifest = pd.read_csv(reps_manifest, usecols=["representative_id", "split"])
    train_reps = set(manifest.loc[manifest["split"] == "train", "representative_id"])
    print(f"[reduce] {len(train_reps):,} train representatives of "
          f"{len(manifest):,} total", flush=True)

    # target -> best record. `best_qtm` drives the verdict; `max_tm` is tracked
    # independently because the conservative normalisation can peak on a
    # different alignment than the query-normalised one.
    best: dict[str, dict] = {}
    query_cache: dict[str, str] = {}
    n_rows = n_train_rows = 0
    t0 = time.time()

    reader = pd.read_csv(hits_path, sep="\t", names=FIELDS, chunksize=chunksize)
    for chunk in reader:
        n_rows += len(chunk)
        chunk = chunk[chunk["target"].isin(train_reps)]
        n_train_rows += len(chunk)
        chunk = chunk.assign(max_tm=chunk[["qtmscore", "ttmscore"]].max(axis=1))
        for target, qtm, max_tm, query, fident, alntm, lddt in zip(
            chunk["target"], chunk["qtmscore"], chunk["max_tm"], chunk["query"],
            chunk["fident"], chunk["alntmscore"], chunk["lddt"],
        ):
            current = best.get(target)
            if current is None:
                key = query_cache.get(query)
                if key is None:
                    key = query_cache.setdefault(query, normalize_query(str(query), known))
                best[target] = {
                    "struct_cluster_id": target, "best_qtm": qtm, "max_tm": max_tm,
                    "nearest_eval_key": key, "fident_at_best_qtm": fident,
                    "alntmscore": alntm, "lddt": lddt,
                }
                continue
            if max_tm > current["max_tm"]:
                current["max_tm"] = max_tm
            if qtm > current["best_qtm"]:
                key = query_cache.get(query)
                if key is None:
                    key = query_cache.setdefault(query, normalize_query(str(query), known))
                current.update(best_qtm=qtm, nearest_eval_key=key,
                               fident_at_best_qtm=fident, alntmscore=alntm, lddt=lddt)
        print(f"[reduce] {n_rows:,} rows read, {len(best):,} train clusters, "
              f"{time.time() - t0:.0f}s", flush=True)

    stats = {"n_alignments": n_rows, "n_alignments_vs_train_reps": n_train_rows}
    return pd.DataFrame(list(best.values())), stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work", type=Path, default=Path("/data/exp225_decontam"))
    ap.add_argument("--structures", type=Path,
                    default=Path("/data/exp225_decontam/eval_structures"))
    ap.add_argument("--reps-dir", type=Path, default=DEFAULT_REPS_DIR)
    ap.add_argument("--reference", type=Path, default=HERE / "data/reference/eval_structures.csv")
    ap.add_argument("--max-seqs", type=int, default=1_000_000)
    ap.add_argument("--evalue", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--skip-search", action="store_true")
    ap.add_argument("--droplist-out", type=Path, default=None,
                    help="default: <work>/droplist_structure_afdb.parquet")
    ap.add_argument("--summary-out", type=Path,
                    default=HERE / "data/structure_droplist_summary.csv")
    ap.add_argument("--provenance-out", type=Path,
                    default=HERE / "data/structure_droplist.provenance.json")
    args = ap.parse_args()

    reference = pd.read_csv(args.reference)
    known = set(reference["key"])
    droplist_out = args.droplist_out or args.work / "droplist_structure_afdb.parquet"

    hits_path = args.work / "structure_hits_afdb.tsv"
    if not args.skip_search:
        hits_path = search(
            args.work, args.structures, args.reps_dir / "db" / "targetDB",
            max_seqs=args.max_seqs, evalue=args.evalue, threads=args.threads,
        )
    elif not hits_path.exists():
        raise SystemExit(f"--skip-search but {hits_path} does not exist")

    clusters, hit_stats = reduce_hits(hits_path, args.reps_dir / "reps_manifest.csv", known)
    droplist_out.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_parquet(droplist_out, compression="zstd", index=False)
    print(f"[clusters] {len(clusters):,} train clusters hit -> {droplist_out}", flush=True)

    corpus = CORPORA[ARM_AFDB]
    rows = []
    for tier, threshold in ((TIER_B, STRUCT_REDUNDANT_TM), (TIER_C, STRUCT_FOLD_TM)):
        for field, label in (("best_qtm", "qtm"), ("max_tm", "max_qt_tm")):
            selected = clusters[clusters[field] >= threshold]
            rows.append(
                {
                    "tier": tier,
                    "tm_field": label,
                    "tm_threshold": threshold,
                    "n_clusters_purged": len(selected),
                    "n_eval_proteins_responsible": selected["nearest_eval_key"].nunique(),
                    "reference_version": REFERENCE_VERSION,
                }
            )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"[purge] tier {row['tier']} ({row['tm_field']} >= {row['tm_threshold']}): "
              f"{row['n_clusters_purged']:,} clusters", flush=True)

    queries_with_hits = clusters["nearest_eval_key"].nunique()
    args.provenance_out.write_text(
        json.dumps(
            {
                "reference_version": REFERENCE_VERSION,
                "arm": ARM_AFDB,
                "corpus_documents": corpus.n_documents,
                "reps_db": str(args.reps_dir),
                "alignment_type": "1 (TM-align)",
                "max_seqs": args.max_seqs,
                "reporting_evalue": args.evalue,
                "thresholds": {
                    "redundant_tm": STRUCT_REDUNDANT_TM,
                    "fold_tm": STRUCT_FOLD_TM,
                },
                "n_alignments": hit_stats["n_alignments"],
                "n_alignments_vs_train_reps": hit_stats["n_alignments_vs_train_reps"],
                "n_train_clusters_with_any_hit": len(clusters),
                "n_eval_proteins_with_any_train_hit": int(queries_with_hits),
                "droplist": str(droplist_out),
                "purge_counts": {
                    f"{row['tier']}_{row['tm_field']}": row["n_clusters_purged"]
                    for row in rows
                },
                "nearest_eval_dataset_counts": dict(
                    Counter(
                        key.partition("__")[0] for key in clusters["nearest_eval_key"]
                    )
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[provenance] -> {args.provenance_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

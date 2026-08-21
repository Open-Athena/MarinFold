# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 3 — what each tier costs, priced before anything is retrained.

The issue's question is not "can we decontaminate" — of course we can — but
"what does it cost", and specifically whether Tier C is affordable at all. #41
warned that nearly every FoldBench fold is represented somewhere in AFDB train;
if that is true at cluster granularity then the fold-level purge deletes a large
fraction of the corpus, and the honest fix is a better eval set rather than a
smaller training set. That is H0, and this script is what distinguishes it from
H1.

Four tables come out:

* ``survival_by_tier.csv`` — documents kept and dropped per (arm, tier).
* ``survival_by_axis.csv`` — the **sequence-only / structure-only / both**
  decomposition. The structure-only column is the direct answer to "was #41's
  warning worth acting on": it counts the training documents a sequence filter
  would have kept and a structural one removes. If it is small, #91's
  sequence-only funnel was defensible after all; if it is large, every corpus
  we have is contaminated on an axis nobody filtered.
* ``cluster_purge.csv`` — how the fold-level purge distributes over clusters,
  since a purge that removes a handful of enormous clusters is a very different
  proposition from one that removes many small ones.
* ``tier_scope.csv`` — the structural tiers priced separately for the designed
  and the natural halves of the eval set, because the two cost wildly different
  amounts per query and only one of them is arguably worth paying for.

**The ESM-Atlas arm has no structural tier here, and that is reported rather
than defaulted to zero.** Its 66.76 M rows are clustered at 40 % *sequence*
identity only, so there is no structural database to query and no
``struct_cluster_id`` to purge; building one is the ~$1k Foldseek job the issue
gates on this table.

    uv run python survival.py --work /data/exp225_decontam
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from decontam_lib import (
    ARM_AFDB,
    ARM_ESM,
    ARMS,
    CORPORA,
    REFERENCE_VERSION,
    STRUCT_FOLD_TM,
    STRUCT_REDUNDANT_TM,
    TIER_A,
    TIER_B,
    TIER_C,
    TIER_LABELS,
    TIER_RULES,
)

HERE = Path(__file__).resolve().parent

#: Which TM threshold each structural tier applies. Tier A has none.
TIER_TM = {TIER_A: None, TIER_B: STRUCT_REDUNDANT_TM, TIER_C: STRUCT_FOLD_TM}

#: exp65's de novo set — the designed half of the eval set (396 of 554).
DESIGNED_DATASET = "denovo_pdb"


def load_afdb(work: Path, tm_field: str) -> pd.DataFrame:
    """The AFDB corpus index, annotated with both axes' drop flags per row."""
    index = pd.read_parquet(work / "index_afdb.parquet",
                            columns=["entry_id", "struct_cluster_id", "shard", "row"])
    sequence = pd.read_parquet(work / "droplist_sequence.parquet", columns=["arm", "entry_id"])
    dropped_ids = set(sequence.loc[sequence["arm"] == ARM_AFDB, "entry_id"])
    index["seq_dropped"] = index["entry_id"].isin(dropped_ids)

    clusters = pd.read_parquet(work / "droplist_structure_afdb.parquet")
    for tier, threshold in TIER_TM.items():
        if threshold is None:
            continue
        purged = set(clusters.loc[clusters[tm_field] >= threshold, "struct_cluster_id"])
        index[f"struct_dropped_{tier}"] = index["struct_cluster_id"].isin(purged)
    return index


def survival_rows(index: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Per-tier survival and the per-axis decomposition, for the AFDB arm."""
    corpus = CORPORA[ARM_AFDB]
    total = len(index)
    by_tier, by_axis = [], []
    for tier in (TIER_A, TIER_B, TIER_C):
        structural = (
            index[f"struct_dropped_{tier}"] if TIER_TM[tier] is not None
            else pd.Series(False, index=index.index)
        )
        dropped = index["seq_dropped"] | structural
        n_dropped = int(dropped.sum())
        by_tier.append(
            {
                "arm": ARM_AFDB,
                "tier": tier,
                "tier_label": TIER_LABELS[tier],
                "rule": TIER_RULES[tier],
                "n_documents": total,
                "n_dropped": n_dropped,
                "pct_dropped": round(100 * n_dropped / total, 4),
                "n_surviving": total - n_dropped,
                "n_clusters_before": int(index["struct_cluster_id"].nunique()),
                "n_clusters_after": int(index.loc[~dropped, "struct_cluster_id"].nunique()),
                "measurable": True,
            }
        )
        by_axis.append(
            {
                "arm": ARM_AFDB,
                "tier": tier,
                "sequence_only": int((index["seq_dropped"] & ~structural).sum()),
                "structure_only": int((~index["seq_dropped"] & structural).sum()),
                "both": int((index["seq_dropped"] & structural).sum()),
                "kept": total - n_dropped,
            }
        )

    # ESM-Atlas: Tier A is real; the structural tiers are unmeasured, not zero.
    esm = CORPORA[ARM_ESM]
    for tier in (TIER_A, TIER_B, TIER_C):
        measurable = tier == TIER_A
        by_tier.append(
            {
                "arm": ARM_ESM,
                "tier": tier,
                "tier_label": TIER_LABELS[tier],
                "rule": TIER_RULES[tier],
                "n_documents": esm.n_documents,
                "n_dropped": None,
                "pct_dropped": None,
                "n_surviving": None,
                "n_clusters_before": None,
                "n_clusters_after": None,
                "measurable": measurable,
            }
        )
    return by_tier, by_axis


def fill_esm_tier_a(by_tier: list[dict], work: Path) -> None:
    """Tier A for ESM-Atlas comes straight off the sequence drop list."""
    sequence = pd.read_parquet(work / "droplist_sequence.parquet", columns=["arm"])
    n_dropped = int((sequence["arm"] == ARM_ESM).sum())
    esm = CORPORA[ARM_ESM]
    for row in by_tier:
        if row["arm"] == ARM_ESM and row["tier"] == TIER_A:
            row["n_dropped"] = n_dropped
            row["pct_dropped"] = round(100 * n_dropped / esm.n_documents, 4)
            row["n_surviving"] = esm.n_documents - n_dropped


def cluster_purge_rows(work: Path, index: pd.DataFrame, tm_field: str) -> list[dict]:
    """How the fold-level purge distributes over clusters, per structural tier."""
    sizes = index.groupby("struct_cluster_id").size()
    clusters = pd.read_parquet(work / "droplist_structure_afdb.parquet")
    rows = []
    for tier in (TIER_B, TIER_C):
        threshold = TIER_TM[tier]
        purged = clusters.loc[clusters[tm_field] >= threshold, "struct_cluster_id"]
        in_corpus = sizes.reindex(purged.unique()).dropna()
        rows.append(
            {
                "tier": tier,
                "tm_field": tm_field,
                "tm_threshold": threshold,
                "n_clusters_hit": int(purged.nunique()),
                "n_clusters_present_in_corpus": int(len(in_corpus)),
                "n_documents_purged": int(in_corpus.sum()),
                "median_cluster_size": float(in_corpus.median()) if len(in_corpus) else 0.0,
                "max_cluster_size": int(in_corpus.max()) if len(in_corpus) else 0,
            }
        )
    return rows


def scope_rows(work: Path, index: pd.DataFrame, reference: Path, tm_field: str) -> list[dict]:
    """What the structural tiers cost when scoped to part of the eval set.

    The 554 are 396 de novo designs and 158 natural proteins, and the two behave
    completely differently as fold-purge queries. A designed protein is usually
    a small idealised bundle, which is the same fold as an enormous share of
    AFDB — so it purges far more of the corpus per query than a natural protein
    does, for a leakage risk that is arguably the smallest in the set (#213's
    homology-free subset is 80 % designs precisely because designs have no
    evolutionary relatives to leak through).

    Splitting the purge by scope is what turns "Tier C costs a third of the
    corpus" into an actionable statement, because the two halves of that third
    are not equally worth paying for.
    """
    reference_table = pd.read_csv(reference)[["key", "dataset"]]
    designed = set(reference_table.loc[reference_table["dataset"] == DESIGNED_DATASET, "key"])

    clusters = pd.read_parquet(work / "droplist_structure_afdb.parquet")
    clusters["designed"] = clusters["nearest_eval_key"].isin(designed)
    sizes = index.groupby("struct_cluster_id").size()
    total = len(index)

    scopes = {
        "all_554": clusters,
        "natural_158": clusters[~clusters["designed"]],
        "designed_396": clusters[clusters["designed"]],
    }
    rows = []
    for scope, subset in scopes.items():
        for tier in (TIER_B, TIER_C):
            threshold = TIER_TM[tier]
            purged = subset.loc[subset[tm_field] >= threshold, "struct_cluster_id"].unique()
            present = sizes.reindex(purged).dropna()
            rows.append(
                {
                    "eval_scope": scope,
                    "tier": tier,
                    "tm_threshold": threshold,
                    "n_clusters_purged": int(len(purged)),
                    "n_documents_purged": int(present.sum()),
                    "pct_of_corpus": round(100 * present.sum() / total, 3),
                }
            )
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[survival] {len(rows)} rows -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work", type=Path, default=Path("/data/exp225_decontam"))
    ap.add_argument("--tm-field", choices=("best_qtm", "max_tm"), default="best_qtm",
                    help="best_qtm normalises TM by the eval structure, matching the "
                         "fold_verdict convention of #41/#65; max_tm is the conservative "
                         "max over both normalisations")
    ap.add_argument("--by-tier-out", type=Path, default=HERE / "data/survival_by_tier.csv")
    ap.add_argument("--by-axis-out", type=Path, default=HERE / "data/survival_by_axis.csv")
    ap.add_argument("--cluster-out", type=Path, default=HERE / "data/cluster_purge.csv")
    ap.add_argument("--scope-out", type=Path, default=HERE / "data/tier_scope.csv")
    ap.add_argument("--reference", type=Path,
                    default=HERE / "data/reference/eval_structures.csv")
    ap.add_argument("--provenance-out", type=Path, default=HERE / "data/survival.provenance.json")
    args = ap.parse_args()

    index = load_afdb(args.work, args.tm_field)
    by_tier, by_axis = survival_rows(index)
    fill_esm_tier_a(by_tier, args.work)

    write_csv(by_tier, args.by_tier_out)
    write_csv(by_axis, args.by_axis_out)
    write_csv(cluster_purge_rows(args.work, index, args.tm_field), args.cluster_out)
    scopes = scope_rows(args.work, index, args.reference, args.tm_field)
    write_csv(scopes, args.scope_out)

    for row in by_tier:
        if row["n_dropped"] is None:
            print(f"  {row['arm']:<10} tier {row['tier']}: NOT MEASURABLE "
                  "(no structural database for this arm)", flush=True)
            continue
        print(f"  {row['arm']:<10} tier {row['tier']}: "
              f"{row['n_dropped']:>9,} / {row['n_documents']:>10,} dropped "
              f"({row['pct_dropped']:6.3f}%)", flush=True)
    for row in by_axis:
        print(f"  {row['arm']:<10} tier {row['tier']} axes: "
              f"sequence-only {row['sequence_only']:,}, "
              f"structure-only {row['structure_only']:,}, both {row['both']:,}", flush=True)
    for row in scopes:
        print(f"  scope {row['eval_scope']:<13} tier {row['tier']}: "
              f"{row['n_documents_purged']:>9,} docs ({row['pct_of_corpus']:6.3f}%)", flush=True)

    args.provenance_out.write_text(
        json.dumps(
            {
                "reference_version": REFERENCE_VERSION,
                "tm_field": args.tm_field,
                "tm_thresholds": {t: TIER_TM[t] for t in TIER_TM},
                "arms_with_a_structural_database": [ARM_AFDB],
                "arms_without_one": [ARM_ESM],
                "inputs": {
                    "corpus_index": str(args.work / "index_afdb.parquet"),
                    "sequence_droplist": str(args.work / "droplist_sequence.parquet"),
                    "structure_droplist": str(args.work / "droplist_structure_afdb.parquet"),
                },
                "corpus_documents": {arm: CORPORA[arm].n_documents for arm in ARMS},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[provenance] -> {args.provenance_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

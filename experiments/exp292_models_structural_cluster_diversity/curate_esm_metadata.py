"""Quality- and sequence-filter one located ESM production shard.

Only lightweight Atlas columns are read here. Clusters with at most three
surviving omitted members are selected immediately; larger clusters are emitted
as a structural-ranking queue. Coordinate blobs are left for that smaller next
stage.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from build_esm_plan import require_source_region
from locate_esm_rows import ATLAS_STORAGE, ATLAS_URI, ATLAS_VERSION
from production_policy import select_three
from sequence_exclusion import (
    MMSEQS_SHA256,
    MMSEQS_VERSION,
    read_fasta,
    run_sequence_screen,
)
from structure_audit import STANDARD_AMINO_ACIDS

MIN_PLDDT = 0.8
MIN_PTM = 0.5
MIN_LENGTH = 60
MAX_LENGTH = 1000


def fetch_metadata(located: pa.Table, batch_size: int) -> list[dict]:
    """Fetch source sequence/confidence by exact row ID and verify every hash."""
    records = sorted(located.to_pylist(), key=lambda row: row["row_index"])
    dataset = lance.dataset(
        ATLAS_URI, version=ATLAS_VERSION, storage_options=ATLAS_STORAGE
    )
    output = []
    for offset in range(0, len(records), batch_size):
        planned = records[offset : offset + batch_size]
        table = dataset.take(
            [row["row_index"] for row in planned],
            columns=["protein_hash", "sequence", "mean_plddt", "ptm"],
        )
        source = table.to_pylist()
        if len(source) != len(planned):
            raise ValueError("Atlas metadata take returned an incomplete batch")
        for plan_row, source_row in zip(planned, source, strict=True):
            if source_row["protein_hash"] != plan_row["protein_hash"]:
                raise ValueError(
                    f"Atlas row drift at {plan_row['row_index']}: "
                    f"{source_row['protein_hash']} != {plan_row['protein_hash']}"
                )
            sequence = source_row["sequence"]
            if hashlib.md5(sequence.encode()).hexdigest() != source_row["protein_hash"]:
                raise ValueError(
                    f"Atlas sequence does not match hash {source_row['protein_hash']}"
                )
            output.append(
                {
                    **plan_row,
                    "entry_id": source_row["protein_hash"],
                    "sequence": sequence,
                    "seq_len": len(sequence),
                    "global_plddt": float(source_row["mean_plddt"]) * 100,
                    "ptm": float(source_row["ptm"]),
                    "struct_cluster_id": plan_row["cluster_id"],
                    "source": "esmfold2",
                    "split": "train",
                }
            )
        print(
            f"Atlas metadata: {min(offset + batch_size, len(records)):,}/{len(records):,}",
            flush=True,
        )
    return output


def quality_filter(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply source integrity and candidate quality rules cluster by cluster."""
    groups = {}
    for row in rows:
        groups.setdefault(row["struct_cluster_id"], []).append(row)
    kept = []
    rejected = []
    for cluster_id, members in groups.items():
        anchors = [row for row in members if row["is_anchor"]]
        if len(anchors) != 1:
            raise ValueError(f"{cluster_id}: expected one current ESM anchor")
        anchor = anchors[0]
        anchor_noncanonical = bool(set(anchor["sequence"]) - STANDARD_AMINO_ACIDS)
        if anchor_noncanonical:
            rejected.extend(
                {**row, "rejection_reason": "noncanonical_anchor_cluster"}
                for row in members
            )
            continue
        if int(anchor["seq_len"]) != int(anchor["anchor_seq_len"]):
            raise ValueError(f"{cluster_id}: canonical anchor length changed")
        if not (
            math.isfinite(anchor["global_plddt"])
            and math.isfinite(anchor["ptm"])
            and anchor["global_plddt"] >= MIN_PLDDT * 100
            and anchor["ptm"] >= MIN_PTM
        ):
            rejected.extend(
                {**row, "rejection_reason": "anchor_source_quality"} for row in members
            )
            continue
        kept.append(anchor)
        for candidate in (row for row in members if not row["is_anchor"]):
            reason = None
            if set(candidate["sequence"]) - STANDARD_AMINO_ACIDS:
                reason = "noncanonical_candidate"
            elif not MIN_LENGTH <= int(candidate["seq_len"]) <= MAX_LENGTH:
                reason = "candidate_length"
            elif not math.isfinite(candidate["global_plddt"]):
                reason = "candidate_plddt_nonfinite"
            elif candidate["global_plddt"] < MIN_PLDDT * 100:
                reason = "candidate_plddt"
            elif not math.isfinite(candidate["ptm"]):
                reason = "candidate_ptm_nonfinite"
            elif candidate["ptm"] < MIN_PTM:
                reason = "candidate_ptm"
            if reason:
                rejected.append({**candidate, "rejection_reason": reason})
            else:
                kept.append(candidate)
    return kept, rejected


def partition_after_screen(
    quality_rows: list[dict], excluded: dict[str, dict]
) -> tuple[list[dict], list[dict], list[dict]]:
    """Select no-choice clusters and queue only clusters needing structure ranks."""
    groups = {}
    sequence_rejections = []
    for row in quality_rows:
        if not row["is_anchor"] and row["entry_id"] in excluded:
            sequence_rejections.append(
                {
                    **row,
                    **excluded[row["entry_id"]],
                    "rejection_reason": "heldout_sequence",
                }
            )
            continue
        groups.setdefault(row["struct_cluster_id"], []).append(row)
    selected = []
    structural_queue = []
    for members in groups.values():
        candidates = [row for row in members if not row["is_anchor"]]
        if not candidates:
            continue
        if len(candidates) <= 3:
            selected.extend(select_three(members, []))
        else:
            structural_queue.extend(members)
    return selected, structural_queue, sequence_rejections


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write nonempty row dictionaries with Zstandard compression."""
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def main() -> None:
    """Curate one located shard and publish its two disjoint output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--located", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--mmseqs-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--take-batch", type=int, default=4096)
    parser.add_argument("--max-clusters", type=int)
    args = parser.parse_args()
    require_source_region()
    if hashlib.sha256(args.mmseqs_archive.read_bytes()).hexdigest() != MMSEQS_SHA256:
        raise ValueError("Unexpected frozen MMseqs2 archive")
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    located = pq.read_table(args.located)
    if args.max_clusters is not None:
        cluster_ids = sorted(set(located["cluster_id"].to_pylist()))[
            : args.max_clusters
        ]
        located = located.filter(
            pc.is_in(located["cluster_id"], value_set=pa.array(cluster_ids))
        )
    fetched = fetch_metadata(located, args.take_batch)
    quality, quality_rejections = quality_filter(fetched)
    candidates = [row for row in quality if not row["is_anchor"]]
    references = read_fasta(args.reference)
    excluded, screen_stats = run_sequence_screen(
        candidates, references, args.mmseqs, args.work / "sequence-screen", args.threads
    )
    selected, queue, sequence_rejections = partition_after_screen(quality, excluded)
    write_rows(args.output / "selected_without_alignment.parquet", selected)
    write_rows(args.output / "structural_queue.parquet", queue)
    write_rows(args.output / "quality_rejections.parquet", quality_rejections)
    write_rows(args.output / "sequence_rejections.parquet", sequence_rejections)
    summary = {
        "status": "complete",
        "located_rows": located.num_rows,
        "quality_rows": len(quality),
        "quality_rejections": len(quality_rejections),
        "sequence_rejections": len(sequence_rejections),
        "selected_without_alignment": len(selected),
        "structural_queue_rows": len(queue),
        "structural_queue_clusters": len(
            {row["struct_cluster_id"] for row in queue}
        ),
        "atlas_uri": ATLAS_URI,
        "atlas_version": ATLAS_VERSION,
        "mmseqs_archive_sha256": MMSEQS_SHA256,
        "expected_mmseqs_version": MMSEQS_VERSION,
        "sequence_screen": screen_stats,
        "references": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in args.reference
        ],
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Structurally rank ESM candidates only where a cluster has a choice.

Each input cluster has one retained anchor and at least four quality- and
sequence-passing candidates. Coordinate blobs are fetched by pinned Atlas row
ID, decoded with exp91's atom37 checks, and compared lazily: candidate-to-anchor
first, then only candidate-to-addition pairs needed by greedy selection.
"""

import argparse
import hashlib
import json
import multiprocessing
import socket
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import lance
import pyarrow as pa
import pyarrow.parquet as pq

from build_esm_plan import require_source_region
from locate_esm_rows import ATLAS_STORAGE, ATLAS_URI, ATLAS_VERSION
from production_policy import select_three_dynamic
from sample_esm import decode_protein
from structure_audit import Protein, compare


def fetch_structures(
    queue: pa.Table, batch_size: int
) -> tuple[dict[str, Protein], list[dict]]:
    """Fetch, decode, and verify all coordinate blobs in one structural queue."""
    records = sorted(queue.to_pylist(), key=lambda row: row["row_index"])
    dataset = lance.dataset(
        ATLAS_URI, version=ATLAS_VERSION, storage_options=ATLAS_STORAGE
    )
    proteins = {}
    timings = []
    for offset in range(0, len(records), batch_size):
        planned = records[offset : offset + batch_size]
        started = perf_counter()
        table = dataset.take(
            [row["row_index"] for row in planned],
            columns=[
                "protein_hash",
                "sequence",
                "mean_plddt",
                "ptm",
                "structure_blob",
            ],
        )
        elapsed = perf_counter() - started
        source = table.to_pylist()
        if len(source) != len(planned):
            raise ValueError("Atlas structure take returned an incomplete batch")
        for plan_row, source_row in zip(planned, source, strict=True):
            entry_id = plan_row["entry_id"]
            if source_row["protein_hash"] != entry_id:
                raise ValueError(f"Atlas row drift while fetching {entry_id}")
            if source_row["sequence"] != plan_row["sequence"]:
                raise ValueError(f"Atlas sequence changed while fetching {entry_id}")
            sequence, coords, plddt = decode_protein(source_row)
            if sequence != plan_row["sequence"]:
                raise ValueError(f"Decoded structure sequence changed for {entry_id}")
            if entry_id in proteins:
                raise ValueError(f"Duplicate structure in shard: {entry_id}")
            proteins[entry_id] = Protein(sequence, coords, plddt)
            timings.append(
                {
                    "entry_id": entry_id,
                    "struct_cluster_id": plan_row["struct_cluster_id"],
                    "atlas_row_id": plan_row["row_index"],
                    "structure_blob_bytes": len(source_row["structure_blob"]),
                    "structure_blob_sha256": hashlib.sha256(
                        source_row["structure_blob"]
                    ).hexdigest(),
                    "batch_rows": len(planned),
                    "batch_fetch_seconds": elapsed,
                    "hostname": socket.gethostname(),
                }
            )
        print(
            f"Atlas structures: {min(offset + batch_size, len(records)):,}/{len(records):,}",
            flush=True,
        )
    return proteins, timings


def rank_cluster(job: tuple[list[dict], dict[str, Protein]]) -> tuple[list[dict], list[dict]]:
    """Run lazy structural-first selection for one original cluster."""
    rows, proteins = job

    def compare_pair(a: dict, b: dict) -> dict:
        return compare(proteins[a["entry_id"]], proteins[b["entry_id"]])

    return select_three_dynamic(rows, compare_pair=compare_pair)


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write a nonempty list of records to parquet."""
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def main() -> None:
    """Fetch and structurally rank one metadata-curated production shard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--take-batch", type=int, default=256)
    args = parser.parse_args()
    require_source_region()
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    queue = pq.read_table(args.queue)
    proteins, fetch_timings = fetch_structures(queue, args.take_batch)
    groups = {}
    for row in queue.to_pylist():
        groups.setdefault(row["struct_cluster_id"], []).append(row)
    jobs = [
        (members, {row["entry_id"]: proteins[row["entry_id"]] for row in members})
        for members in groups.values()
    ]
    selected = []
    pairs = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        for index, (chosen, measured) in enumerate(
            pool.map(rank_cluster, jobs, chunksize=8), 1
        ):
            selected.extend(chosen)
            pairs.extend(measured)
            if index % 1000 == 0:
                print(
                    f"Structural ranking: {index:,}/{len(jobs):,} clusters, "
                    f"{len(pairs):,} measured pairs",
                    flush=True,
                )
    if len(selected) != 3 * len(groups):
        raise ValueError(
            f"Expected exactly three selections in each queued cluster; "
            f"got {len(selected):,} across {len(groups):,} clusters"
        )
    write_rows(args.output / "selected_after_alignment.parquet", selected)
    write_rows(args.output / "pair_metrics.parquet", pairs)
    write_rows(args.output / "fetch_timings.parquet", fetch_timings)
    summary = {
        "status": "complete",
        "queue_rows": queue.num_rows,
        "queue_clusters": len(groups),
        "selected": len(selected),
        "structural_diversity": sum(
            row["selection_tier"] == "structural_diversity" for row in selected
        ),
        "quality_fill": sum(row["selection_tier"] == "quality_fill" for row in selected),
        "measured_pairs": len(pairs),
        "all_pairs_avoided": sum(
            len(members) * (len(members) - 1) // 2 for members in groups.values()
        )
        - len(pairs),
        "structure_blob_bytes": sum(
            row["structure_blob_bytes"] for row in fetch_timings
        ),
        "atlas_uri": ATLAS_URI,
        "atlas_version": ATLAS_VERSION,
        "tmtools_version": "0.3.0",
        "workers": args.workers,
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

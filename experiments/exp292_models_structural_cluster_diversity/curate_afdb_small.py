"""Curate the AFDB clusters that exp53's minimum-cluster-size rule removed.

These clusters never entered training, so there is no retained anchor to measure
novelty against — and because each holds at most two usable members, the
three-slot policy admits every member that passes quality. The frozen structural
comparison therefore has nothing to decide here and is not run: the whole arm is
integrity checks plus the held-out sequence screen.

Rows are labelled ``untrained_cluster`` rather than reusing the anchored arm's
``structural_diversity`` / ``quality_fill`` tiers. Those tiers describe a choice
made against retained anchors, and describing an unanchored member with either
would overstate what was measured. The distinct label also keeps this population
separable in any later analysis, which matters because it is a different
population: shorter, marginally less confident, and one sequence cluster each.
"""

import shutil
import socket
from collections.abc import Iterable, Iterator
from pathlib import Path
from time import perf_counter

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from curate_afdb import (
    COORDINATE_COLUMNS,
    MAX_LENGTH,
    MIN_LENGTH,
    MIN_PLDDT,
    publish_shard,
    stage_tools,
)
from sequence_exclusion import MMSEQS_SHA256, read_fasta, run_sequence_screen

SELECTION_TIER = "untrained_cluster"
# Emitted for schema parity with the anchored manifest. They are null by
# construction here: with no anchor and no competing candidate there is nothing
# to compare a member against.
UNMEASURED_FIELDS = (
    "max_selected_tm",
    "max_selected_core_tm",
    "min_selected_coverage",
    "min_selected_length_ratio",
    "max_selected_sequence_identity",
    "structural_novelty",
    "structurally_comparable",
    "strict_structural_diversity",
    "max_anchor_sequence_identity",
)


def quality_filter(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Apply source integrity and candidate quality to unanchored members.

    Every row is a candidate: there is no anchor whose failure could disqualify
    a cluster. A length disagreement between the source structure and the plan
    metadata is an integrity failure and raises rather than filtering.
    """
    kept: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        if row["is_anchor"]:
            raise ValueError(f"{row['entry_id']}: untrained clusters hold no anchor")
        if len(row["sequence"]) != int(row["seq_len"]):
            raise ValueError(f"{row['entry_id']}: source length changed since planning")
        reason = None
        if int(row["noncanonical_residues"]):
            reason = "noncanonical_candidate"
        elif not MIN_LENGTH <= int(row["seq_len"]) <= MAX_LENGTH:
            reason = "candidate_length"
        elif not np.isfinite(row["global_plddt"]):
            reason = "candidate_plddt_nonfinite"
        elif float(row["global_plddt"]) < MIN_PLDDT:
            reason = "candidate_plddt"
        if reason:
            rejected.append({**row, "rejection_reason": reason})
        else:
            kept.append(row)
    return kept, rejected


def select_all(rows: Iterable[dict]) -> list[dict]:
    """Admit every surviving member, ranked within its cluster for provenance."""
    clusters: dict[str, list[dict]] = {}
    for row in rows:
        clusters.setdefault(row["struct_cluster_id"], []).append(row)
    selected = []
    for members in clusters.values():
        ordered = sorted(
            members, key=lambda row: (-float(row["global_plddt"]), str(row["entry_id"]))
        )
        for rank, row in enumerate(ordered, 1):
            selected.append(
                {
                    **row,
                    **dict.fromkeys(UNMEASURED_FIELDS),
                    "selection_rank": rank,
                    "selection_tier": SELECTION_TIER,
                }
            )
    return selected


def curate_table(
    table: pa.Table,
    references: list[tuple[str, str]],
    mmseqs: Path,
    work: Path,
    threads: int,
) -> dict:
    """Curate one validated untrained-cluster shard."""
    started = perf_counter()
    metadata_columns = [
        name for name in table.schema.names if name not in COORDINATE_COLUMNS
    ]
    rows = table.select(metadata_columns).to_pylist()
    quality, quality_rejections = quality_filter(rows)
    excluded, screen_stats = run_sequence_screen(
        quality, references, mmseqs, work / "sequence-screen", threads
    )
    sequence_rejections = [
        {**row, **excluded[row["entry_id"]], "rejection_reason": "heldout_sequence"}
        for row in quality
        if row["entry_id"] in excluded
    ]
    survivors = [row for row in quality if row["entry_id"] not in excluded]
    selected = select_all(survivors)
    order = [*metadata_columns, *UNMEASURED_FIELDS, "selection_rank", "selection_tier"]
    return {
        "selected": [{key: row.get(key) for key in order} for row in selected],
        "pair_metrics": [],
        "quality_rejections": quality_rejections,
        "sequence_rejections": sequence_rejections,
        "summary": {
            "shard_rows": table.num_rows,
            "quality_rows": len(quality),
            "quality_rejections": len(quality_rejections),
            "sequence_rejections": len(sequence_rejections),
            "clusters": len({row["struct_cluster_id"] for row in selected}),
            "selected": len(selected),
            "selection_tier": SELECTION_TIER,
            "sequence_screen": screen_stats,
            "mmseqs_archive_sha256": MMSEQS_SHA256,
            "elapsed_seconds": perf_counter() - started,
            "hostname": socket.gethostname(),
        },
    }


def curate_shard_files(
    paths: Iterable[str],
    shard_info=None,
    *,
    archive_uri: str,
    reference_uris: tuple[str, ...],
    aux_prefix: str,
    work: str,
    threads: int = 1,
) -> Iterator[dict]:
    """Curate untrained-cluster shards and yield their admitted members."""
    mmseqs, reference_paths, digests = stage_tools(archive_uri, reference_uris, work)
    references = read_fasta(list(reference_paths))
    for path in paths:
        shard = Path(path).stem
        with fsspec.open(path, "rb") as handle:
            table = pq.read_table(pa.BufferReader(handle.read()))
        scratch = Path(work) / "shards" / shard
        result = curate_table(table, references, mmseqs, scratch, threads)
        result["summary"]["shard"] = shard
        result["summary"]["source"] = path
        result["summary"]["reference_sha256"] = list(digests)
        publish_shard(result, aux_prefix, shard)
        shutil.rmtree(scratch, ignore_errors=True)
        yield from result["selected"]

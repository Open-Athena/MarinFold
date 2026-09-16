"""Curate one validated AFDB shard into production training additions.

The production plan hash-partitions on ``struct_cluster_id``, so every row of a
structural cluster — all retained training anchors and the whole bounded
candidate reservoir — lands in the same shard. That locality lets quality
filtering, held-out sequence exclusion and structural-first selection all run
inside one shard with no cross-shard join.

Coordinates were already validated and stored by the source-local fetch job, so
this stage never touches AFDB again. Alignment work stays bounded: a cluster
with at most three surviving candidates needs no comparison at all, and larger
clusters measure only candidate-to-anchor and candidate-to-addition pairs.
"""

import hashlib
import json
import multiprocessing
import os
import shutil
import socket
import tarfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path
from time import perf_counter

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from production_policy import ADDITIONS_PER_CLUSTER, select_three_dynamic
from sequence_exclusion import MMSEQS_SHA256, read_fasta, run_sequence_screen
from structure_audit import Protein, compare

MIN_PLDDT = 80.0
MIN_LENGTH = 60
MAX_LENGTH = 1000
COORDINATE_COLUMNS = ("ca_coords", "per_residue_plddt")


def protein_from_arrays(sequence: str, coords, plddt) -> Protein:
    """Rebuild validated C-alpha geometry from the stored parquet arrays."""
    geometry = np.asarray(coords, dtype=np.float64)
    confidence = np.asarray(plddt, dtype=np.float64)
    if geometry.shape != (len(sequence), 3) or confidence.shape != (len(sequence),):
        raise ValueError("Stored structure arrays do not match their sequence")
    return Protein(sequence, geometry, confidence)


def quality_filter(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Apply source integrity and candidate quality rules cluster by cluster.

    An anchor that fails integrity or confidence disqualifies its whole cluster:
    novelty is measured against the anchors, so an unusable anchor makes every
    comparison in that cluster uninterpretable. Length disagreement between the
    source structure and the plan metadata is an integrity failure, not a
    filter, and raises.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["struct_cluster_id"], []).append(row)
    kept: list[dict] = []
    rejected: list[dict] = []
    for cluster_id, members in groups.items():
        anchors = [row for row in members if row["is_anchor"]]
        if not anchors:
            raise ValueError(f"{cluster_id}: shard holds no retained training anchor")
        for row in members:
            if len(row["sequence"]) != int(row["seq_len"]):
                raise ValueError(f"{row['entry_id']}: source length changed since planning")
        unusable = next(
            (
                "noncanonical_anchor_cluster"
                if int(anchor["noncanonical_residues"])
                else "anchor_source_quality"
                for anchor in anchors
                if int(anchor["noncanonical_residues"])
                or not np.isfinite(anchor["global_plddt"])
                or float(anchor["global_plddt"]) < MIN_PLDDT
            ),
            None,
        )
        if unusable:
            rejected.extend({**row, "rejection_reason": unusable} for row in members)
            continue
        kept.extend(anchors)
        for candidate in (row for row in members if not row["is_anchor"]):
            reason = None
            if int(candidate["noncanonical_residues"]):
                reason = "noncanonical_candidate"
            elif not MIN_LENGTH <= int(candidate["seq_len"]) <= MAX_LENGTH:
                reason = "candidate_length"
            elif not np.isfinite(candidate["global_plddt"]):
                reason = "candidate_plddt_nonfinite"
            elif float(candidate["global_plddt"]) < MIN_PLDDT:
                reason = "candidate_plddt"
            if reason:
                rejected.append({**candidate, "rejection_reason": reason})
            else:
                kept.append(candidate)
    return kept, rejected


def partition_after_screen(
    quality_rows: Iterable[dict], excluded: dict[str, dict]
) -> tuple[list[list[dict]], list[dict]]:
    """Drop excluded candidates and group the clusters that still have a choice."""
    groups: dict[str, list[dict]] = {}
    sequence_rejections: list[dict] = []
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
    clusters = [
        members
        for members in groups.values()
        if any(not row["is_anchor"] for row in members)
    ]
    return clusters, sequence_rejections


def select_cluster(
    job: tuple[list[dict], dict[str, Protein]],
) -> tuple[list[dict], list[dict]]:
    """Run structural-first selection for one cluster, measuring pairs lazily.

    Takes a single argument so it can be mapped across a process pool. A cluster
    that needs no alignment is passed an empty structure map and never compares.
    """
    members, proteins = job

    def compare_pair(a: dict, b: dict) -> dict:
        return compare(proteins[a["entry_id"]], proteins[b["entry_id"]])

    return select_three_dynamic(members, compare_pair=compare_pair)


def map_clusters(jobs: list[tuple], workers: int) -> Iterator[tuple]:
    """Rank clusters, spreading the alignment cost over the worker's cores.

    Pairwise TM-alignment, not I/O, dominates this stage, so extra cores do real
    work here. One worker keeps everything in-process, which is what the tests
    and a local run use.
    """
    if workers <= 1:
        yield from (select_cluster(job) for job in jobs)
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        yield from pool.map(select_cluster, jobs, chunksize=8)


def curate_table(
    table: pa.Table,
    references: list[tuple[str, str]],
    mmseqs: Path,
    work: Path,
    threads: int,
    workers: int = 1,
    progress_every: int = 250,
) -> dict:
    """Curate every cluster in one validated shard.

    Coordinates stay in Arrow until a cluster actually needs an alignment, so a
    shard whose clusters are all decided on metadata never materializes them.
    """
    started = perf_counter()
    metadata_columns = [
        name for name in table.schema.names if name not in COORDINATE_COLUMNS
    ]
    rows = table.select(metadata_columns).to_pylist()
    quality, quality_rejections = quality_filter(rows)
    candidates = [row for row in quality if not row["is_anchor"]]
    excluded, screen_stats = run_sequence_screen(
        candidates, references, mmseqs, work / "sequence-screen", threads
    )
    clusters, sequence_rejections = partition_after_screen(quality, excluded)
    entries = table["entry_id"].to_pylist()
    positions = {entry: index for index, entry in enumerate(entries)}
    if len(positions) != len(entries):
        raise ValueError("Validated shard contains duplicate entry IDs")
    coordinates = table["ca_coords"]
    confidence = table["per_residue_plddt"]

    def protein(row: dict) -> Protein:
        index = positions[row["entry_id"]]
        return protein_from_arrays(
            row["sequence"], coordinates[index].as_py(), confidence[index].as_py()
        )

    jobs = []
    aligned_clusters = 0
    for members in clusters:
        needs_alignment = (
            sum(not row["is_anchor"] for row in members) > ADDITIONS_PER_CLUSTER
        )
        aligned_clusters += needs_alignment
        structures = (
            {row["entry_id"]: protein(row) for row in members} if needs_alignment else {}
        )
        jobs.append((members, structures))
    selected: list[dict] = []
    pairs: list[dict] = []
    alignment_started = perf_counter()
    for index, (chosen, measured) in enumerate(map_clusters(jobs, workers), 1):
        selected.extend(chosen)
        pairs.extend(measured)
        if index % progress_every == 0 or index == len(jobs):
            print(
                f"Selection: {index:,}/{len(jobs):,} clusters, "
                f"{aligned_clusters:,} aligned, {len(pairs):,} measured pairs, "
                f"{perf_counter() - alignment_started:,.0f}s",
                flush=True,
            )
    return {
        "selected": selected,
        "pair_metrics": pairs,
        "quality_rejections": quality_rejections,
        "sequence_rejections": sequence_rejections,
        "summary": {
            "shard_rows": table.num_rows,
            "quality_rows": len(quality),
            "quality_rejections": len(quality_rejections),
            "sequence_rejections": len(sequence_rejections),
            "clusters_with_a_choice": len(clusters),
            "clusters_aligned": aligned_clusters,
            "selected": len(selected),
            "structural_diversity": sum(
                row["selection_tier"] == "structural_diversity" for row in selected
            ),
            "quality_fill": sum(
                row["selection_tier"] == "quality_fill" for row in selected
            ),
            "measured_pairs": len(pairs),
            "selection_workers": workers,
            "alignment_seconds": perf_counter() - alignment_started,
            "sequence_screen": screen_stats,
            "elapsed_seconds": perf_counter() - started,
            "hostname": socket.gethostname(),
        },
    }


@cache
def stage_tools(
    archive_uri: str, reference_uris: tuple[str, ...], root: str
) -> tuple[Path, tuple[Path, ...], tuple[str, ...]]:
    """Download, verify and unpack the frozen screen tooling once per worker.

    A Zephyr worker serves many shards, so the archive is fetched and its hash
    checked a single time per process. A mismatch raises rather than screening
    with an unknown binary. The unpack directory is process-private so two
    processes sharing a pod cannot read each other's half-written binary.
    """
    base = Path(root) / f"tooling-{os.getpid()}"
    tools = base / "tools"
    reference_dir = base / "reference"
    tools.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    archive = tools / Path(archive_uri).name
    with fsspec.open(archive_uri, "rb") as handle:
        payload = handle.read()
    if hashlib.sha256(payload).hexdigest() != MMSEQS_SHA256:
        raise ValueError("Frozen MMseqs2 archive hash mismatch")
    archive.write_bytes(payload)
    with tarfile.open(archive) as bundle:
        bundle.extractall(tools, filter="data")
    references = []
    digests = []
    for uri in reference_uris:
        target = reference_dir / Path(uri).name
        with fsspec.open(uri, "rb") as handle:
            content = handle.read()
        target.write_bytes(content)
        references.append(target)
        digests.append(hashlib.sha256(content).hexdigest())
    return tools / "mmseqs" / "bin" / "mmseqs", tuple(references), tuple(digests)


def write_rows(uri: str, rows: list[dict]) -> None:
    """Write a nonempty record list to one parquet object."""
    if not rows:
        return
    with fsspec.open(uri, "wb") as handle:
        pq.write_table(pa.Table.from_pylist(rows), handle, compression="zstd")


def publish_shard(result: dict, aux_prefix: str, shard: str) -> None:
    """Persist every non-manifest output so no filtered row is unexplained."""
    prefix = aux_prefix.rstrip("/") + f"/shard={shard}"
    for name in ("pair_metrics", "quality_rejections", "sequence_rejections"):
        write_rows(f"{prefix}/{name}.parquet", result[name])
    with fsspec.open(f"{prefix}/summary.json", "w") as handle:
        handle.write(json.dumps(result["summary"], indent=2) + "\n")


def curate_shard_files(
    paths: Iterable[str],
    shard_info=None,
    *,
    archive_uri: str,
    reference_uris: tuple[str, ...],
    aux_prefix: str,
    work: str,
    threads: int = 1,
    workers: int = 1,
) -> Iterator[dict]:
    """Curate validated parquets and yield their selected additions.

    Each input file is read as a whole object; rejections, measured pairs and
    the per-shard accounting are published beside the manifest so every row that
    does not become an addition still has a recorded reason.
    """
    mmseqs, reference_paths, digests = stage_tools(archive_uri, reference_uris, work)
    references = read_fasta(list(reference_paths))
    for path in paths:
        shard = Path(path).stem
        with fsspec.open(path, "rb") as handle:
            table = pq.read_table(pa.BufferReader(handle.read()))
        scratch = Path(work) / "shards" / shard
        result = curate_table(table, references, mmseqs, scratch, threads, workers)
        result["summary"]["shard"] = shard
        result["summary"]["source"] = path
        result["summary"]["reference_sha256"] = list(digests)
        result["summary"]["mmseqs_archive_sha256"] = MMSEQS_SHA256
        publish_shard(result, aux_prefix, shard)
        shutil.rmtree(scratch, ignore_errors=True)
        yield from result["selected"]

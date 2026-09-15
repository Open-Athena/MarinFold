"""Recover a small curation sample from the original ESM clusters in us-west-2.

The full membership TSV is streamed once in its AWS region. Structures are
retrieved through the Atlas protein_hash scalar index, with exact completeness
checks. This diagnostic sample is drawn from specified materialization plan
shards, not a population-representative yield survey. Nothing is training-ready.
"""

import argparse
import hashlib
import heapq
import json
import multiprocessing
import shutil
import socket
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import boto3
import brotli
import lance
import msgpack
import msgpack_numpy
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests

from structure_audit import (
    audit_cluster,
    noncanonical_sequence,
    select_candidates,
    write_csv,
)

BUCKET = "marinfold-exp91-usw2"
MEMBERSHIP = "exp91/out/clu_cluster.tsv"
ATLAS = "s3://esm-protein-atlas/v1/folds/folds_1B.lance"
ATLAS_VERSION = 3
STRUCTURE_COLUMNS = (
    "protein_hash",
    "sequence",
    "mean_plddt",
    "ptm",
    "structure_blob",
    "pae",
)


def require_source_region() -> None:
    """Refuse a bulk membership read anywhere outside the source's EC2 region."""
    session = requests.Session()
    session.trust_env = False
    base = "http://169.254.169.254/latest/"
    token = session.put(
        base + "api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        timeout=5,
    )
    token.raise_for_status()
    identity = session.get(
        base + "dynamic/instance-identity/document",
        headers={"X-aws-ec2-metadata-token": token.text},
        timeout=5,
    )
    identity.raise_for_status()
    if identity.json()["region"] != "us-west-2":
        raise ValueError("The 10.77 GB membership must be processed in us-west-2")


def choose_clusters(
    plan: pa.Table, retained: set[str], per_bin: int, seed: int
) -> list[dict]:
    """Stratify a development sample and require an actual current training anchor."""
    groups = {}
    for row in plan.to_pylist():
        length, size = row["seq_len"], row["cluster_size"]
        if (
            row["protein_hash"] not in retained
            or row["mean_plddt"] < 0.8
            or not 60 <= length <= 1000
            or not 2 <= size <= 100000
        ):
            continue
        length_bin = (
            "60-249" if length < 250 else "250-499" if length < 500 else "500-1000"
        )
        size_bin = "2-9" if size < 10 else "10-99" if size < 100 else "100+"
        groups.setdefault((length_bin, size_bin), []).append(
            {**row, "length_bin": length_bin, "size_bin": size_bin}
        )
    chosen = []
    for group in sorted(groups):
        chosen.extend(
            sorted(
                groups[group],
                key=lambda r: hashlib.sha256(
                    f"{seed}:{r['cluster_id']}".encode()
                ).hexdigest(),
            )[:per_bin]
        )
    if not chosen:
        raise ValueError("No eligible clusters with retained training anchors")
    if len({r["cluster_id"] for r in chosen}) != len(chosen):
        raise ValueError("Duplicate clusters in materialization plans")
    return chosen


def sample_members(
    stream, clusters: list[dict], cap: int, seed: int
) -> tuple[dict, int]:
    """Keep a deterministic bounded reservoir while verifying complete groups."""
    ids = pa.array([r["cluster_id"] for r in clusters])
    counts = {h: 0 for h in ids.to_pylist()}
    anchors = {r["cluster_id"]: r["protein_hash"] for r in clusters}
    anchor_counts = dict(counts)
    self_counts = dict(counts)
    heaps = {h: [] for h in counts}
    total = 0
    reader = pacsv.open_csv(
        stream,
        read_options=pacsv.ReadOptions(
            column_names=["cluster", "member"], block_size=16 << 20
        ),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            column_types={"cluster": pa.string(), "member": pa.string()}
        ),
    )
    for batch_id, batch in enumerate(reader):
        total += batch.num_rows
        keep = pc.is_in(batch.column(0), value_set=ids)
        for row in batch.filter(keep).to_pylist():
            cluster, member = row["cluster"], row["member"]
            counts[cluster] += 1
            if member == cluster:
                self_counts[cluster] += 1
            if member == anchors[cluster]:
                anchor_counts[cluster] += 1
                continue
            score = int(hashlib.sha256(f"{seed}:{member}".encode()).hexdigest(), 16)
            heapq.heappush(heaps[cluster], (-score, member))
            if len(heaps[cluster]) > cap:
                heapq.heappop(heaps[cluster])
        if batch_id % 32 == 0:
            print(f"Membership scan: {total:,} rows", flush=True)
    for row in clusters:
        h = row["cluster_id"]
        if (
            counts[h] != row["cluster_size"]
            or self_counts[h] != 1
            or anchor_counts[h] != 1
        ):
            raise ValueError(
                f"Incomplete original group {h}: {counts[h]} members, {self_counts[h]} self rows; expected {row['cluster_size']}"
            )
        if len({m for _, m in heaps[h]}) != len(heaps[h]):
            raise ValueError(f"Duplicate sampled member in {h}")
    return {
        h: [m for _, m in sorted(heap, reverse=True)] for h, heap in heaps.items()
    }, total


def decode_protein(row: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """Validate the Atlas atom37 encoding against its independent sequence column."""
    decoded = msgpack.unpackb(
        brotli.decompress(row["structure_blob"]),
        raw=False,
        strict_map_key=False,
        object_hook=msgpack_numpy.decode,
    )
    sequence = decoded["sequence"]
    n = len(sequence)
    if sequence != row["sequence"]:
        raise ValueError(f"Atlas/blob sequence mismatch for {row['protein_hash']}")
    if hashlib.md5(sequence.encode()).hexdigest() != row["protein_hash"]:
        raise ValueError("Atlas hash does not match its own sequence")
    if np.asarray(decoded["chain_boundaries"]).tolist() != [[0, n]]:
        raise ValueError("Expected one complete chain")
    mask = np.asarray(decoded["atom37_mask"], dtype=bool)
    positions = np.asarray(decoded["atom37_positions"], dtype=np.float64)
    confidence = np.asarray(decoded["confidence"], dtype=np.float64)
    if (
        mask.shape != (n, 37)
        or not mask[:, 1].all()
        or positions.shape != (int(mask.sum()), 3)
    ):
        raise ValueError("Incomplete or inconsistent atom37 geometry")
    if (
        confidence.shape != (n,)
        or not np.isfinite(confidence).all()
        or np.any((confidence < 0) | (confidence > 1))
    ):
        raise ValueError("Expected per-residue confidence on the 0-1 scale")
    if not np.isfinite(positions).all():
        raise ValueError("Non-finite atomic coordinates")
    ca_offsets = (
        np.cumsum(mask.sum(axis=1)) - mask.sum(axis=1) + mask[:, :1].sum(axis=1)
    )
    return sequence, positions[ca_offsets], confidence * 100


def retrieve(
    ds: lance.LanceDataset,
    hashes: list[str],
    columns: tuple[str, ...] = STRUCTURE_COLUMNS,
) -> pa.Table:
    """Use the scalar index and fail if any hash is absent or duplicated."""
    if any(len(h) != 32 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
        raise ValueError("Invalid protein hash")
    predicate = "protein_hash IN (" + ",".join(f"'{h}'" for h in hashes) + ")"
    scanner = ds.scanner(
        filter=predicate,
        columns=list(columns),
        with_row_id=True,
    )
    if "ScalarIndexQuery" not in scanner.explain_plan():
        raise ValueError("Atlas retrieval would scan instead of using its hash index")
    table = scanner.to_table()
    found = table.column("protein_hash").to_pylist()
    if len(found) != len(hashes) or set(found) != set(hashes):
        raise ValueError(
            f"Atlas lookup is incomplete or duplicated: requested {len(hashes)}, got {len(found)}"
        )
    return table


def retrieve_timed(
    job: tuple[int, list[str]], ds: lance.LanceDataset
) -> tuple[int, pa.Table, float]:
    """Time one independently indexed lookup batch."""
    offset, hashes = job
    start = perf_counter()
    table = retrieve(ds, hashes)
    return offset, table, perf_counter() - start


def retrieve_batches(
    ds: lance.LanceDataset, hashes: list[str]
) -> Iterator[tuple[int, pa.Table, float]]:
    """Overlap four small indexed reads, bounding decoded-buffer memory to 128 rows."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(hashes), 128):
            jobs = [
                (offset, hashes[offset : offset + 32])
                for offset in range(start, min(start + 128, len(hashes)), 32)
            ]
            yield from pool.map(partial(retrieve_timed, ds=ds), jobs)


def main() -> None:
    """Stream membership, fetch indexed structures, then measure curation pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-bin", type=int, default=4)
    parser.add_argument("--candidates-per-cluster", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=292)
    args = parser.parse_args()
    require_source_region()
    start = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "structures"
    cache_dir.mkdir(exist_ok=True)
    input_sampling = args.plan.with_name("sampling.json")
    if input_sampling.exists():
        shutil.copy2(input_sampling, args.output / "sampling.json")
    retained = set(
        pq.read_table(args.retained, columns=["entry_id"])["entry_id"].to_pylist()
    )
    clusters = choose_clusters(
        pq.read_table(args.plan), retained, args.per_bin, args.seed
    )
    write_csv(args.output / "clusters.csv", clusters)
    s3 = boto3.client("s3", region_name="us-west-2")
    head = s3.head_object(Bucket=BUCKET, Key=MEMBERSHIP)
    response = s3.get_object(Bucket=BUCKET, Key=MEMBERSHIP, IfMatch=head["ETag"])
    with response["Body"] as stream:
        members, total = sample_members(
            stream, clusters, args.candidates_per_cluster, args.seed
        )
    membership_seconds = perf_counter() - start
    if total != 163144153:
        raise ValueError(f"Original membership row count changed: {total}")
    (args.output / "membership_sample.json").write_text(json.dumps(members, indent=2))
    print(
        f"Recovered {sum(map(len, members.values()))} omitted members in {membership_seconds:.1f}s",
        flush=True,
    )
    ds = lance.dataset(
        ATLAS,
        version=ATLAS_VERSION,
        storage_options={
            "aws_skip_signature": "true",
            "region": "us-west-2",
            "timeout": "120s",
            "connect_timeout": "20s",
        },
    )
    by_hash = {}
    for cluster in clusters:
        h = cluster["cluster_id"]
        for member in [cluster["protein_hash"], *members[h]]:
            if member in by_hash:
                raise ValueError(
                    f"Member assigned to multiple original clusters: {member}"
                )
            by_hash[member] = cluster
    hashes = sorted(by_hash)
    rows, fetched, rejected = [], [], []
    rejected_clusters = set()
    for offset, batch, elapsed in retrieve_batches(ds, hashes):
        for record in batch.to_pylist():
            h = record["protein_hash"]
            cluster = by_hash[h]
            is_anchor = h == cluster["protein_hash"]
            seq, coords, plddt = decode_protein(record)
            if noncanonical_sequence(seq):
                if is_anchor:
                    rejected_clusters.add(cluster["cluster_id"])
                rejected.append(
                    {
                        "entry_id": h,
                        "struct_cluster_id": cluster["cluster_id"],
                        "seq_len": len(seq),
                        "mean_plddt": record["mean_plddt"],
                        "ptm": record["ptm"],
                        "reason": "noncanonical_anchor; exclude_cluster"
                        if is_anchor
                        else "noncanonical_sequence",
                    }
                )
                continue
            if not is_anchor and (
                record["mean_plddt"] < 0.8
                or record["ptm"] < 0.5
                or not 60 <= len(seq) <= 1000
            ):
                rejected.append(
                    {
                        "entry_id": h,
                        "struct_cluster_id": cluster["cluster_id"],
                        "seq_len": len(seq),
                        "mean_plddt": record["mean_plddt"],
                        "ptm": record["ptm"],
                        "reason": "quality_or_length",
                    }
                )
                continue
            if is_anchor and len(seq) != cluster["seq_len"]:
                raise ValueError(f"Original anchor length mismatch: {h}")
            np.savez_compressed(
                cache_dir / f"{h}.npz", sequence=seq, coords=coords, plddt=plddt
            )
            (cache_dir / f"{h}.blob").write_bytes(record["structure_blob"])
            if record["pae"] is not None:
                (cache_dir / f"{h}.pae.zip").write_bytes(record["pae"])
            rows.append(
                {
                    "entry_id": h,
                    "struct_cluster_id": cluster["cluster_id"],
                    "is_anchor": str(is_anchor).lower(),
                    "seq_len": len(seq),
                    "global_plddt": record["mean_plddt"] * 100,
                    "ptm": record["ptm"],
                    "cluster_size": cluster["cluster_size"],
                    "n_anchors": 1,
                    "source": "esmfold2",
                    "split": "train",
                    "length_bin": cluster["length_bin"],
                    "size_bin": cluster["size_bin"],
                }
            )
            fetched.append(
                {
                    "entry_id": h,
                    "struct_cluster_id": cluster["cluster_id"],
                    "source_uri": ATLAS,
                    "atlas_version": ATLAS_VERSION,
                    "atlas_row_id": record["_rowid"],
                    "n_residues": len(seq),
                    "mean_plddt": float(plddt.mean()),
                    "confident_fraction": float(np.mean(plddt >= 70)),
                    "sha256": hashlib.sha256(record["structure_blob"]).hexdigest(),
                    "bytes": len(record["structure_blob"]),
                    "batch_id": offset // 32,
                    "batch_lookup_seconds": elapsed,
                    "hostname": socket.gethostname(),
                }
            )
        print(
            f"Atlas retrieval: {min(offset + 32, len(hashes))}/{len(hashes)}, {len(rows)} quality-passing structures",
            flush=True,
        )
    if rejected_clusters:
        write_csv(
            args.output / "cluster_rejections.csv",
            [
                {
                    **r,
                    "reason": "noncanonical_anchor; source/training residue mapping requires separate audit",
                }
                for r in clusters
                if r["cluster_id"] in rejected_clusters
            ],
        )
        rows = [r for r in rows if r["struct_cluster_id"] not in rejected_clusters]
        fetched = [
            r for r in fetched if r["struct_cluster_id"] not in rejected_clusters
        ]
    rows.sort(
        key=lambda r: (
            r["struct_cluster_id"],
            r["is_anchor"] != "true",
            r["entry_id"],
        )
    )
    write_csv(args.output / "sample.csv", rows)
    write_csv(args.output / "fetch_timings.csv", fetched)
    if rejected:
        write_csv(args.output / "quality_rejections.csv", rejected)
    groups = {}
    for row in rows:
        groups.setdefault(row["struct_cluster_id"], []).append(row)
    pairs = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for i, batch in enumerate(
            pool.map(
                partial(audit_cluster, cache_dir=cache_dir),
                [(g, {}) for g in groups.values()],
            ),
            1,
        ):
            pairs.extend(batch)
            print(
                f"Aligned {i}/{len(groups)} clusters ({len(pairs)} pairs)", flush=True
            )
    write_csv(args.output / "pairs.csv", pairs)
    candidates = select_candidates(rows, pairs)
    write_csv(args.output / "candidates.csv", candidates)
    summary = {
        "source": "esmfold2",
        "clusters_excluded_noncanonical_anchor": len(rejected_clusters),
        "structures": len(rows),
        "pairs": len(pairs),
        "candidates": len(candidates),
        "provisional_diverse": sum(r["selected_order"] > 0 for r in candidates),
        "clusters_with_diverse": len(
            {r["struct_cluster_id"] for r in candidates if r["selected_order"] > 0}
        ),
        "membership_uri": f"s3://{BUCKET}/{MEMBERSHIP}",
        "membership_etag": head["ETag"],
        "membership_bytes": head["ContentLength"],
        "membership_rows": total,
        "membership_seconds": membership_seconds,
        "atlas_version": ATLAS_VERSION,
        "elapsed_seconds": perf_counter() - start,
        "tmtools_version": "0.3.0",
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "retained_sha256": hashlib.sha256(args.retained.read_bytes()).hexdigest(),
        "sampling_scope": "stratified curation sample from supplied anchor metadata; see sampling.json for source and inclusion design; not an unweighted population yield estimate",
        "status": "visual curation; candidate-level decontamination still required",
    }
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

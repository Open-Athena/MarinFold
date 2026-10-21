# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage E — generate contacts-v1 documents for the PINDER heterodimer arm.

Structurally this is ``extract.py`` with the hard part removed. A tar has no
index, so AFCDB had to walk ~28,000 headers to find one member and that walk
became 88% of the cost. A zip has a central directory, so ``pinder_select.py``
already resolved every system to a byte offset: this worker just fetches,
inflates and generates.

Measured per system: ~0.35 s to fetch and inflate, ~0.33 s to generate, so the
whole 449,835-system arm is ~87 pod-hours against AFCDB's ~1,400.

Everything that made ``extract.py`` survivable is reused rather than
reimplemented -- the status-checked retrying range reader, the single-threaded
generator warm-up, and the pinned native thread counts. Work is split into
fixed-size batches; a batch whose ledger file exists is skipped, so preemption
costs at most one batch.

    uv run python pinder_extract.py \\
        --manifest /data/exp294_stageE/selection/pinder_selected.parquet \\
        --out gs://.../pinder --batch-size 2000 --shard-index 0 --shard-count 20
"""

import argparse
import hashlib
import json
import struct
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import duckdb
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import (
    _GENERATOR,
    _RETRYABLE,
    _generator,
    _localise,
    _sql_literal,
    http_range,
)

DOC_SCHEMA = pa.schema(
    [
        ("system_id", pa.string()),
        ("source_arm", pa.string()),
        ("document", pa.string()),
        ("sha1", pa.string()),
        ("seq_len", pa.int32()),
        ("num_tokens", pa.int32()),
        ("num_chains", pa.int32()),
        ("chain_ids", pa.list_(pa.string())),
        ("chain_lengths", pa.list_(pa.int32())),
        ("contacts_pre_filter", pa.int32()),
        ("contacts_emitted", pa.int32()),
        ("contacts_emitted_inter_chain", pa.int32()),
        ("contacts_pre_filter_inter_chain", pa.int32()),
        ("truncated", pa.bool_()),
        ("uniprot_R", pa.string()),
        ("uniprot_L", pa.string()),
        ("uniprot_pair", pa.string()),
        ("pdb_id", pa.string()),
        ("cluster_id", pa.string()),
        ("split", pa.string()),
        ("resolution", pa.float64()),
        ("method", pa.string()),
        ("release_date", pa.string()),
        ("total_residues_manifest", pa.int32()),
        ("intermolecular_contacts", pa.int32()),
        ("buried_sasa", pa.float64()),
        ("pair_also_in_afcdb", pa.bool_()),
        # The structure's own sequences, for decontamination. PINDER chains are
        # crystal constructs, so the UniProt sequence is not what was modelled.
        ("sequence_R", pa.string()),
        ("sequence_L", pa.string()),
    ]
)
LEDGER_SCHEMA = pa.schema(
    [("system_id", pa.string()), ("status", pa.string()), ("reason", pa.string())]
)
_MAX_PDB_BYTES = 64 << 20


def fetch_member(url: str, lho: int, csize: int, usize: int) -> str:
    """Read one zip member by offset and inflate it.

    The local header repeats the name and extra lengths, which can differ from
    the central directory's, so the payload offset has to be read here rather
    than assumed.
    """
    head = http_range(url, lho, 30)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{url}@{lho}: not a local file header ({head[:4]!r})")
    method = struct.unpack("<H", head[8:10])[0]
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    payload = http_range(url, lho + 30 + name_len + extra_len, csize)
    if method == 0:
        raw = payload
    elif method == 8:
        raw = zlib.decompressobj(-15).decompress(payload, _MAX_PDB_BYTES)
    else:
        raise RuntimeError(f"{url}@{lho}: unsupported zip compression {method}")
    if usize and len(raw) != usize:
        raise RuntimeError(f"{url}@{lho}: inflated {len(raw)} bytes, expected {usize}")
    return raw.decode("utf-8", "replace")


def _one(row: dict[str, Any]) -> tuple[dict | None, dict]:
    """Fetch, parse and generate one PINDER system.

    A *transport* failure propagates, so the batch defers and is retried. A
    *structure* defect does not: some deposits carry malformed fields (gemmi
    rejects one with "Wrong format for charge: 1O"), and those never parse no
    matter how often the batch is retried. Eight batches deferred on exactly
    that before this distinction existed. A bad structure therefore becomes a
    named ledger row, like every other designed-in rejection.
    """
    gemmi, _zstd, generate = _generator()
    text = fetch_member(
        row["zip_url"], row["local_header_offset"],
        row["compressed_bytes"], row["uncompressed_bytes"],
    )
    try:
        structure = gemmi.read_pdb_string(text)
        structure.setup_entities()
    except Exception as error:  # noqa: BLE001 - a malformed deposit, not our bug
        return None, {
            "system_id": row["system_id"],
            "status": "rejected",
            "reason": f"unparseable_structure:{type(error).__name__}",
        }
    try:
        result = generate(
            structure,
            entry_id=row["system_id"],
            config=_GENERATOR["config"],
            rotamer_library=_GENERATOR["rotamers"],
        )
    except Exception as error:  # noqa: BLE001 - see above; recorded, not swallowed
        return None, {
            "system_id": row["system_id"],
            "status": "rejected",
            "reason": f"ungeneratable_structure:{type(error).__name__}",
        }
    sid = row["system_id"]
    if result is None:
        return None, {"system_id": sid, "status": "rejected", "reason": "unserializable"}
    if result.num_chains < 2:
        return None, {"system_id": sid, "status": "rejected", "reason": "collapsed_to_monomer"}
    if result.contacts_emitted_inter_chain < 1:
        # PINDER counts interface contacts with its own geometry; pyconfind
        # sometimes finds none at the contacts-v1 cut. A complex document with
        # no interface teaches nothing about complexes, so it is filtered here
        # rather than shipped. Unlike AFCDB, a *residue-count* mismatch against
        # the manifest is NOT an error for this arm: these are crystal
        # structures and unresolved residues are normal.
        return None, {"system_id": sid, "status": "rejected", "reason": "no_interface_contacts"}
    seqs = [gemmi.one_letter_code([r.name for r in chain]) for chain in structure[0]]
    return (
        {
            "system_id": sid,
            "source_arm": "pinder",
            "document": result.document,
            "sha1": hashlib.sha1(result.document.encode()).hexdigest(),
            "seq_len": int(result.seq_len),
            "num_tokens": int(result.num_tokens),
            "num_chains": int(result.num_chains),
            "chain_ids": list(result.chain_ids),
            "chain_lengths": [int(v) for v in result.chain_lengths],
            "contacts_pre_filter": int(result.contacts_pre_filter),
            "contacts_emitted": int(result.contacts_emitted),
            "contacts_emitted_inter_chain": int(result.contacts_emitted_inter_chain),
            "contacts_pre_filter_inter_chain": int(result.contacts_pre_filter_inter_chain),
            "truncated": bool(result.truncated),
            "uniprot_R": row["uniprot_R"],
            "uniprot_L": row["uniprot_L"],
            "uniprot_pair": row["uniprot_pair"],
            "pdb_id": row["pdb_id"],
            "cluster_id": row["cluster_id"],
            "split": row["split"],
            "resolution": float(row["resolution"]) if row["resolution"] is not None else None,
            "method": row["method"],
            "release_date": str(row["release_date"]) if row["release_date"] is not None else None,
            "total_residues_manifest": int(row["total_residues"]),
            "intermolecular_contacts": int(row["intermolecular_contacts"]),
            "buried_sasa": float(row["buried_sasa"]) if row["buried_sasa"] is not None else None,
            "pair_also_in_afcdb": bool(row["pair_also_in_afcdb"]),
            "sequence_R": seqs[0] if len(seqs) > 0 else None,
            "sequence_L": seqs[1] if len(seqs) > 1 else None,
        },
        {"system_id": sid, "status": "generated", "reason": "ok"},
    )


def run(
    manifest: str,
    out_dir: str,
    *,
    batch_size: int = 2000,
    fetch_concurrency: int = 8,
    shard_index: int = 0,
    shard_count: int = 1,
    max_deferred: int = 10,
) -> dict[str, Any]:
    """Generate this shard's batches, skipping any already complete."""
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} out of range for {shard_count}")
    rows = (
        duckdb.connect()
        .execute(f"SELECT * FROM read_parquet({_sql_literal(_localise(manifest))}) ORDER BY local_header_offset")
        .arrow().read_all().to_pylist()
    )
    if not rows:
        raise ValueError(f"{manifest}: no rows")
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    mine = [(i, b) for i, b in enumerate(batches)][shard_index::shard_count]

    prefix = str(out_dir).rstrip("/")
    fs, _ = fsspec.core.url_to_fs(prefix)
    for sub in ("documents", "ledger"):
        fs.makedirs(f"{prefix}/{sub}", exist_ok=True)
    try:
        done = {p.split("/")[-1].removesuffix(".parquet") for p in fs.ls(f"{prefix}/ledger", detail=False)}
    except FileNotFoundError:
        done = set()

    _generator()  # warm the JIT and rotamer library before any thread runs
    documents = ledger_rows = 0
    deferred: list[int] = []
    reasons: Counter[str] = Counter()
    began = time.monotonic()

    for n, (bi, batch) in enumerate(mine, start=1):
        stem = f"batch-{bi:06d}"
        if stem in done:
            continue
        try:
            with ThreadPoolExecutor(max_workers=max(1, fetch_concurrency)) as pool:
                results = list(pool.map(_one, batch))
        except (RuntimeError, *_RETRYABLE) as error:
            deferred.append(bi)
            print(f"  DEFERRED {stem}: {error}", flush=True)
            if len(deferred) >= max_deferred:
                raise RuntimeError(
                    f"{len(deferred)} batches deferred; the source looks unavailable"
                ) from error
            continue
        docs = [d for d, _ in results if d is not None]
        led = [entry for _, entry in results]
        if docs:
            with fs.open(f"{prefix}/documents/{stem}.parquet", "wb") as sink:
                pq.write_table(pa.Table.from_pylist(docs, schema=DOC_SCHEMA), sink, compression="zstd")
        with fs.open(f"{prefix}/ledger/{stem}.parquet", "wb") as sink:
            pq.write_table(pa.Table.from_pylist(led, schema=LEDGER_SCHEMA), sink, compression="zstd")
        documents += len(docs)
        ledger_rows += len(led)
        reasons.update(e["reason"] for e in led)
        print(f"[{n}/{len(mine)}] {stem} docs={len(docs)}/{len(batch)}", flush=True)

    elapsed = time.monotonic() - began
    summary = {
        "manifest": manifest,
        "out_dir": prefix,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "batches_in_shard": len(mine),
        "batches_deferred": len(deferred),
        "documents": documents,
        "ledger_rows": ledger_rows,
        "reasons": dict(sorted(reasons.items())),
        "seconds": round(elapsed, 1),
        "seconds_per_document": round(elapsed / max(documents, 1), 4),
    }
    with fs.open(f"{prefix}/pinder_extract-{shard_index:05d}-of-{shard_count:05d}.json", "w") as h:
        h.write(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--fetch-concurrency", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-deferred", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(
        args.manifest, args.out, batch_size=args.batch_size,
        fetch_concurrency=args.fetch_concurrency, shard_index=args.shard_index,
        shard_count=args.shard_count, max_deferred=args.max_deferred,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

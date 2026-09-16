# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage D worker — turn selected AFCDB models into contacts-v1 documents.

The unit of work is one **source tar**, because that is the unit of cost. A tar
has no index, so finding a member means walking 512-byte headers from the start
(``next = off + 512 + roundup(size, 512)``), and that walk costs the same
whether the manifest wants 4 members from the tar or 400. Everything else here
follows from that:

* **Walk once, stop early.** The walk ends as soon as every wanted member in
  the tar has been located, so a manifest concentrated in few tars is cheap and
  one spread thinly is not. This is why ``pilot.py`` ships a separate
  throughput probe at run density.
* **Never stream the whole tar.** Selected members are fetched by byte range.
  Streaming would move the full 48.8 TB release for about a tenth of its
  members, and the homodimer ``chunk_*`` tars carry a redundant ``.pdb.zst``
  per model on top of that.
* **Overlap the walk with the work.** The walk is latency-bound and generation
  is CPU-bound, so located members are handed to a thread pool while the walk
  continues.
* **Warm the JIT once per process.** pyconfind's numba backend costs ~7.5 s on
  its first call and ~0.9 s after, so a per-shard warm-up would dominate a
  small shard.

Fail-loud: a fetch, decompress, parse or generate failure raises and kills the
worker. Only designed-in outcomes -- a structure the generator cannot serialize,
or one that collapses to a single chain -- become named ledger rows. A silently
dropped model is a corpus that quietly disagrees with its own manifest.

    uv run python extract.py --manifest /data/exp294/pilot/throughput_probe.parquet \\
        --out /data/exp294/probe_docs --fetch-concurrency 8
"""

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

MEMBER_TEMPLATE = "AF-{model_id}-model_v1.cif.zst"
_MAX_ATTEMPTS = 8
_RETRYABLE = (urllib.error.URLError, TimeoutError, ConnectionError, OSError)
#: A decompressed AFCDB mmCIF is ~1 MB; this is generous headroom that still
#: refuses a zip bomb.
_MAX_CIF_BYTES = 256 << 20

DOC_SCHEMA = pa.schema(
    [
        ("source_model_key", pa.string()),
        ("model_id", pa.string()),
        ("complex_type", pa.string()),
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
        ("accession_a", pa.string()),
        ("accession_b", pa.string()),
        ("total_residues_manifest", pa.int32()),
        ("quality_ratio", pa.float64()),
        ("ipsae_score", pa.float64()),
        ("pdockq2_score", pa.float64()),
        ("confidence_tier", pa.string()),
        ("source_tar_uri", pa.string()),
        ("member_bytes", pa.int32()),
        ("cif_bytes", pa.int32()),
    ]
)
LEDGER_SCHEMA = pa.schema(
    [
        ("source_model_key", pa.string()),
        ("source_tar_uri", pa.string()),
        ("status", pa.string()),
        ("reason", pa.string()),
    ]
)


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))


def http_range(url: str, start: int, length: int, *, timeout: float = 180.0) -> bytes:
    """Fetch ``[start, start+length)``, retrying refusals and drops."""
    end = start + length - 1
    for attempt in range(_MAX_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise RuntimeError(f"{url}: expected 206, got {response.status}")
                payload = response.read()
            if len(payload) != length:
                raise RuntimeError(
                    f"{url}: asked for {length} bytes at {start}, got {len(payload)}"
                )
            return payload
        except _RETRYABLE as error:
            if attempt == _MAX_ATTEMPTS - 1:
                raise RuntimeError(f"{url}: range {start}+{length} failed ({error})") from error
            _sleep_backoff(attempt)
    raise AssertionError("unreachable")


def walk_members(url: str, wanted: set[str], *, header_reads: list[int]) -> dict[str, tuple[int, int]]:
    """Locate ``wanted`` member names, stopping as soon as all are found.

    Returns ``{name: (offset, size)}``. A tar carries no index, so this is the
    only way to find a member; ``header_reads`` records the cost, which is
    per tar rather than per member and is what the throughput probe measures.
    """
    found: dict[str, tuple[int, int]] = {}
    offset = 0
    reads = 0
    while found.keys() != wanted:
        header = http_range(url, offset, 512)
        reads += 1
        if header == b"\0" * 512:
            break
        name = header[0:100].rstrip(b"\0").decode("utf-8", "replace")
        raw_size = header[124:136].rstrip(b"\0 ").decode("ascii", "replace")
        try:
            size = int(raw_size or "0", 8)
        except ValueError as error:
            raise RuntimeError(f"{url}: bad tar header size {raw_size!r} at {offset}") from error
        if name in wanted:
            found[name] = (offset + 512, size)
        offset += 512 + ((size + 511) // 512) * 512
    header_reads.append(reads)
    return found


@dataclass
class TarResult:
    """What one source tar produced."""

    tar_uri: str
    documents: list[dict[str, Any]] = field(default_factory=list)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    header_reads: int = 0
    member_bytes: int = 0
    walk_seconds: float = 0.0
    work_seconds: float = 0.0


_GENERATOR: dict[str, Any] = {}


def _generator() -> tuple[Any, Any, Any]:
    """Import and JIT-warm the generator once per process, not per shard."""
    if not _GENERATOR:
        import gemmi
        import zstandard
        from marinfold.document_structures.contacts_v1 import (
            GenerationConfig,
            generate_document,
        )

        _GENERATOR["gemmi"] = gemmi
        _GENERATOR["zstd"] = zstandard.ZstdDecompressor()
        _GENERATOR["generate"] = generate_document
        # max_chains mirrors exp222's multimer config so an AFCDB document is
        # directly comparable to a PDB one.
        _GENERATOR["config"] = GenerationConfig(max_chains=60)
    return _GENERATOR["gemmi"], _GENERATOR["zstd"], _GENERATOR["generate"]


def _document_row(result: Any, row: dict[str, Any], member_bytes: int, cif_bytes: int) -> dict[str, Any]:
    return {
        "source_model_key": row["source_model_key"],
        "model_id": row["model_id"],
        "complex_type": row["complex_type"],
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
        "accession_a": row["accession_a"],
        "accession_b": row["accession_b"],
        "total_residues_manifest": int(row["total_residues"]),
        "quality_ratio": float(row["quality_ratio"]),
        "ipsae_score": float(row["ipsae_score"]),
        "pdockq2_score": float(row["pdockq2_score"]),
        "confidence_tier": row["confidence_tier"],
        "source_tar_uri": row["source_tar_uri"],
        "member_bytes": int(member_bytes),
        "cif_bytes": int(cif_bytes),
    }


def _one_model(url: str, row: dict[str, Any], offset: int, size: int) -> tuple[dict | None, dict]:
    """Fetch, decompress, parse and generate one member."""
    gemmi, zstd, generate = _generator()
    payload = http_range(url, offset, size)
    raw = zstd.decompress(payload, max_output_size=_MAX_CIF_BYTES)
    structure = gemmi.read_structure_string(raw.decode())
    structure.setup_entities()
    result = generate(structure, entry_id=row["model_id"], config=_GENERATOR["config"])
    key = row["source_model_key"]
    if result is None:
        return None, {"source_model_key": key, "source_tar_uri": url,
                      "status": "rejected", "reason": "unserializable"}
    if result.num_chains < 2:
        # pyconfind saw fewer chains than the metadata did. Such a document is
        # a monomer and does not belong in a complex corpus.
        return None, {"source_model_key": key, "source_tar_uri": url,
                      "status": "rejected", "reason": "collapsed_to_monomer"}
    return (
        _document_row(result, row, size, len(raw)),
        {"source_model_key": key, "source_tar_uri": url, "status": "generated", "reason": "ok"},
    )


def extract_tar(url: str, rows: list[dict[str, Any]], *, fetch_concurrency: int = 8) -> TarResult:
    """Walk one tar and generate every selected model it holds."""
    out = TarResult(tar_uri=url)
    by_member = {MEMBER_TEMPLATE.format(model_id=r["model_id"]): r for r in rows}
    reads: list[int] = []

    began = time.monotonic()
    located = walk_members(url, set(by_member), header_reads=reads)
    out.walk_seconds = time.monotonic() - began
    out.header_reads = reads[0] if reads else 0

    for name in set(by_member) - set(located):
        row = by_member[name]
        out.ledger.append(
            {"source_model_key": row["source_model_key"], "source_tar_uri": url,
             "status": "rejected", "reason": "member_not_in_tar"}
        )

    began = time.monotonic()
    _generator()  # warm before the pool so every thread sees a hot JIT
    with ThreadPoolExecutor(max_workers=max(1, fetch_concurrency)) as pool:
        futures = [
            pool.submit(_one_model, url, by_member[name], offset, size)
            for name, (offset, size) in sorted(located.items(), key=lambda kv: kv[1][0])
        ]
        for future in futures:
            document, ledger = future.result()
            if document is not None:
                out.documents.append(document)
                out.member_bytes += document["member_bytes"]
            out.ledger.append(ledger)
    out.work_seconds = time.monotonic() - began
    return out


def run(
    manifest: str,
    out_dir: str,
    *,
    fetch_concurrency: int = 8,
    tar_concurrency: int = 1,
    limit_tars: int | None = None,
) -> dict[str, Any]:
    """Extract every model in ``manifest``, grouped by source tar."""
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT source_model_key, model_id, complex_type, accession_a, accession_b,
               total_residues, quality_ratio, ipsae_score, pdockq2_score,
               confidence_tier, source_tar_uri
        FROM read_parquet({_sql_literal(manifest)})
        ORDER BY source_tar_uri, source_model_key
        """
    ).arrow().read_all().to_pylist()
    if not rows:
        raise ValueError(f"{manifest}: no rows to extract")

    by_tar: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_tar.setdefault(row["source_tar_uri"], []).append(row)
    tars = sorted(by_tar)
    if limit_tars:
        tars = tars[:limit_tars]

    fs, _ = fsspec.core.url_to_fs(str(out_dir).rstrip("/"))
    fs.makedirs(str(out_dir).rstrip("/"), exist_ok=True)

    began = time.monotonic()
    results: list[TarResult] = []
    if tar_concurrency > 1:
        with ThreadPoolExecutor(max_workers=tar_concurrency) as pool:
            futures = [
                pool.submit(extract_tar, tar, by_tar[tar], fetch_concurrency=fetch_concurrency)
                for tar in tars
            ]
            for index, future in enumerate(futures, start=1):
                results.append(future.result())
                print(f"[{index}/{len(tars)}] {results[-1].tar_uri.split('/')[-1]} "
                      f"docs={len(results[-1].documents)}", flush=True)
    else:
        for index, tar in enumerate(tars, start=1):
            result = extract_tar(tar, by_tar[tar], fetch_concurrency=fetch_concurrency)
            results.append(result)
            print(f"[{index}/{len(tars)}] {tar.split('/')[-1]} "
                  f"docs={len(result.documents)} reads={result.header_reads} "
                  f"walk={result.walk_seconds:.1f}s work={result.work_seconds:.1f}s", flush=True)
    elapsed = time.monotonic() - began

    documents = [d for r in results for d in r.documents]
    ledger = [entry for r in results for entry in r.ledger]
    prefix = str(out_dir).rstrip("/")
    with fs.open(f"{prefix}/documents.parquet", "wb") as sink:
        pq.write_table(pa.Table.from_pylist(documents, schema=DOC_SCHEMA), sink, compression="zstd")
    with fs.open(f"{prefix}/extract_ledger.parquet", "wb") as sink:
        pq.write_table(pa.Table.from_pylist(ledger, schema=LEDGER_SCHEMA), sink, compression="zstd")

    expected = sum(len(by_tar[t]) for t in tars)
    reasons = Counter(entry["reason"] for entry in ledger)
    header_reads = sum(r.header_reads for r in results)
    member_bytes = sum(r.member_bytes for r in results)
    summary = {
        "manifest": manifest,
        "out_dir": prefix,
        "tars": len(tars),
        "models_requested": expected,
        "documents": len(documents),
        "ledger_rows": len(ledger),
        "reasons": dict(sorted(reasons.items())),
        "header_reads": header_reads,
        "header_reads_per_tar": round(header_reads / max(len(tars), 1), 1),
        "member_bytes": member_bytes,
        "seconds": round(elapsed, 1),
        "walk_seconds": round(sum(r.walk_seconds for r in results), 1),
        "work_seconds": round(sum(r.work_seconds for r in results), 1),
        "seconds_per_tar": round(elapsed / max(len(tars), 1), 2),
        "seconds_per_document": round(elapsed / max(len(documents), 1), 4),
        "documents_per_tar": round(len(documents) / max(len(tars), 1), 1),
        "payload_mb_per_document": round(member_bytes / 1e6 / max(len(documents), 1), 3),
    }
    with fs.open(f"{prefix}/extract.json", "w") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    if len(ledger) != expected:
        raise RuntimeError(
            f"ledger has {len(ledger)} rows for {expected} requested models; "
            "every model must have a terminal status"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fetch-concurrency", type=int, default=8)
    parser.add_argument("--tar-concurrency", type=int, default=1)
    parser.add_argument("--limit-tars", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(
        args.manifest,
        args.out,
        fetch_concurrency=args.fetch_concurrency,
        tar_concurrency=args.tar_concurrency,
        limit_tars=args.limit_tars,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1 — parse the contacts-v1 train corpus into a search index (Pass 1).

For every document in the 2067 local train shards we recover the one-letter
sequence and its 0-based ground-truth contacts (``knn_lib.parse_document``), then
write two artifacts under ``_scratch``:

* ``train_seqs.fasta`` — one record per *document*, header = a globally unique
  ``doc_id`` (``{shard:05d}_{row}``). Documents are NOT deduplicated: a sequence
  that recurs in the train set *should* be able to vote multiple times — that
  recurrence is exactly the memorization signal this baseline probes.
* ``contacts_store/{shard:05d}.parquet`` — ``doc_id, entry_id, seq_len, i[], j[]``
  so Step 3 can look up a hit's contacts without re-parsing.

A unique ``doc_id`` (rather than the raw ``entry_id``) is the FASTA header so that
crops/rounds sharing an accession can't collide into one mmseqs target. Each shard
is processed independently and skipped if both its outputs already exist, so the
whole pass is resumable. ~0.7 s/shard; ~1 min wall on 64 cores.

    uv run python build_train_index.py --scratch _scratch [--limit-shards 20]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from knn_lib import parse_document

TRAIN_GLOB = "contacts_v1-{:05d}-of-02067.parquet"
N_SHARDS = 2067
CONTACTS_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("entry_id", pa.string()),
    ("seq_len", pa.int32()),
    ("i", pa.list_(pa.int32())),
    ("j", pa.list_(pa.int32())),
])


def process_shard(job: tuple[int, str, str]) -> tuple[int, int, str]:
    """Parse one shard -> (fasta part, contacts parquet). Returns (shard, n_docs, status)."""
    shard_idx, train_dir, scratch = job
    src = Path(train_dir) / TRAIN_GLOB.format(shard_idx)
    fasta_path = Path(scratch) / "fasta_parts" / f"{shard_idx:05d}.fasta"
    parquet_path = Path(scratch) / "contacts_store" / f"{shard_idx:05d}.parquet"
    if fasta_path.exists() and parquet_path.exists():
        return shard_idx, 0, "skip"

    table = pq.read_table(src, columns=["document", "entry_id", "seq_len", "n_term_index"])
    documents = table.column("document").to_pylist()
    entry_ids = table.column("entry_id").to_pylist()
    seq_lens = table.column("seq_len").to_pylist()
    n_terms = table.column("n_term_index").to_pylist()

    fasta_lines: list[str] = []
    doc_ids: list[str] = []
    out_entry: list[str] = []
    out_seqlen: list[int] = []
    out_i: list[list[int]] = []
    out_j: list[list[int]] = []
    for row, (doc, eid, slen, nti) in enumerate(zip(documents, entry_ids, seq_lens, n_terms)):
        seq, contacts = parse_document(doc, slen, nti)
        doc_id = f"{shard_idx:05d}_{row}"
        fasta_lines.append(f">{doc_id}\n{seq}\n")
        doc_ids.append(doc_id)
        out_entry.append(eid)
        out_seqlen.append(slen)
        out_i.append([c[0] for c in contacts])
        out_j.append([c[1] for c in contacts])

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp names then rename so a killed run never leaves a half-file
    # that the resume check would mistake for done.
    tmp_fasta = fasta_path.with_suffix(".fasta.tmp")
    tmp_fasta.write_text("".join(fasta_lines))
    batch = pa.table({"doc_id": doc_ids, "entry_id": out_entry, "seq_len": out_seqlen,
                      "i": out_i, "j": out_j}, schema=CONTACTS_SCHEMA)
    tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(batch, tmp_parquet, compression="zstd")
    tmp_fasta.rename(fasta_path)
    tmp_parquet.rename(parquet_path)
    return shard_idx, len(doc_ids), "done"


def concat_fasta(scratch: Path) -> int:
    """Stream all fasta parts into one train_seqs.fasta. Returns the sequence count."""
    parts = sorted((scratch / "fasta_parts").glob("*.fasta"))
    out = scratch / "train_seqs.fasta"
    tmp = out.with_suffix(".fasta.tmp")
    n = 0
    with tmp.open("w") as dst:
        for part in parts:
            text = part.read_text()
            n += text.count("\n>") + (1 if text.startswith(">") else 0)
            dst.write(text)
    tmp.rename(out)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path,
                    default=Path("/home/bizon/exp53_scratch/documents/train"))
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=min(64, mp.cpu_count()))
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="smoke: only the first N shards")
    ap.add_argument("--no-concat", action="store_true",
                    help="skip building the combined fasta (Step 2 needs it eventually)")
    args = ap.parse_args()

    n_shards = min(args.limit_shards or N_SHARDS, N_SHARDS)
    jobs = [(s, str(args.train_dir), str(args.scratch)) for s in range(n_shards)]
    print(f"[index] {n_shards} shards, {args.workers} workers -> {args.scratch}", flush=True)

    t0 = time.time()
    total_docs = done = skipped = 0
    with mp.Pool(args.workers) as pool:
        for k, (shard, ndocs, status) in enumerate(pool.imap_unordered(process_shard, jobs)):
            total_docs += ndocs
            done += status == "done"
            skipped += status == "skip"
            if (k + 1) % 200 == 0 or k + 1 == n_shards:
                print(f"  [{k + 1}/{n_shards}] {total_docs:,} docs "
                      f"({done} parsed, {skipped} skipped, {(time.time() - t0):.0f}s)", flush=True)

    if not args.no_concat:
        print("[index] concatenating fasta parts ...", flush=True)
        n = concat_fasta(args.scratch)
        print(f"[index] train_seqs.fasta: {n:,} sequences", flush=True)
    print(f"[index] done: {total_docs:,} docs parsed in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

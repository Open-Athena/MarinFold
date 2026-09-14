# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage A — the backbone keep-list, from the decontaminated corpus itself.

Metadata only: no structures are fetched and nothing is parsed. Reads
``entry_id`` + provenance columns from the published ``contacts_v1_decontam``
train corpus (#225) and emits a Stage-B manifest.

Two decisions worth stating:

* **The keep-list is read from the decontaminated corpus, not recomputed.**
  Deriving it by re-applying #225's droplist to the raw exp53 manifest would
  give a second, independently-buggy implementation of the same filter. Taking
  the entry ids straight out of the published corpus makes contamination
  inheritance a fact rather than a claim: we only ever redesign a backbone
  that survived #225, and a redesigned sequence cannot reintroduce an eval
  *sequence* because it is a new sequence.
* **Output is sorted by ``seq_len``.** ProteinMPNN batches must be
  exact-length (see ``redesign.batch_by_exact_length``), so a length-sorted
  manifest means each Stage-B shard covers a narrow length band and the
  equal-length groups inside it stay full. Sorting is free here and worth
  a large constant factor on the GPU.

Stage A is **local and standalone**. It is the only step that reads the HF
bucket, which needs ``huggingface_hub>=1.5`` — and that conflicts with the
transformers pin ``marinfold`` pulls in, so it is deliberately not in this
experiment's pyproject. The Iris workers never touch the bucket (they read the
Stage-A manifest from GCS), so run this one out-of-project::

    uv run --no-project --with 'huggingface_hub>=1.5' --with pyarrow \
        --with fsspec python select_backbones.py --out /data/exp266/manifest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# The published decontaminated AFDB corpus (#225). Public bucket, anonymous read.
DEFAULT_CORPUS = (
    "hf://buckets/open-athena/MarinFold/data/document_structures/"
    "contacts_v1_decontam/train"
)

# AFDB's public GCS layout. exp53's manifest stores this column verbatim; it is
# a pure function of entry_id, so we rebuild it instead of joining back to the
# 12,005-shard afdb-24M manifest for one derivable string.
AFDB_URI_TEMPLATE = "gs://public-datasets-deepmind-alphafold-v4/{entry_id}-model_v4.cif"

# Columns carried from the parent corpus so a redesigned document can be traced
# to the native document it came from.
CARRY_COLUMNS = (
    "entry_id",
    "seq_len",
    "global_plddt",
    "struct_cluster_id",
    "seq_cluster_id",
    "split",
    "round",
    "contacts_emitted",   # native contact count, for the density comparison
    "sha1",               # native document hash: pins the parent row exactly
)

# ``contacts_emitted`` / ``sha1`` also exist on the *redesigned* documents
# (contacts-v1's own metadata_row), so the parent's values are renamed here
# rather than colliding on the output row.
RENAME = {"contacts_emitted": "native_contacts_emitted", "sha1": "native_sha1"}


def read_keep_list(corpus: str, columns: tuple[str, ...],
                   max_shards: int | None = None, workers: int = 32) -> pa.Table:
    """Read the provenance columns of every row in the decontaminated corpus.

    The 2,067 shard reads are threaded. Sequentially this is HTTP-latency-bound
    and runs for hours — a 12-shard sample extrapolated to ~26 min but the real
    thing was on track for several. The reads are independent and pyarrow
    releases the GIL while decoding, so a pool is a straight win. Same lesson
    as the pipeline skill's per-row fetch, one level up.
    """
    import fsspec

    fs, _ = fsspec.core.url_to_fs(corpus)
    files = sorted(fs.glob(f"{corpus.rstrip('/')}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {corpus}")
    if max_shards is not None:
        files = files[:max_shards]

    with fs.open(files[0], "rb") as handle:
        present = set(pq.ParquetFile(handle).schema_arrow.names)
    missing = {"entry_id", "seq_len"} - present
    if missing:
        raise ValueError(f"{files[0]}: corpus is missing required column(s) {missing}")
    wanted = [c for c in columns if c in present]

    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    seen = {"n": 0}

    def read_one(path: str) -> pa.Table:
        with fs.open(path, "rb") as handle:
            table = pq.read_table(handle, columns=wanted)
        with lock:
            seen["n"] += 1
            if seen["n"] % 200 == 0 or seen["n"] == len(files):
                print(f"[stage-a] {seen['n']}/{len(files)} shards", flush=True)
        return table

    # pool.map preserves input order, so the concatenated table keeps the
    # corpus's own shard order (exp53 wrote it round-descending).
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tables = list(pool.map(read_one, files))
    return pa.concat_tables(tables)


def build_manifest(table: pa.Table, *, min_len: int, max_len: int) -> pa.Table:
    """Filter to designable lengths, add ``gcs_uri``, sort by ``seq_len``."""
    import pyarrow.compute as pc

    seq_len = pc.cast(table.column("seq_len"), pa.int32())
    table = table.set_column(table.schema.get_field_index("seq_len"), "seq_len", seq_len)

    # A designed-in filter predicate, not a swallowed error: ProteinMPNN is not
    # meaningful below a couple of dozen residues, and contacts-v1 caps at 2000.
    keep = pc.and_(pc.greater_equal(seq_len, min_len), pc.less_equal(seq_len, max_len))
    table = table.filter(keep)

    for old, new in RENAME.items():
        idx = table.schema.get_field_index(old)
        if idx >= 0:
            table = table.set_column(idx, new, table.column(old))

    uris = pa.array(
        [AFDB_URI_TEMPLATE.format(entry_id=e) for e in table.column("entry_id").to_pylist()],
        type=pa.string(),
    )
    table = table.append_column("gcs_uri", uris)
    return table.sort_by([("seq_len", "ascending"), ("entry_id", "ascending")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for the sharded Stage-B manifest.")
    ap.add_argument("--rows-per-shard", type=int, default=20_000,
                    help="Larger than exp53's shards on purpose: exact-length "
                         "ProteinMPNN batches need enough same-length rows in "
                         "one shard to fill the GPU.")
    ap.add_argument("--workers", type=int, default=32,
                    help="Threads reading corpus shards from the HF bucket.")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="Smoke cap: read only the first N corpus shards.")
    ap.add_argument("--sample", type=int, default=None,
                    help="Emit a length-REPRESENTATIVE sample of N backbones by "
                         "striding the sorted manifest. Use this for smokes "
                         "instead of the driver's --num-backbones, which takes "
                         "the first N of a length-sorted manifest and therefore "
                         "hands you only the shortest proteins in the corpus.")
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=2000)
    args = ap.parse_args()

    table = read_keep_list(args.corpus, CARRY_COLUMNS, args.max_shards, args.workers)
    print(f"[stage-a] read {table.num_rows:,} rows from {args.corpus}")

    manifest = build_manifest(table, min_len=args.min_len, max_len=args.max_len)
    dropped = table.num_rows - manifest.num_rows
    print(f"[stage-a] kept {manifest.num_rows:,} "
          f"(dropped {dropped:,} outside [{args.min_len}, {args.max_len}] residues)")

    if args.sample is not None and args.sample < manifest.num_rows:
        # Stride, don't slice. The manifest is sorted by seq_len, so the first N
        # rows are the N shortest proteins — a sample on which ProteinMPNN
        # behaves measurably differently (on the 200 shortest, contacts/residue
        # came out ABOVE native, the opposite of the corpus-wide direction).
        # Striding keeps the length distribution and stays sorted, which is what
        # the exact-length ProteinMPNN batching wants.
        import pyarrow.compute as pc

        step = manifest.num_rows / args.sample
        idx = [min(int(i * step), manifest.num_rows - 1) for i in range(args.sample)]
        manifest = manifest.take(pa.array(idx, type=pa.int64()))
        lengths = manifest.column("seq_len").to_pylist()
        print(f"[stage-a] strided sample: {manifest.num_rows:,} rows, "
              f"seq_len {min(lengths)}–{max(lengths)}, "
              f"mean {sum(lengths)/len(lengths):.1f}")

    args.out.mkdir(parents=True, exist_ok=True)
    total = (manifest.num_rows + args.rows_per_shard - 1) // args.rows_per_shard
    for i in range(total):
        shard = manifest.slice(i * args.rows_per_shard, args.rows_per_shard)
        pq.write_table(
            shard, args.out / f"manifest-{i:05d}-of-{total:05d}.parquet",
            compression="zstd",
        )
    lengths = manifest.column("seq_len").to_pylist()
    print(f"[stage-a] wrote {total} shards to {args.out} "
          f"(seq_len {min(lengths)}–{max(lengths)}, "
          f"{sum(lengths):,} residues, {8 * manifest.num_rows:,} designs to come)")


if __name__ == "__main__":
    main()

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-shard sequence-only document generation — the testable core of exp64.

This module is import-light (no multiprocessing, no huggingface_hub) so it can
be unit-tested against a tiny inline ``.fasta.zst`` without any network or
parallel runtime. ``cli.py`` next door wraps :func:`process_shard` in a
``multiprocessing`` pool over the 61 UniRef50 shards and handles downloads /
resume markers.

It calls **straight into** marinfold's sequence-only generator
(``generate_sequence_only_document`` → a typed parquet row); exp64
re-implements no document logic (issue #64). Each input record is one UniRef50
sequence; the structure-free generator turns its one-letter sequence into a
``<contacts-v1.sequence_only>`` document (the same sequence section contacts-v1
emits, no structure section). Rows are bucketed into an arbitrary
``train``/``val``/``test`` split by hashing the entry id and streamed out as
size-bounded parquet shards per split.
"""

import hashlib
import io
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    generate_sequence_only_document,
)
from marinfold.document_structures.contacts_v1.vocab import (
    CONTEXT_LENGTH,
    NUM_POSITION_INDICES,
)


# Value of the ``structure`` column on every output row — distinguishes these
# rows from the contacts-v1 corpus when the two are read together.
STRUCTURE_NAME = "contacts-v1.sequence_only"

SPLITS = ("train", "val", "test")

# Serializable residue range, inherited from contacts-v1: a chain must have at
# least 2 residues and at most NUM_POSITION_INDICES (2000), so it can be
# uniquely indexed under wrap-around. Sequences outside this are dropped.
MIN_RESIDUES = 2
MAX_RESIDUES = NUM_POSITION_INDICES

# Sequence-only generation never touches pyconfind; the flag is the whole knob.
SEQUENCE_ONLY_CONFIG = GenerationConfig(sequence_only=True)

# Explicit, fully-typed output schema (pinned so every shard/file is
# byte-schema-identical — no per-file type inference drift, no accidental
# all-null columns). Only the columns meaningful for a structure-free corpus;
# the contact-statistics columns of contacts-v1 are omitted by design.
OUTPUT_SCHEMA = pa.schema([
    ("document", pa.string()),
    ("entry_id", pa.string()),
    ("seq_len", pa.int32()),
    ("start_index", pa.int32()),
    ("n_term_index", pa.int32()),
    ("c_term_index", pa.int32()),
    ("num_tokens", pa.int32()),
    ("sha1", pa.string()),
    ("split", pa.string()),
    ("structure", pa.string()),
])


@dataclass
class ShardCounts:
    """Tally of one shard's processing (for the run summary)."""

    shard_index: int
    seen: int = 0
    written: int = 0
    skipped_length: int = 0
    skipped_unserializable: int = 0
    files_written: int = 0
    tokens: int = 0
    per_split: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLITS})
    per_split_tokens: dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in SPLITS}
    )

    def as_dict(self) -> dict:
        row = {
            "shard_index": self.shard_index,
            "seen": self.seen,
            "written": self.written,
            "skipped_length": self.skipped_length,
            "skipped_unserializable": self.skipped_unserializable,
            "files_written": self.files_written,
            "tokens": self.tokens,
        }
        row.update({f"{s}_rows": self.per_split[s] for s in SPLITS})
        row.update({f"{s}_tokens": self.per_split_tokens[s] for s in SPLITS})
        return row


def parse_fasta_header(header: str) -> str:
    """Extract the entry id from a FASTA header line.

    ``>UniRef50_P00350 Cluster: ...`` -> ``UniRef50_P00350`` (the first
    whitespace-delimited token after ``>``).
    """
    return header[1:].split(None, 1)[0]


def iter_fasta(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Yield ``(entry_id, sequence)`` from FASTA text lines.

    Concatenates wrapped sequence lines; blank lines are ignored.
    """
    entry_id: str | None = None
    seq_parts: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if entry_id is not None:
                yield entry_id, "".join(seq_parts)
            entry_id = parse_fasta_header(line)
            seq_parts = []
        else:
            seq_parts.append(line)
    if entry_id is not None:
        yield entry_id, "".join(seq_parts)


def iter_fasta_zst(path: Path | str) -> Iterator[tuple[str, str]]:
    """Stream ``(entry_id, sequence)`` from a zstd-compressed FASTA shard."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        yield from iter_fasta(text)


def assign_split(
    entry_id: str, *, val_per_mille: int = 5, test_per_mille: int = 5
) -> str:
    """Deterministic, arbitrary train/val/test split by hashing the entry id.

    ``bucket = sha1(entry_id) % 1000``; ``[0, test)`` -> ``test``,
    ``[test, test+val)`` -> ``val``, the rest -> ``train``. Defaults give
    ~0.5% val + ~0.5% test. Independent of the contacts-v1 splits — issue #64
    says the split can be arbitrary.
    """
    bucket = int(hashlib.sha1(entry_id.encode()).hexdigest()[:8], 16) % 1000
    if bucket < test_per_mille:
        return "test"
    if bucket < test_per_mille + val_per_mille:
        return "val"
    return "train"


def build_row(result, *, split: str) -> dict:
    """One output-row dict matching :data:`OUTPUT_SCHEMA`."""
    return {
        "document": result.document,
        "entry_id": result.entry_id,
        "seq_len": result.seq_len,
        "start_index": result.start_index,
        "n_term_index": result.n_term_index,
        "c_term_index": result.c_term_index,
        "num_tokens": result.num_tokens,
        "sha1": result.sha1,
        "split": split,
        "structure": STRUCTURE_NAME,
    }


def write_rows(rows: list[dict], path: Path) -> None:
    """Write a list of output rows to one parquet file under :data:`OUTPUT_SCHEMA`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    pq.write_table(table, str(path), compression="zstd")


def process_shard(
    fasta_path: Path | str,
    *,
    shard_index: int,
    out_dir: Path,
    rows_per_file: int = 200_000,
    val_per_mille: int = 5,
    test_per_mille: int = 5,
    limit: int | None = None,
    context_length: int = CONTEXT_LENGTH,
    min_residues: int = MIN_RESIDUES,
    max_residues: int = MAX_RESIDUES,
) -> ShardCounts:
    """Generate sequence-only documents for one UniRef50 ``.fasta.zst`` shard.

    Streams the shard, drops sequences outside ``[min_residues, max_residues]``,
    generates one document per surviving record, buckets it into a split, and
    flushes size-bounded parquet files to ``out_dir/<split>/`` named
    ``uniref50-<shard>-<chunk>.parquet``. Memory is bounded by ``rows_per_file``
    (one buffer per split). ``limit`` caps the number of records *read* from the
    shard (for quick samples; ``None`` = whole shard).
    """
    out_dir = Path(out_dir)
    buffers: dict[str, list[dict]] = {s: [] for s in SPLITS}
    chunk_idx: dict[str, int] = {s: 0 for s in SPLITS}
    counts = ShardCounts(shard_index=shard_index)

    def flush(split: str, *, force: bool) -> None:
        buf = buffers[split]
        if not buf or (len(buf) < rows_per_file and not force):
            return
        path = out_dir / split / f"uniref50-{shard_index:05d}-{chunk_idx[split]:04d}.parquet"
        write_rows(buf, path)
        chunk_idx[split] += 1
        counts.files_written += 1
        buf.clear()

    for entry_id, seq in iter_fasta_zst(fasta_path):
        if limit is not None and counts.seen >= limit:
            break
        counts.seen += 1
        # len(seq) == residue count (one residue per character, including the
        # rare non-standard letters that map to <UNK>), so this matches the
        # bound build_document enforces — and skips the >2000 ones cheaply.
        if not (min_residues <= len(seq) <= max_residues):
            counts.skipped_length += 1
            continue
        result = generate_sequence_only_document(
            seq, entry_id=entry_id, context_length=context_length,
            config=SEQUENCE_ONLY_CONFIG,
        )
        if result is None:
            counts.skipped_unserializable += 1
            continue
        split = assign_split(
            entry_id, val_per_mille=val_per_mille, test_per_mille=test_per_mille
        )
        buffers[split].append(build_row(result, split=split))
        counts.written += 1
        counts.tokens += result.num_tokens
        counts.per_split[split] += 1
        counts.per_split_tokens[split] += result.num_tokens
        flush(split, force=False)

    for split in SPLITS:
        flush(split, force=True)
    return counts

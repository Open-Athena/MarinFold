# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the exp64 sequence-only generation core (no network).

A tiny synthetic ``.fasta.zst`` exercises the full per-shard path: FASTA
parsing, the length filter, the deterministic split, typed-parquet output, and
byte-identity with marinfold's ``generate_sequence_only_document`` (exp64
re-implements no document logic).
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import zstandard

# The experiment dir's modules are siblings of tests/ (flat experiment layout).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_rows  # noqa: E402
from generate_rows import (  # noqa: E402
    OUTPUT_SCHEMA,
    SPLITS,
    STRUCTURE_NAME,
    assign_split,
    build_row,
    iter_fasta,
    iter_fasta_zst,
    parse_fasta_header,
    process_shard,
)
from marinfold.document_structures.contacts_v1 import (  # noqa: E402
    generate_sequence_only_document,
)


def _make_fasta_zst(path: Path, records: list[tuple[str, str]]) -> Path:
    text = "".join(f">{eid}\n{seq}\n" for eid, seq in records)
    path.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
    return path


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------


def test_parse_fasta_header():
    assert parse_fasta_header(">UniRef50_P00350 Cluster: foo bar") == "UniRef50_P00350"
    assert parse_fasta_header(">UniRef50_Q9X0E7") == "UniRef50_Q9X0E7"


def test_iter_fasta_multi_record_and_wrapped_lines():
    text = [
        ">UniRef50_A desc\n", "MKT\n", "AYI\n", "\n",
        ">UniRef50_B other\n", "GGG\n",
    ]
    assert list(iter_fasta(text)) == [
        ("UniRef50_A", "MKTAYI"),
        ("UniRef50_B", "GGG"),
    ]


def test_iter_fasta_zst_roundtrip(tmp_path):
    records = [("UniRef50_A", "MKTAYIAK"), ("UniRef50_B", "GGGSSS")]
    path = _make_fasta_zst(tmp_path / "shard.fasta.zst", records)
    assert list(iter_fasta_zst(path)) == records


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def test_assign_split_is_deterministic():
    assert assign_split("UniRef50_P00350") == assign_split("UniRef50_P00350")


def test_assign_split_boundaries():
    # All-train, all-test, all-val by collapsing the per-mille windows.
    assert assign_split("anything", val_per_mille=0, test_per_mille=0) == "train"
    assert assign_split("anything", val_per_mille=0, test_per_mille=1000) == "test"
    assert assign_split("anything", val_per_mille=1000, test_per_mille=0) == "val"


def test_assign_split_partitions_a_population():
    # With the defaults every id lands in exactly one split, and train
    # dominates (the windows are tiny).
    ids = [f"UniRef50_{i:06d}" for i in range(3000)]
    got = {assign_split(i) for i in ids}
    assert got <= set(SPLITS)
    counts = {s: sum(assign_split(i) == s for i in ids) for s in SPLITS}
    assert counts["train"] > counts["val"] + counts["test"]
    assert sum(counts.values()) == len(ids)


# ---------------------------------------------------------------------------
# Row shape
# ---------------------------------------------------------------------------


def test_build_row_keys_match_schema():
    res = generate_sequence_only_document("MAGFSTKV", entry_id="u")
    row = build_row(res, split="train")
    assert set(row) == set(OUTPUT_SCHEMA.names)
    assert row["structure"] == STRUCTURE_NAME
    assert row["split"] == "train"


# ---------------------------------------------------------------------------
# Per-shard end-to-end
# ---------------------------------------------------------------------------


def test_process_shard_filters_and_writes(tmp_path):
    records = [
        ("UniRef50_ok1", "MAGFST"),          # 6 residues -> kept
        ("UniRef50_ok2", "MKTAYIAKQR"),      # 10 residues -> kept
        ("UniRef50_short", "M"),             # 1 residue -> dropped (too short)
        ("UniRef50_long", "A" * 2001),       # 2001 residues -> dropped (too long)
    ]
    fasta = _make_fasta_zst(tmp_path / "shard-000000.fasta.zst", records)
    out_dir = tmp_path / "out"

    # Force everything to train so the assertions don't depend on hashing.
    counts = process_shard(
        fasta, shard_index=0, out_dir=out_dir,
        val_per_mille=0, test_per_mille=0,
    )
    assert counts.seen == 4
    assert counts.written == 2
    assert counts.skipped_length == 2
    assert counts.skipped_unserializable == 0
    assert counts.per_split["train"] == 2
    assert counts.tokens > 0

    train_files = sorted((out_dir / "train").glob("*.parquet"))
    assert len(train_files) == 1
    assert not (out_dir / "val").exists()
    table = pq.read_table(train_files[0])
    assert table.schema.equals(OUTPUT_SCHEMA)
    assert table.num_rows == 2
    assert set(table.column("entry_id").to_pylist()) == {"UniRef50_ok1", "UniRef50_ok2"}


def test_process_shard_output_is_byte_identical_to_lib(tmp_path):
    records = [("UniRef50_ok1", "MAGFSTKVLI"), ("UniRef50_ok2", "MKTAYIAKQRQ")]
    fasta = _make_fasta_zst(tmp_path / "shard-000000.fasta.zst", records)
    out_dir = tmp_path / "out"
    process_shard(fasta, shard_index=0, out_dir=out_dir,
                  val_per_mille=0, test_per_mille=0)

    by_id = {
        r["entry_id"]: r["document"]
        for r in pq.read_table(out_dir / "train" / "uniref50-00000-0000.parquet").to_pylist()
    }
    for eid, seq in records:
        expected = generate_sequence_only_document(seq, entry_id=eid).document
        assert by_id[eid] == expected


def test_process_shard_limit_caps_records(tmp_path):
    records = [(f"UniRef50_{i}", "MAGFST") for i in range(20)]
    fasta = _make_fasta_zst(tmp_path / "shard-000000.fasta.zst", records)
    counts = process_shard(fasta, shard_index=0, out_dir=tmp_path / "out",
                           val_per_mille=0, test_per_mille=0, limit=5)
    assert counts.seen == 5
    assert counts.written == 5


def test_process_shard_chunks_by_rows_per_file(tmp_path):
    records = [(f"UniRef50_{i}", "MAGFST") for i in range(10)]
    fasta = _make_fasta_zst(tmp_path / "shard-000000.fasta.zst", records)
    out_dir = tmp_path / "out"
    counts = process_shard(fasta, shard_index=3, out_dir=out_dir,
                           val_per_mille=0, test_per_mille=0, rows_per_file=4)
    files = sorted((out_dir / "train").glob("*.parquet"))
    # 10 rows / 4 per file -> 3 files (4 + 4 + 2).
    assert len(files) == 3
    assert counts.files_written == 3
    assert [f.name for f in files] == [
        "uniref50-00003-0000.parquet",
        "uniref50-00003-0001.parquet",
        "uniref50-00003-0002.parquet",
    ]
    assert sum(pq.ParquetFile(f).metadata.num_rows for f in files) == 10

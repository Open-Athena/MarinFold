# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: a tiny synthetic afdb-24M-shaped parquet manifest.

The synthetic manifest has the same columns as ``timodonnell/afdb-24M``
and a hand-built set of structural clusters chosen to exercise every
Stage-A selection rule (round ranking by pLDDT, ``entry_id`` tie break,
the ``<3``-member drop, and the ``seq_len`` pre-filter). No network or
cached HF data is needed.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# afdb-24M columns (cif_content is dummy here — Stage A never reads it).
_AFDB_COLUMNS = (
    "entry_id", "uniprot_accession", "tax_id", "organism_name",
    "global_plddt", "seq_len", "seq_cluster_id", "struct_cluster_id",
    "split", "gcs_uri", "cif_content",
)


def _row(entry_id, cluster, plddt, split, seq_len=100):
    return {
        "entry_id": entry_id,
        "uniprot_accession": entry_id,
        "tax_id": 9606,
        "organism_name": "synthetic",
        "global_plddt": float(plddt),
        "seq_len": int(seq_len),
        "seq_cluster_id": entry_id,
        "struct_cluster_id": cluster,
        "split": split,
        "gcs_uri": f"gs://bucket/{entry_id}.cif",
        "cif_content": "",
    }


# Cluster design (see test assertions):
#   A (train, 6): rounds 0..4 keep top-5 by pLDDT desc; lowest-pLDDT dropped.
#   B (train, 3): kept; rounds follow pLDDT, not insertion order.
#   C (val,   2): dropped (< 3 members).
#   D (test,  4): kept; rounds 0..3.
#   E (train, 3 but one seq_len out of range): only 2 usable -> dropped.
#   F (train, 3, all equal pLDDT): rounds break ties by entry_id ascending.
_ROWS = [
    _row("a1", "A", 90, "train"), _row("a2", "A", 85, "train"),
    _row("a3", "A", 80, "train"), _row("a4", "A", 75, "train"),
    _row("a5", "A", 70, "train"), _row("a6", "A", 65, "train"),
    _row("b1", "B", 50, "train"), _row("b2", "B", 60, "train"),
    _row("b3", "B", 55, "train"),
    _row("c1", "C", 88, "val"), _row("c2", "C", 77, "val"),
    _row("d1", "D", 70, "test"), _row("d2", "D", 80, "test"),
    _row("d3", "D", 60, "test"), _row("d4", "D", 90, "test"),
    _row("e1", "E", 80, "train"), _row("e2", "E", 90, "train", seq_len=5000),
    _row("e3", "E", 70, "train"),
    _row("f3", "F", 80, "train"), _row("f1", "F", 80, "train"),
    _row("f2", "F", 80, "train"),
]


@pytest.fixture
def synthetic_afdb(tmp_path: Path) -> Path:
    """Write the synthetic manifest and return the directory holding it."""
    cols = {c: [r[c] for r in _ROWS] for c in _AFDB_COLUMNS}
    table = pa.table(cols)
    out = tmp_path / "afdb"
    out.mkdir()
    pq.write_table(table, out / "shard_000000.parquet")
    return out

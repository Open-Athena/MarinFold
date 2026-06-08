# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B worker tests — real generation against cached afdb-24M cifs.

These need pyconfind and a locally cached ``timodonnell/afdb-24M`` shard
(the inline ``cif_content`` column); they ``skip`` when neither is
available, so CI without the cache still runs the Stage-A suite. The point
is to prove the worker (a) produces a well-formed output row and (b) is
**byte-identical** to calling ``contacts_v1.generate_document`` directly —
exp53 must not alter the document logic.
"""

import glob
import hashlib
import os

import pytest

pytest.importorskip("pyconfind")
import pyarrow.parquet as pq  # noqa: E402

import generate_rows as gr  # noqa: E402

_AFDB_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--timodonnell--afdb-24M/snapshots"
)


def _cached_rows(n: int) -> list[dict]:
    shards = sorted(glob.glob(os.path.join(_AFDB_CACHE, "*", "**", "*.parquet"), recursive=True))
    if not shards:
        pytest.skip("no cached afdb-24M shard with inline cif_content")
    cols = ["entry_id", "cif_content", "struct_cluster_id", "seq_cluster_id",
            "split", "uniprot_accession", "tax_id", "organism_name"]
    rows = pq.read_table(shards[0], columns=cols).slice(0, n).to_pylist()
    for r in rows:
        r["round"] = 0  # stand-in manifest round
    return rows


def test_generate_shard_inline_schema():
    out = list(gr.generate_shard(_cached_rows(8), cif_text_column="cif_content"))
    assert out, "expected at least one generated document from the first rows"
    row = out[0]
    for key in ("structure", "document", "entry_id", "seq_len", "global_plddt",
                "contacts_emitted", "num_tokens", "sha1",
                "round", "struct_cluster_id", "seq_cluster_id", "split"):
        assert key in row, f"missing output column {key!r}"
    assert row["structure"] == "contacts-v1"
    assert row["document"].startswith("<contacts-v1>")
    assert row["document"].rstrip().endswith("<end>")
    assert row["sha1"] == hashlib.sha1(row["document"].encode()).hexdigest()
    assert row["round"] == 0
    # every emitted row carries the full, stable key set
    assert set(out[0]) == set(out[-1])


def test_byte_identical_to_direct_generate_document():
    from marinfold.document_structures.contacts_v1 import generate_document

    rows = _cached_rows(8)
    out = {r["entry_id"]: r for r in gr.generate_shard(rows, cif_text_column="cif_content")}
    for row in rows:
        structure = gr.structure_from_cif(row["cif_content"], entry_id=row["entry_id"])
        direct = generate_document(structure, entry_id=row["entry_id"])
        if direct is None:
            assert row["entry_id"] not in out
        else:
            assert out[row["entry_id"]]["sha1"] == direct.sha1
            assert out[row["entry_id"]]["document"] == direct.document

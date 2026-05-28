# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""CSR parquet store: roundtrip + byte-identity through the store.

The CSR layer is load-bearing for the on-the-fly training-time dataloader.
These tests pin two contracts:

1. **Field-level roundtrip**: every numeric column survives parquet write +
   read with the right dtype and shape.
2. **End-to-end byte-identity**: generating from a ParsedStructure that came
   out of CSR is exactly the same as generating from the same structure that
   came straight out of gemmi. (CSR is a faithful re-serialization, not a
   lossy compression.)

Together these mean a trainer using ``CSRDocumentDataset`` over a precomputed
CSR shard produces *byte-identical docs* to a trainer running the full
CIF→parse→generate pipeline, with the only difference being that the CSR
path skips ~2.4 ms of parse + the GCS GET per row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import csr_store
import generate
import parse


def _to_arrow_then_back(ps: parse.ParsedStructure, tmp_path: Path
                        ) -> parse.ParsedStructure:
    """Write a single-row CSR parquet to disk, read it back, reconstruct.

    Exercises the actual on-disk roundtrip (not just dict-roundtrip) so any
    arrow type / dtype mismatch in the schema surfaces here, not in production.
    """
    row = csr_store.parsed_structure_to_row(ps)
    schema = csr_store.schema_with_passthrough({})
    batch = csr_store.rows_to_record_batch([row], schema)
    path = tmp_path / "one.parquet"
    pq.write_table(pa.Table.from_batches([batch]), str(path))

    pf = pq.ParquetFile(str(path))
    [batch_read] = list(pf.iter_batches(batch_size=1))
    return csr_store.row_to_parsed_structure(batch_read.to_pylist()[0])


def test_csr_roundtrip_preserves_shapes_and_values(tmp_path, synthetic_cif):
    """Every column round-trips with the exact shape + dtype the in-memory
    ParsedStructure had. This is the *floor* — if it fails, nothing else
    downstream can work."""
    original = parse.parse_cif_content(synthetic_cif, entry_id="AF-RT-F1")
    restored = _to_arrow_then_back(original, tmp_path)

    assert restored.entry_id == original.entry_id
    assert restored.sequence == original.sequence
    assert restored.num_residues == original.num_residues
    assert restored.global_plddt == original.global_plddt
    # All numeric columns: dtype + shape + exact equality (float64 is bit-
    # exact through parquet's float64 logical type).
    for name in ("plddt_per_residue", "cb_or_ca_xyz",
                 "atom_offsets", "atom_name_id", "atom_xyz"):
        a = getattr(original, name)
        b = getattr(restored, name)
        assert a.dtype == b.dtype, f"{name}: dtype drift {a.dtype} → {b.dtype}"
        assert a.shape == b.shape, f"{name}: shape drift {a.shape} → {b.shape}"
        np.testing.assert_array_equal(a, b, err_msg=f"{name} not bit-equal")


def test_doc_byte_identical_through_csr(tmp_path, synthetic_cif):
    """The training-time contract: CIF→doc and CIF→CSR→doc must be identical.

    If this passes, the dataloader can serve docs that match a precomputed
    reference corpus exactly — which is what makes the CSR store a drop-in
    for the doc store *without* losing the byte-identity audit that
    tests/test_byte_identity.py establishes vs exp34.
    """
    entry_id = "AF-RT-F1"
    original = parse.parse_cif_content(synthetic_cif, entry_id=entry_id)
    restored = _to_arrow_then_back(original, tmp_path)

    cfg = generate.GenerationConfig()
    doc_direct = generate.generate_one(original, context_length=8192, cfg=cfg)
    doc_csr = generate.generate_one(restored, context_length=8192, cfg=cfg)

    assert doc_direct == doc_csr, (
        "CSR roundtrip perturbed the generated document — somewhere a "
        "dtype/shape/value drifted. SHA1 direct={}, csr={}".format(
            hashlib.sha1(doc_direct.encode()).hexdigest(),
            hashlib.sha1(doc_csr.encode()).hexdigest(),
        )
    )


def test_dataset_path_discovers_fragments_in_directory(tmp_path, synthetic_cif):
    """One directory path → pyarrow discovers every parquet inside as a
    fragment of one logical dataset.

    Writes two separate shard files into the same directory (mirrors the
    natural Zephyr-write output) and points a single ``CSRDocumentDataset``
    at the directory. The user never lists, globs, or sorts paths — that's
    pyarrow's job.
    """
    from dataset import CSRDocumentDataset

    schema = csr_store.schema_with_passthrough({})
    csr_dir = tmp_path / "csr"
    csr_dir.mkdir()
    for shard_idx in range(2):
        rows = []
        for i in range(3):
            ps = parse.parse_cif_content(
                synthetic_cif, entry_id=f"AF-S{shard_idx}R{i}-F1")
            rows.append(csr_store.parsed_structure_to_row(ps))
        batch = csr_store.rows_to_record_batch(rows, schema)
        pq.write_table(
            pa.Table.from_batches([batch]),
            str(csr_dir / f"shard-{shard_idx:05d}.parquet"),
        )

    ds = CSRDocumentDataset(dataset_path=str(csr_dir), epoch=0)
    outputs = list(ds)
    assert len(outputs) == 6
    # Filename-sorted fragment order, then within-fragment row order.
    expected_ids = [f"AF-S{s}R{r}-F1" for s in (0, 1) for r in range(3)]
    assert sorted(o["entry_id"] for o in outputs) == sorted(expected_ids)


def test_dataset_path_accepts_glob(tmp_path, synthetic_cif):
    """A glob string (``shard-*.parquet``) selects a subset of fragments.

    Confirms ``_open_dataset`` does filesystem-side glob expansion so the
    user can use shell-style patterns without writing their own discovery
    code. The non-matching file ('extra.parquet') is left in place to make
    sure it's actually filtered, not just absent.
    """
    from dataset import CSRDocumentDataset

    schema = csr_store.schema_with_passthrough({})
    for name, entry_id in [
        ("shard-00000.parquet", "AF-A-F1"),
        ("shard-00001.parquet", "AF-B-F1"),
        ("extra.parquet",       "AF-EXCLUDED-F1"),
    ]:
        ps = parse.parse_cif_content(synthetic_cif, entry_id=entry_id)
        batch = csr_store.rows_to_record_batch(
            [csr_store.parsed_structure_to_row(ps)], schema,
        )
        pq.write_table(pa.Table.from_batches([batch]), str(tmp_path / name))

    ds = CSRDocumentDataset(
        dataset_path=str(tmp_path / "shard-*.parquet"), epoch=0,
    )
    outputs = list(ds)
    assert {o["entry_id"] for o in outputs} == {"AF-A-F1", "AF-B-F1"}


def test_dataset_predicate_pushdown_via_filter(tmp_path, synthetic_cif):
    """Predicate pushdown: ``filter=`` cuts decode at the C++ scan layer.

    Stages 4 rows with a ``split`` passthrough column (2 train, 2 val),
    then asserts the filter shrinks the output set without ever
    materializing the filtered-out rows on the Python side. This is the
    main perf win pyarrow.dataset gives us over the per-file loop.
    """
    import pyarrow.compute as pc
    from dataset import CSRDocumentDataset

    passthrough_types = {"split": pa.string()}
    schema = csr_store.schema_with_passthrough(passthrough_types)
    rows = []
    for i, split in enumerate(["train", "val", "train", "val"]):
        ps = parse.parse_cif_content(synthetic_cif, entry_id=f"AF-F{i:02d}-F1")
        row = csr_store.parsed_structure_to_row(ps, passthrough={"split": split})
        rows.append(row)
    batch = csr_store.rows_to_record_batch(rows, schema)
    path = tmp_path / "rich.parquet"
    pq.write_table(pa.Table.from_batches([batch]), str(path))

    ds = CSRDocumentDataset(
        dataset_path=str(path),
        epoch=0,
        # Filter touches a column NOT in CSR_READ_COLUMNS — the dataloader
        # must extend the projection to include it; we do that explicitly
        # so the predicate has something to evaluate against.
        columns=csr_store.CSR_READ_COLUMNS + ("split",),
        filter=pc.field("split") == "train",
    )
    outputs = list(ds)
    assert len(outputs) == 2
    assert {o["entry_id"] for o in outputs} == {"AF-F00-F1", "AF-F02-F1"}


def test_dataset_streams_docs_from_csr(tmp_path, synthetic_cif):
    """End-to-end: write a multi-row CSR shard, point ``CSRDocumentDataset``
    at it, iterate, get a doc per row. The minimal architectural smoke test
    for the on-the-fly training pipeline."""
    from dataset import CSRDocumentDataset

    # Three rows: same structure, three different entry_ids → three docs.
    rows = []
    for i in range(3):
        ps = parse.parse_cif_content(synthetic_cif, entry_id=f"AF-DS{i:02d}-F1")
        rows.append(csr_store.parsed_structure_to_row(ps))
    schema = csr_store.schema_with_passthrough({})
    batch = csr_store.rows_to_record_batch(rows, schema)
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_batches([batch]), str(shard))

    ds = CSRDocumentDataset(dataset_path=str(shard), epoch=0)
    outputs = list(ds)
    assert len(outputs) == 3
    assert [o["entry_id"] for o in outputs] == [
        "AF-DS00-F1", "AF-DS01-F1", "AF-DS02-F1",
    ]
    for o in outputs:
        assert o["document"].startswith("<contacts-and-distances-v2> <begin_sequence>")
        assert o["document"].endswith("<end>")

    # Per-epoch reseeding: same dataset, different epoch → different docs.
    # This is the core "free augmentation" property of on-the-fly generation;
    # if it ever stops holding, the architecture has silently regressed to a
    # static-corpus equivalent.
    ds_e1 = CSRDocumentDataset(dataset_path=str(shard), epoch=1)
    outputs_e1 = list(ds_e1)
    by_id_e0 = {o["entry_id"]: o["document"] for o in outputs}
    by_id_e1 = {o["entry_id"]: o["document"] for o in outputs_e1}
    diffs = sum(by_id_e0[k] != by_id_e1[k] for k in by_id_e0)
    assert diffs == 3, (
        f"epoch reseeding produced identical docs for {3 - diffs}/3 structures"
    )

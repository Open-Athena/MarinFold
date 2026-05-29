# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""CSR parquet store: roundtrip + dataloader infrastructure.

The CSR layer is load-bearing for any future doc-format experiment that
wants on-the-fly training-time generation. These tests pin the
substrate-side contracts; specific doc formats validate their own
byte-identity in their own experiments (e.g. exp5 vs exp34 for v2).

Contracts asserted here:

1. **Column-level roundtrip**: every numeric column survives parquet
   write + read with the right dtype, shape, and exact value. This is
   the foundation — *any* pure function of those columns is byte-equal
   through CSR if this holds (the implication is what lets specific doc
   formats skip a CSR-specific byte-identity test of their own).
2. **Multi-fragment via pyarrow.dataset**: pointing at a directory
   auto-discovers all parquet files in it (the natural Zephyr-write shape).
3. **Glob form**: a shell-style pattern selects a subset of fragments.
4. **Predicate pushdown**: ``filter=`` is evaluated at the C++ scan layer.
5. **Callback extension point**: ``generator=`` accepts any
   ``DocumentGenerator``-shaped callable; the dataloader passes structures
   to it with the per-(entry_id, epoch) reseed already baked in.
6. **Subclass extension point**: overriding ``_structure_to_doc`` can
   enrich the yielded dict (add fields, yield multiple docs per structure).
"""

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import csr_store
import parse


# ---------- helpers --------------------------------------------------------


def _to_arrow_then_back(ps: parse.ParsedStructure, tmp_path: Path
                        ) -> parse.ParsedStructure:
    """Write a single-row CSR parquet to disk, read it back, reconstruct.

    Exercises the actual on-disk roundtrip (not just dict-roundtrip) so
    any arrow type / dtype mismatch in the schema surfaces here, not in
    production.
    """
    row = csr_store.parsed_structure_to_row(ps)
    schema = csr_store.schema_with_passthrough({})
    batch = csr_store.rows_to_record_batch([row], schema)
    path = tmp_path / "one.parquet"
    pq.write_table(pa.Table.from_batches([batch]), str(path))

    pf = pq.ParquetFile(str(path))
    [batch_read] = list(pf.iter_batches(batch_size=1))
    return csr_store.row_to_parsed_structure(batch_read.to_pylist()[0])


def _echo_generator(structure: parse.ParsedStructure) -> str:
    """Trivial doc-format-agnostic test generator.

    Encodes entry_id + a hash over every CSR field into a deterministic
    string. Sensitive enough that any roundtrip drift (any field
    silently changing through write/read) would change the output.
    Used in tests that exercise the dataloader plumbing without taking
    on a v2-specific dependency.
    """
    h = hashlib.sha1()
    h.update(structure.entry_id.encode())
    h.update(",".join(structure.sequence).encode())
    h.update(structure.plddt_per_residue.tobytes())
    h.update(structure.cb_or_ca_xyz.tobytes())
    h.update(structure.atom_offsets.tobytes())
    h.update(structure.atom_name_id.tobytes())
    h.update(structure.atom_xyz.tobytes())
    return f"<doc entry_id={structure.entry_id} sha1={h.hexdigest()[:16]}>"


# ---------- tests ----------------------------------------------------------


def test_csr_roundtrip_preserves_shapes_and_values(tmp_path, synthetic_cif):
    """Every column round-trips with the exact shape + dtype + values the
    in-memory ParsedStructure had. This is the *foundation* — if it fails,
    nothing else downstream can work, and any byte-identity test for any
    doc format depends on it transitively."""
    original = parse.parse_cif_content(synthetic_cif, entry_id="AF-RT-F1")
    restored = _to_arrow_then_back(original, tmp_path)

    assert restored.entry_id == original.entry_id
    assert restored.sequence == original.sequence
    assert restored.num_residues == original.num_residues
    assert restored.global_plddt == original.global_plddt
    for name in ("plddt_per_residue", "cb_or_ca_xyz",
                 "atom_offsets", "atom_name_id", "atom_xyz"):
        a = getattr(original, name)
        b = getattr(restored, name)
        assert a.dtype == b.dtype, f"{name}: dtype drift {a.dtype} → {b.dtype}"
        assert a.shape == b.shape, f"{name}: shape drift {a.shape} → {b.shape}"
        np.testing.assert_array_equal(a, b, err_msg=f"{name} not bit-equal")


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

    ds = CSRDocumentDataset(
        dataset_path=str(csr_dir), generator=_echo_generator, epoch=0,
    )
    outputs = list(ds)
    assert len(outputs) == 6
    expected_ids = [f"AF-S{s}R{r}-F1" for s in (0, 1) for r in range(3)]
    assert sorted(o["entry_id"] for o in outputs) == sorted(expected_ids)


def test_dataset_path_accepts_glob(tmp_path, synthetic_cif):
    """A glob string (``shard-*.parquet``) selects a subset of fragments.

    Confirms ``_open_dataset`` does filesystem-side glob expansion so the
    user can use shell-style patterns without writing their own discovery
    code. The non-matching file is left in place to make sure it's
    actually filtered, not just absent.
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
        dataset_path=str(tmp_path / "shard-*.parquet"),
        generator=_echo_generator, epoch=0,
    )
    outputs = list(ds)
    assert {o["entry_id"] for o in outputs} == {"AF-A-F1", "AF-B-F1"}


def test_dataset_predicate_pushdown_via_filter(tmp_path, synthetic_cif):
    """Predicate pushdown: ``filter=`` cuts decode at the C++ scan layer.

    Stages 4 rows with a ``split`` passthrough column (2 train, 2 val),
    then asserts the filter shrinks the output set without ever
    materializing the filtered-out rows on the Python side. This is the
    main perf win pyarrow.dataset gives us over per-file loops.
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
        generator=_echo_generator,
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


def test_dataset_accepts_custom_generator_callback(tmp_path, synthetic_cif):
    """``generator=`` swaps the doc algorithm without touching the dataloader.

    Exercises the composition extension point: pass any
    ``DocumentGenerator``-shaped callable. Validates that (1) the callback
    actually runs on each structure, (2) its output reaches the consumer
    unchanged, and (3) the dataloader wires per-epoch reseeding into
    ``structure.entry_id`` so callbacks that seed off entry_id get fresh
    draws each epoch without the dataloader knowing anything about them.
    """
    from dataset import CSRDocumentDataset

    schema = csr_store.schema_with_passthrough({})
    rows = [
        csr_store.parsed_structure_to_row(
            parse.parse_cif_content(synthetic_cif, entry_id=f"AF-CB{i:02d}-F1"))
        for i in range(3)
    ]
    batch = csr_store.rows_to_record_batch(rows, schema)
    path = tmp_path / "cb.parquet"
    pq.write_table(pa.Table.from_batches([batch]), str(path))

    # Stub generator returns the (possibly reseeded) entry_id verbatim — so
    # we can read the per-epoch suffix straight out of the yielded doc.
    def echo(structure):
        return f"DOC:{structure.entry_id}"

    ds_e0 = CSRDocumentDataset(dataset_path=str(path), generator=echo, epoch=0)
    out_e0 = list(ds_e0)
    assert [o["document"] for o in out_e0] == [
        "DOC:AF-CB00-F1|e0", "DOC:AF-CB01-F1|e0", "DOC:AF-CB02-F1|e0",
    ]
    # Yielded entry_id is the ORIGINAL (without the |eN suffix) so callers
    # can still join back to manifest rows by entry_id.
    assert [o["entry_id"] for o in out_e0] == ["AF-CB00-F1", "AF-CB01-F1", "AF-CB02-F1"]

    # Different epoch → callback sees the new suffix → different docs, all
    # without the callback caring about epoch.
    ds_e1 = CSRDocumentDataset(dataset_path=str(path), generator=echo, epoch=1)
    out_e1 = list(ds_e1)
    assert [o["document"] for o in out_e1] == [
        "DOC:AF-CB00-F1|e1", "DOC:AF-CB01-F1|e1", "DOC:AF-CB02-F1|e1",
    ]


def test_dataset_subclass_can_enrich_output_dict(tmp_path, synthetic_cif):
    """Subclassing ``_structure_to_doc`` adds fields / multiplies output.

    The callback covers the common case (swap the algorithm); subclassing
    covers the case where the *output shape* should change — e.g. attach a
    token-id field, surface a passthrough column, or yield N variants per
    structure.
    """
    from dataset import CSRDocumentDataset

    class TaggedDataset(CSRDocumentDataset):
        def _structure_to_doc(self, structure):
            for row in super()._structure_to_doc(structure):
                row["seq_len"] = structure.num_residues  # derived field
                yield row

    schema = csr_store.schema_with_passthrough({})
    rows = [
        csr_store.parsed_structure_to_row(
            parse.parse_cif_content(synthetic_cif, entry_id=f"AF-SUB{i:02d}-F1"))
        for i in range(2)
    ]
    batch = csr_store.rows_to_record_batch(rows, schema)
    path = tmp_path / "sub.parquet"
    pq.write_table(pa.Table.from_batches([batch]), str(path))

    ds = TaggedDataset(dataset_path=str(path), generator=_echo_generator, epoch=0)
    outputs = list(ds)
    assert all("seq_len" in o for o in outputs)
    assert all(isinstance(o["seq_len"], int) and o["seq_len"] > 0 for o in outputs)

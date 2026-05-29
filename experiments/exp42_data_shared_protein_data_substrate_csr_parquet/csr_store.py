# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""CSR parquet store for parsed protein structures.

A precomputed, on-the-fly-friendly serialization of :class:`parse.ParsedStructure`
that lets training dataloaders skip both the GCS GET *and* the gemmi parse —
the two costs that make doc generation a batch preprocessing step today.

One CSR parquet row = one ParsedStructure. Per-row columns:

============  ============================  ==========================================
column        arrow type                    contents
============  ============================  ==========================================
entry_id      string                        canonical id (e.g. ``AF-A0A090SHW3-F1``)
sequence      list<string>                  length N — 3-letter AA names, ``UNK`` for non-canonical
plddt         list<float64>                 length N — per-residue mean pLDDT (``-inf`` if empty)
cb_or_ca_xyz  list<float64>                 length 3*N — flattened ``(N, 3)`` CB-or-CA positions
atom_offsets  list<int32>                   length N+1 — CSR offsets into atom_xyz / atom_name_id
atom_name_id  list<uint8>                   length T — indices into ``parse._ATOM_NAMES_TUPLE``
atom_xyz      list<float64>                 length 3*T — flattened ``(T, 3)`` atom positions
global_plddt  float64                       precomputed sum(plddt)/N over Python floats
============  ============================  ==========================================

Plus opt-in passthrough columns (``split``, cluster ids, ``gcs_uri`` …) carried
verbatim from the upstream manifest.

Design notes:

* **2D arrays are flattened** to 1D parquet lists with the leading dimension
  recoverable from a sibling column (``len(sequence)`` for cb_or_ca_xyz_flat,
  ``len(atom_offsets) - 1 ⇒ T = atom_offsets[-1]`` for atom_xyz_flat). One-D
  lists round-trip cleanly through ``ChunkedArray.values`` as zero-copy numpy.
  ``pyarrow.FixedShapeTensorArray`` would also work but adds an extension-type
  dependency on the reader; the flatten trick has none.
* **float64 throughout** for the POC — matches the in-memory layout, makes the
  byte-identity test (CIF → docs vs CIF → CSR → docs) trivially equality, and
  keeps the door open to float32 as a later size optimization.
* **Single zstd-compressed parquet per Zephyr shard**, mirroring exp5's existing
  output sharding pattern, so the same {shard:05d}-of-{total:05d} naming +
  reshard logic in cli.py applies unchanged.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from parse import ParsedStructure


# Arrow schema for the CSR columns — order here is the *output* column order.
# Passthrough columns from the upstream manifest are appended dynamically by
# ``schema_with_passthrough`` so the writer accepts whatever the manifest carries.
CSR_BASE_FIELDS: tuple[tuple[str, pa.DataType], ...] = (
    ("entry_id", pa.string()),
    ("sequence", pa.list_(pa.string())),
    ("plddt", pa.list_(pa.float64())),
    ("cb_or_ca_xyz", pa.list_(pa.float64())),  # flat length 3*N
    ("atom_offsets", pa.list_(pa.int32())),
    ("atom_name_id", pa.list_(pa.uint8())),
    ("atom_xyz", pa.list_(pa.float64())),  # flat length 3*T
    ("global_plddt", pa.float64()),
)


# Minimum column set the zero-copy reader needs to reconstruct a
# ParsedStructure. Used as the default ``columns=`` projection when reading
# through pyarrow.dataset, so passthrough columns (split, cluster ids,
# gcs_uri, …) carried by the parquet schema are NOT decoded unless a caller
# explicitly asks for them. Materially cuts I/O when passthrough columns are
# wide (afdb-1.6M's organism_name and cluster ids are ~50 bytes/row each).
CSR_READ_COLUMNS: tuple[str, ...] = tuple(n for n, _ in CSR_BASE_FIELDS)


def schema_with_passthrough(passthrough_types: dict[str, pa.DataType]) -> pa.Schema:
    """Build the full per-shard schema (CSR core + passthrough columns).

    ``passthrough_types`` is a ``{column_name: arrow_type}`` mapping resolved
    from the manifest schema by the writer's caller. Order is CSR fields first,
    then passthrough in insertion order — keeps the schema stable across shards
    as long as the manifest schema is stable.
    """
    fields = [pa.field(n, t) for n, t in CSR_BASE_FIELDS]
    fields.extend(pa.field(n, t) for n, t in passthrough_types.items())
    return pa.schema(fields)


def parsed_structure_to_row(ps: ParsedStructure, *, passthrough: dict | None = None) -> dict:
    """Convert one ParsedStructure to a dict matching the CSR schema.

    Numpy arrays are passed through directly — pyarrow accepts numpy buffers
    zero-copy when the dtype matches the schema field, so the only real cost
    is the ``ravel()`` on the (N, 3) / (T, 3) arrays (which is a view, not a
    copy, when the array is already C-contiguous).
    """
    row = {
        "entry_id": ps.entry_id,
        "sequence": list(ps.sequence),
        "plddt": ps.plddt_per_residue,            # float64[N]
        "cb_or_ca_xyz": ps.cb_or_ca_xyz.ravel(),  # float64[3*N]
        "atom_offsets": ps.atom_offsets,          # int32[N+1]
        "atom_name_id": ps.atom_name_id,          # uint8[T]
        "atom_xyz": ps.atom_xyz.ravel(),          # float64[3*T]
        "global_plddt": float(ps.global_plddt),
    }
    if passthrough:
        row.update(passthrough)
    return row


def rows_to_record_batch(rows: list[dict], schema: pa.Schema) -> pa.RecordBatch:
    """Pivot a list of row dicts to a column-wise pyarrow RecordBatch."""
    if not rows:
        return pa.RecordBatch.from_pylist([], schema=schema)
    columns = {name: [r.get(name) for r in rows] for name in schema.names}
    return pa.RecordBatch.from_pydict(columns, schema=schema)


# --------------------------------------------------------------------------
# Reader side — used by the on-the-fly dataloader
# --------------------------------------------------------------------------


def row_to_parsed_structure(row: dict) -> ParsedStructure:
    """Inverse of :func:`parsed_structure_to_row`.

    Accepts a plain dict (as produced by ``RecordBatch.to_pylist()`` or by
    column-wise readers that hand back numpy buffers). Reconstructs the
    per-residue numpy views needed by ``atoms_for(i)`` once at load time.
    """
    atom_offsets = np.asarray(row["atom_offsets"], dtype=np.int32)
    atom_name_id = np.asarray(row["atom_name_id"], dtype=np.uint8)
    atom_xyz_flat = np.asarray(row["atom_xyz"], dtype=np.float64)
    atom_xyz = atom_xyz_flat.reshape(-1, 3)
    cb_xyz_flat = np.asarray(row["cb_or_ca_xyz"], dtype=np.float64)
    cb_or_ca_xyz = cb_xyz_flat.reshape(-1, 3)
    plddt = np.asarray(row["plddt"], dtype=np.float64)
    sequence = tuple(row["sequence"])

    # Per-residue numpy views — same layout as parse.from_gemmi materializes.
    per_residue_atoms = tuple(
        (atom_name_id[atom_offsets[i]:atom_offsets[i + 1]],
         atom_xyz[atom_offsets[i]:atom_offsets[i + 1]])
        for i in range(len(sequence))
    )

    return ParsedStructure(
        entry_id=row["entry_id"],
        source=row.get("__source__", f"<csr:{row['entry_id']}>"),
        sequence=sequence,
        plddt_per_residue=plddt,
        cb_or_ca_xyz=cb_or_ca_xyz,
        atom_offsets=atom_offsets,
        atom_name_id=atom_name_id,
        atom_xyz=atom_xyz,
        per_residue_atoms=per_residue_atoms,
        global_plddt=float(row["global_plddt"]),
    )


def _iter_batch_rows_zero_copy(batch):
    """Yield :class:`ParsedStructure` per row of ``batch`` without going
    through Python objects for the numeric columns.

    The hot path. For each ``list<numeric>`` column in the schema we pull two
    buffer views *once per batch*:

    * ``column.values.to_numpy(zero_copy_only=True)`` — a numpy view of the
      flat concatenated values across all rows (arrow's stored representation).
    * ``column.offsets.to_numpy(zero_copy_only=True)`` — the int32 prefix-sum
      that demarcates per-row boundaries inside that flat buffer.

    Per-row reconstruction is then just two int lookups + a numpy slice (a
    view, no copy). The previous implementation went through ``to_pylist`` →
    Python lists of Python numbers → ``np.asarray`` — ~5000 PyObject
    allocations per row that we now skip entirely.

    Strings (the ``sequence`` column) can't be zero-copied to numpy, so we
    materialize those per row — but a row's sequence is only ~250 strings,
    a rounding error in the per-row budget.

    Lifetime note: the numpy arrays yielded here are views into ``batch``'s
    arrow buffers. They stay valid as long as a ParsedStructure (or anything
    derived from its arrays) holds a reference, thanks to arrow's buffer
    refcounting routed through numpy's ``base`` pointer. The dataloader
    consumes one structure at a time, so we never accumulate the whole
    batch's worth of references.
    """
    # Hoist column buffers once per batch — these calls are O(1), they just
    # rewrap the underlying arrow Buffer as a numpy view.
    seq_col = batch.column("sequence")
    entry_id_col = batch.column("entry_id")
    global_plddt_col = batch.column("global_plddt")

    plddt_col = batch.column("plddt")
    plddt_values = plddt_col.values.to_numpy(zero_copy_only=True)
    plddt_off = plddt_col.offsets.to_numpy(zero_copy_only=True)

    cb_col = batch.column("cb_or_ca_xyz")
    cb_values = cb_col.values.to_numpy(zero_copy_only=True)
    cb_off = cb_col.offsets.to_numpy(zero_copy_only=True)

    ao_col = batch.column("atom_offsets")
    ao_values = ao_col.values.to_numpy(zero_copy_only=True)
    ao_off = ao_col.offsets.to_numpy(zero_copy_only=True)

    an_col = batch.column("atom_name_id")
    an_values = an_col.values.to_numpy(zero_copy_only=True)
    an_off = an_col.offsets.to_numpy(zero_copy_only=True)

    ax_col = batch.column("atom_xyz")
    ax_values = ax_col.values.to_numpy(zero_copy_only=True)
    ax_off = ax_col.offsets.to_numpy(zero_copy_only=True)

    for i in range(batch.num_rows):
        atom_offsets = ao_values[ao_off[i]:ao_off[i + 1]]
        atom_name_id = an_values[an_off[i]:an_off[i + 1]]
        atom_xyz = ax_values[ax_off[i]:ax_off[i + 1]].reshape(-1, 3)
        cb_or_ca_xyz = cb_values[cb_off[i]:cb_off[i + 1]].reshape(-1, 3)
        plddt_per_residue = plddt_values[plddt_off[i]:plddt_off[i + 1]]
        sequence = tuple(seq_col[i].as_py())

        # Per-residue numpy views into the CSR slice — same layout
        # ``parse.from_gemmi`` materializes for the ``atoms_for(i)`` API.
        # These are sub-views (views of views), still zero-copy.
        per_residue_atoms = tuple(
            (atom_name_id[atom_offsets[k]:atom_offsets[k + 1]],
             atom_xyz[atom_offsets[k]:atom_offsets[k + 1]])
            for k in range(len(sequence))
        )

        yield ParsedStructure(
            entry_id=entry_id_col[i].as_py(),
            source=f"<csr-batch:row{i}>",
            sequence=sequence,
            plddt_per_residue=plddt_per_residue,
            cb_or_ca_xyz=cb_or_ca_xyz,
            atom_offsets=atom_offsets,
            atom_name_id=atom_name_id,
            atom_xyz=atom_xyz,
            per_residue_atoms=per_residue_atoms,
            global_plddt=global_plddt_col[i].as_py(),
        )


def iter_parsed_structures(parquet_path: str, *, batch_size: int = 256):
    """Stream :class:`ParsedStructure` instances out of a single CSR parquet
    file. Convenience for tests + ad-hoc scripts; production training reads
    through ``iter_parsed_structures_from_dataset`` over a multi-shard
    pyarrow.dataset for predicate pushdown + column projection.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield from _iter_batch_rows_zero_copy(batch)


def iter_parsed_structures_from_dataset(
    ds,
    *,
    batch_size: int = 256,
    columns: tuple[str, ...] | None = None,
    filter=None,
):
    """Stream :class:`ParsedStructure` instances out of a ``pyarrow.dataset.Dataset``.

    Hands off to pyarrow's native scanner — gets us **predicate pushdown**
    (the ``filter`` is evaluated at the C++ scan layer, so non-matching rows
    aren't decoded) and **column projection** (passthrough columns aren't
    even read from disk). The default ``columns`` is ``CSR_READ_COLUMNS``,
    which is what the zero-copy reconstructor actually needs — passing a
    superset is fine but wastes I/O.
    """
    if columns is None:
        columns = CSR_READ_COLUMNS
    scanner = ds.scanner(columns=list(columns), filter=filter, batch_size=batch_size)
    for batch in scanner.to_batches():
        # to_batches() yields zero-row batches at fragment boundaries when
        # the filter excludes everything in that fragment; skip them so the
        # consumer doesn't pay the per-batch hoist cost on empty data.
        if batch.num_rows == 0:
            continue
        yield from _iter_batch_rows_zero_copy(batch)

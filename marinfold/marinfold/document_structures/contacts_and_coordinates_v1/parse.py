# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Structure analysis for the contacts-and-coordinates-v1 format.

Local to this document-structure package. Imported by ``generate.py`` and
``cli.py``.

This format needs two things from a structure that contacts-v1 already
computes — the residue sequence and the pyconfind side-chain contacts —
plus one thing contacts-v1 throws away: the per-atom 3D coordinates of
every heavy atom. So :func:`analyze_coordinates` runs contacts-v1's
:func:`~marinfold.document_structures.contacts_v1.parse.analyze_structure`
for residues + contacts + pLDDT (and its single-chain validation), then
walks the *same* gemmi structure a second time to pull heavy-atom
coordinates, using the same 37-name ``ATOM_NAMES`` vocab and hydrogen /
non-vocab filtering as contacts-and-distances-v1.

The two views are aligned by ``(chain, author-resnum)``: pyconfind reports
positions keyed by author residue number, and so is the gemmi residue
walk, so a residue's contacts and its coordinates refer to the same
physical residue even if the two libraries filtered edge residues
differently. (AFDB single chains — the corpus this format targets — carry
clean, collision-free residue numbers and no insertion codes; a duplicate
``(chain, resnum)`` key is warned about and the first occurrence kept.)
"""

import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from marinfold.document_structures.contacts_and_distances_v1.vocab import ATOM_NAMES
from marinfold.document_structures.contacts_v1.parse import (
    DEFAULT_CIF_COLUMN,
    DEFAULT_ID_COLUMN,
    AnalyzedStructure,
    RawContact,
    ResidueInfo,
    analyze_structure,
    iter_structure_paths,
)


# (atom_name, x, y, z) — one eligible heavy atom's name and coordinates (Å).
AtomCoord = tuple[str, float, float, float]

_VALID_ATOM_NAMES = frozenset(ATOM_NAMES)


@dataclass(frozen=True)
class AnalyzedCoordStructure:
    """Everything the coordinate document builder needs from one structure.

    Attributes:
        entry_id: identifier used for the output record and the deterministic
            generation seed (e.g. ``"AF-P00767-F1-model_v4"``).
        residues: the protein chain in sequence order (contacts-v1's
            :class:`ResidueInfo`, carrying ``seq_index`` / ``resname`` /
            ``resnum`` / ``chain``).
        contacts: pyconfind contacts (degree > 0), sorted by ``(seq_i, seq_j)``.
        atoms_by_seq_index: per-residue eligible heavy atoms, keyed by the
            residue's 0-based ``seq_index``. A residue with no in-vocab heavy
            atoms maps to an empty tuple. Atom names are de-duplicated within
            a residue (first occurrence wins — drops alt-loc doubles).
        global_plddt: mean CA B-factor across the chain (pLDDT for AFDB).
        source_path: provenance for error messages.
    """

    entry_id: str
    residues: tuple[ResidueInfo, ...]
    contacts: tuple[RawContact, ...]
    atoms_by_seq_index: Mapping[int, tuple[AtomCoord, ...]]
    global_plddt: float
    source_path: Path


def _vocab_safe_atoms(gemmi_residue) -> tuple[AtomCoord, ...]:
    """Eligible heavy atoms of one residue: name + (x, y, z).

    Same filtering as contacts-and-distances-v1's ``_vocab_safe_atoms`` —
    drop hydrogens and any atom whose name is outside ``ATOM_NAMES`` (e.g.
    non-canonical residue atoms) — with one addition: a residue name is
    kept only once (first occurrence), so an atom modelled in two alt-loc
    conformations (two ``CA`` rows) contributes a single coordinate rather
    than a duplicate. An eligible atom is identified downstream by its
    ``(residue, atom name)`` pair, so a residue must not offer the same
    name twice.
    """
    out: list[AtomCoord] = []
    seen: set[str] = set()
    for atom in gemmi_residue:
        if atom.is_hydrogen():
            continue
        name = atom.name.strip()
        if name not in _VALID_ATOM_NAMES or name in seen:
            continue
        seen.add(name)
        out.append((name, atom.pos.x, atom.pos.y, atom.pos.z))
    return tuple(out)


def _atoms_by_residue_key(gemmi_structure) -> dict[tuple[str, int], tuple[AtomCoord, ...]]:
    """Map ``(chain, author-resnum)`` to a residue's eligible heavy atoms.

    Walks the **polymer** residues of the first model only —
    ``chain.get_polymer()`` excludes waters and free ligands (whose atom
    names ``O`` / ``C`` / ``N`` are in the vocab and would otherwise shadow a
    same-numbered protein residue's atoms) while keeping modified residues
    like ``MSE`` that pyconfind canonicalizes and places geometry for. This
    matches the residue set pyconfind reports, so the ``(chain, resnum)``
    alignment in :func:`analyze_coordinates` is against protein residues on
    both sides. Residues with no in-vocab heavy atoms are skipped. A repeated
    ``(chain, resnum)`` key — an insertion code, essentially never seen in
    the AFDB single chains this format targets — keeps the first occurrence
    and warns, so the collision is visible rather than silently mixing two
    residues' atoms.
    """
    out: dict[tuple[str, int], tuple[AtomCoord, ...]] = {}
    if len(gemmi_structure) == 0:
        return out
    # Populate polymer/entity metadata so get_polymer() is reliable even for
    # hand-rolled PDBs (AFDB mmCIFs ship with it; the call is idempotent).
    gemmi_structure.setup_entities()
    for chain in gemmi_structure[0]:
        for res in chain.get_polymer():
            atoms = _vocab_safe_atoms(res)
            if not atoms:
                continue
            key = (chain.name, res.seqid.num)
            if key in out:
                warnings.warn(
                    f"duplicate residue key {key} (insertion code?); keeping "
                    f"the first occurrence's atoms",
                    stacklevel=2,
                )
                continue
            out[key] = atoms
    return out


def analyze_coordinates(
    structure,
    *,
    entry_id: str | None = None,
    native_only: bool = True,
    contact_distance: float = 3.0,
    dcut: float = 25.0,
    clash_distance: float = 2.0,
    assembly: int | str | None = None,
    rotamer_library=None,
) -> AnalyzedCoordStructure:
    """Analyze one structure into residues + contacts + per-atom coordinates.

    Args:
        structure: path to a PDB / mmCIF (optionally ``.gz``) file, or an
            already-parsed ``gemmi.Structure``.
        entry_id: override the entry id (defaults to the gemmi structure name
            or file stem with the structure extension stripped).
        native_only / contact_distance / dcut / clash_distance / assembly /
            rotamer_library: pyconfind knobs, passed straight through to
            contacts-v1's :func:`analyze_structure`.

    Raises:
        ValueError: propagated from :func:`analyze_structure` — no protein
            residues, or more than one protein chain (single-chain only).
    """
    import gemmi

    if isinstance(structure, (str, Path)):
        gemmi_structure = gemmi.read_structure(str(structure))
    else:
        gemmi_structure = structure

    # Residues + contacts + pLDDT + single-chain validation, from contacts-v1.
    analyzed: AnalyzedStructure = analyze_structure(
        gemmi_structure,
        entry_id=entry_id,
        native_only=native_only,
        contact_distance=contact_distance,
        dcut=dcut,
        clash_distance=clash_distance,
        assembly=assembly,
        rotamer_library=rotamer_library,
    )

    # Per-atom coordinates from the same gemmi structure, aligned to the
    # residue sequence by (chain, author-resnum).
    atoms_by_key = _atoms_by_residue_key(gemmi_structure)
    atoms_by_seq_index: dict[int, tuple[AtomCoord, ...]] = {
        r.seq_index: atoms_by_key.get((r.chain, r.resnum), ())
        for r in analyzed.residues
    }

    return AnalyzedCoordStructure(
        entry_id=analyzed.entry_id,
        residues=analyzed.residues,
        contacts=analyzed.contacts,
        atoms_by_seq_index=atoms_by_seq_index,
        global_plddt=analyzed.global_plddt,
        source_path=analyzed.source_path,
    )


def _gemmi_structure_from_cif_text(cif_text: str, *, name: str | None = None):
    """Parse mmCIF text (e.g. an afdb-24M ``cif_content`` cell) to a structure."""
    import gemmi

    structure = gemmi.read_structure_string(cif_text)
    if name and not structure.name:
        structure.name = name
    return structure


def _parquet_paths(path: Path) -> list[Path]:
    """Return the ``.parquet`` shard(s) at/under ``path`` (empty if none)."""
    if path.is_file():
        return [path] if path.suffix.lower() == ".parquet" else []
    if path.is_dir():
        return sorted(path.rglob("*.parquet"))
    return []


def iter_parquet_coordinate_structures(
    parquet_path: Path,
    *,
    cif_column: str = DEFAULT_CIF_COLUMN,
    id_column: str | None = DEFAULT_ID_COLUMN,
    native_only: bool = True,
    contact_distance: float = 3.0,
    dcut: float = 25.0,
    clash_distance: float = 2.0,
    assembly: int | str | None = None,
    rotamer_library=None,
    batch_size: int = 64,
) -> Iterator[AnalyzedCoordStructure]:
    """Analyze structures from a parquet shard's ``cif_column`` (afdb-24M layout).

    Row batches are streamed so a downstream ``num_docs`` cap stops early and
    cheaply; ``id_column`` supplies each structure's ``entry_id``. Rows that
    fail to parse or analyze are skipped with a warning.

    Raises:
        ValueError: ``cif_column`` is not present in the parquet schema.
    """
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_path)
    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = set(parquet_file.schema_arrow.names)
    if cif_column not in schema_names:
        raise ValueError(
            f"{parquet_path}: no column {cif_column!r} "
            f"(available: {sorted(schema_names)})"
        )
    use_id = bool(id_column) and id_column in schema_names
    columns = [cif_column] + ([id_column] if use_id else [])

    row_offset = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        cif_arr = batch.column(cif_column)
        id_arr = batch.column(id_column) if use_id else None
        for i in range(batch.num_rows):
            synthetic_entry_id = f"{parquet_path.stem}:row{row_offset + i}"
            raw_entry_id = id_arr[i].as_py() if id_arr is not None else None
            entry_id = (
                synthetic_entry_id
                if raw_entry_id is None
                or (isinstance(raw_entry_id, str) and not raw_entry_id.strip())
                else raw_entry_id
            )
            cif_text = cif_arr[i].as_py()
            if not cif_text:
                warnings.warn(
                    f"skipping {entry_id}: empty {cif_column!r}", stacklevel=2
                )
                continue
            try:
                structure = _gemmi_structure_from_cif_text(cif_text, name=str(entry_id))
                yield analyze_coordinates(
                    structure,
                    entry_id=str(entry_id),
                    native_only=native_only,
                    contact_distance=contact_distance,
                    dcut=dcut,
                    clash_distance=clash_distance,
                    assembly=assembly,
                    rotamer_library=rotamer_library,
                )
            except (ValueError, RuntimeError) as exc:
                warnings.warn(f"skipping {entry_id}: {exc}", stacklevel=2)
                continue
        row_offset += batch.num_rows


def iter_coordinate_structures(
    path: Path,
    *,
    cif_column: str = DEFAULT_CIF_COLUMN,
    id_column: str | None = DEFAULT_ID_COLUMN,
    native_only: bool = True,
    contact_distance: float = 3.0,
    dcut: float = 25.0,
    clash_distance: float = 2.0,
    assembly: int | str | None = None,
    rotamer_library=None,
) -> Iterator[AnalyzedCoordStructure]:
    """Analyze every structure under ``path``.

    ``path`` may be a structure file / directory of them, **or** a
    ``.parquet`` shard / directory of shards in the afdb-24M layout (parquet
    shards take precedence). Inputs that fail to parse or analyze — including
    multi-chain ones — are skipped with a warning.
    """
    path = Path(path)
    parquet_paths = _parquet_paths(path)
    if parquet_paths:
        for parquet_path in parquet_paths:
            yield from iter_parquet_coordinate_structures(
                parquet_path,
                cif_column=cif_column,
                id_column=id_column,
                native_only=native_only,
                contact_distance=contact_distance,
                dcut=dcut,
                clash_distance=clash_distance,
                assembly=assembly,
                rotamer_library=rotamer_library,
            )
        return
    for p in iter_structure_paths(path):
        try:
            yield analyze_coordinates(
                p,
                native_only=native_only,
                contact_distance=contact_distance,
                dcut=dcut,
                clash_distance=clash_distance,
                assembly=assembly,
                rotamer_library=rotamer_library,
            )
        except (ValueError, RuntimeError) as exc:
            warnings.warn(f"skipping {p}: {exc}", stacklevel=2)
            continue

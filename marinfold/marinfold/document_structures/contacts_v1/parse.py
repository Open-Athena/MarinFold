# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""pyconfind-backed structure analysis for the contacts-v1 format.

Local to this document-structure package. Imported by ``generate.py``
and ``cli.py``.

The heavy lifting — placing rotamers and computing side-chain contact
degree — is done by `pyconfind <https://github.com/timodonnell/pyconfind>`_.
We run it in ``native_only=True`` mode (only the actual amino acid at
each position) and read back two things, both keyed off pyconfind's
ordered *position* list so they stay consistent with each other:

- the residue sequence (``positions`` → :class:`ResidueInfo`), and
- the contacts with contact degree > 0 (``report.contacts`` →
  :class:`RawContact`; each references positions by their 0-based index
  in the sequence, with ``seq_i < seq_j``).

``pyconfind`` is imported lazily inside :func:`analyze_structure` so that
``vocab`` and the pure document-builder in ``generate.py`` can be used
(and unit-tested) without it installed.
"""

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from marinfold.document_structures.contacts_and_distances_v1.vocab import AMINO_ACIDS


# Recognised structure-file extensions when given a directory. gemmi
# auto-detects format from the extension.
_STRUCTURE_EXTS = (
    ".cif", ".cif.gz", ".mmcif", ".mmcif.gz",
    ".pdb", ".pdb.gz", ".ent", ".ent.gz",
)

# Default parquet columns for the afdb-24M layout: each row carries the raw
# mmCIF text in ``cif_content`` and the AFDB id in ``entry_id``.
DEFAULT_CIF_COLUMN = "cif_content"
DEFAULT_ID_COLUMN = "entry_id"

# Map every residue name pyconfind treats as protein (its
# LEGAL_RESIDUE_NAMES: the standard 20 plus HIS variants, MSE, and a few
# modified residues) to a canonical-20 three-letter token. Modified
# residues collapse to their parent amino acid; pyconfind already placed
# rotamers for them, so they carry real geometry. Anything unexpected
# falls through to "UNK" in :func:`_canonical_resname`.
_RESNAME_TO_CANONICAL = {aa: aa for aa in AMINO_ACIDS}
_RESNAME_TO_CANONICAL.update({
    "HSD": "HIS", "HSE": "HIS", "HSC": "HIS", "HSP": "HIS", "HIP": "HIS",
    "MSE": "MET",   # selenomethionine
    "CSO": "CYS",   # S-hydroxycysteine
    "SEC": "CYS",   # selenocysteine
    "SEP": "SER",   # phosphoserine
    "TPO": "THR",   # phosphothreonine
    "PTR": "TYR",   # phosphotyrosine
})


def _canonical_resname(resname: str) -> str:
    """Map a (possibly modified) residue name to a canonical-20 token name."""
    return _RESNAME_TO_CANONICAL.get(resname.strip().upper(), "UNK")


@dataclass(frozen=True)
class ResidueInfo:
    """One residue of the (single) protein chain, in sequence order."""

    seq_index: int   # 0-based position in the chain
    resname: str     # canonical 3-letter name (uppercase), or "UNK"
    resnum: int      # author residue number from the structure
    chain: str       # author chain id


@dataclass(frozen=True)
class RawContact:
    """A side-chain contact (contact degree > 0) between two residues.

    ``seq_i`` / ``seq_j`` are 0-based indices into the residue sequence,
    with ``seq_i < seq_j`` (the lower-triangular form pyconfind emits).
    """

    seq_i: int
    seq_j: int
    degree: float


@dataclass(frozen=True)
class AnalyzedStructure:
    """Everything the document builder needs from one structure.

    Attributes:
        entry_id: identifier used for the output record and as the
            deterministic generation seed (e.g. ``"AF-P00767-F1-model_v4"``).
        residues: the protein chain in sequence order.
        contacts: contacts with degree > 0, sorted by ``(seq_i, seq_j)``.
        global_plddt: mean CA B-factor across the chain (pLDDT for AFDB
            inputs; just a confidence proxy otherwise). NaN if no CA atoms.
        source_path: provenance for error messages.
    """

    entry_id: str
    residues: tuple[ResidueInfo, ...]
    contacts: tuple[RawContact, ...]
    global_plddt: float
    source_path: Path


def _strip_structure_ext(stem: str) -> str:
    """Drop a single structure-file extension from an entry id / file name."""
    for ext in (".cif", ".mmcif", ".pdb", ".ent"):
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return stem


def _mean_ca_bfactor(gemmi_structure) -> float:
    """Mean CA B-factor over the first model (≈ mean pLDDT for AFDB)."""
    import math

    if len(gemmi_structure) == 0:
        return math.nan
    values: list[float] = []
    for chain in gemmi_structure[0]:
        for res in chain:
            for atom in res:
                if atom.name.strip() == "CA":
                    values.append(atom.b_iso)
                    break
    if not values:
        return math.nan
    return sum(values) / len(values)


def analyze_structure(
    structure,
    *,
    entry_id: str | None = None,
    native_only: bool = True,
    contact_distance: float = 3.0,
    dcut: float = 25.0,
    clash_distance: float = 2.0,
    assembly: int | str | None = None,
    rotamer_library=None,
) -> AnalyzedStructure:
    """Run pyconfind on one structure and return an :class:`AnalyzedStructure`.

    Args:
        structure: path to a PDB / mmCIF (optionally ``.gz``) file, or an
            already-parsed ``gemmi.Structure``.
        entry_id: override the entry id (defaults to the gemmi structure
            name or file stem with the structure extension stripped).
        native_only: pyconfind native-only mode (SPEC default ``True``).
        contact_distance / dcut / clash_distance: pyconfind geometry knobs
            (C++ confind defaults).
        assembly: biological assembly passed through to pyconfind. ``None``
            means "use the asymmetric unit as-is" rather than implicitly
            expanding assembly 1.
        rotamer_library: passed through to ``pyconfind.analyze``; ``None``
            auto-downloads + caches the Dunbrack 2010 library once.

    Raises:
        ValueError: the structure has no protein residues, or has more than
            one protein chain (multi-chain support is future work per SPEC).
    """
    import gemmi
    from pyconfind import analyze

    if isinstance(structure, (str, Path)):
        source_path = Path(structure)
        gemmi_structure = gemmi.read_structure(str(source_path))
        default_id = gemmi_structure.name or source_path.name
    else:
        gemmi_structure = structure
        source_path = Path(getattr(structure, "name", "") or "<gemmi.Structure>")
        default_id = getattr(structure, "name", "") or "structure"
    resolved_entry_id = entry_id if entry_id is not None else _strip_structure_ext(default_id)

    global_plddt = _mean_ca_bfactor(gemmi_structure)

    analysis = analyze(
        gemmi_structure,
        native_only=native_only,
        contact_distance=contact_distance,
        dcut=dcut,
        clash_distance=clash_distance,
        assembly=assembly,
        rotamer_library=rotamer_library,
    )

    positions = analysis.positions
    if not positions:
        raise ValueError(f"{source_path}: no protein residues")
    chains = {p.position.chain for p in positions}
    if len(chains) != 1:
        raise ValueError(
            f"{source_path}: expected a single protein chain, found "
            f"{len(chains)} ({sorted(chains)}). contacts-v1 is single-chain "
            f"for now (see SPEC.md)."
        )

    residues = tuple(
        ResidueInfo(
            seq_index=i,
            resname=_canonical_resname(p.position.resname),
            resnum=p.position.resnum,
            chain=p.position.chain,
        )
        for i, p in enumerate(positions)
    )
    contacts = tuple(
        RawContact(seq_i=c.pos_i, seq_j=c.pos_j, degree=float(c.degree))
        for c in analysis.report.contacts
    )
    return AnalyzedStructure(
        entry_id=resolved_entry_id,
        residues=residues,
        contacts=contacts,
        global_plddt=global_plddt,
        source_path=source_path,
    )


def iter_structure_paths(path: Path) -> Iterator[Path]:
    """Yield structure files under ``path`` (a file, or a directory, recursive)."""
    path = Path(path)
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(name.endswith(ext) for ext in _STRUCTURE_EXTS):
            yield p


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


def iter_parquet_analyzed_structures(
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
) -> Iterator[AnalyzedStructure]:
    """Analyze structures from a parquet shard's ``cif_column`` (afdb-24M layout).

    Row batches are streamed (so a ``num_docs`` cap downstream stops early
    and cheaply), and ``id_column`` supplies each structure's ``entry_id``
    (the deterministic generation seed). When ``id_column`` is missing /
    empty a synthetic ``<stem>:row<N>`` id is used. Rows that fail to parse
    or analyze are skipped with a warning.

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
            entry_id = (
                id_arr[i].as_py() if id_arr is not None
                else f"{parquet_path.stem}:row{row_offset + i}"
            )
            cif_text = cif_arr[i].as_py()
            if not cif_text:
                warnings.warn(
                    f"skipping {entry_id}: empty {cif_column!r}", stacklevel=2
                )
                continue
            try:
                structure = _gemmi_structure_from_cif_text(cif_text, name=str(entry_id))
                yield analyze_structure(
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


def iter_analyzed_structures(
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
) -> Iterator[AnalyzedStructure]:
    """Analyze every structure under ``path``.

    ``path`` may be a structure file / a directory of them, **or** a
    ``.parquet`` shard / directory of shards in the afdb-24M layout — in
    which case structures are read from the ``cif_column`` mmCIF text and
    ids from ``id_column``. (If a directory holds parquet shards they take
    precedence over loose structure files.) Inputs that fail to parse or
    analyze — including multi-chain ones — are skipped with a warning.
    """
    path = Path(path)
    parquet_paths = _parquet_paths(path)
    if parquet_paths:
        for parquet_path in parquet_paths:
            yield from iter_parquet_analyzed_structures(
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
            yield analyze_structure(
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

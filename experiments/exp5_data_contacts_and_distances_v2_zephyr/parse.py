# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Columnar gemmi-backed parser for contacts-and-distances-v2 generation.

A from-scratch design — no backward compatibility with marinfold's
``ParsedStructure`` shape. Every per-residue / per-atom array lives at
the *structure* level as numpy arrays, so the doc-generation hot paths
in :mod:`generate` can vectorize (``np.triu_indices`` over the CB-CB
distance matrix, etc.) instead of walking gemmi proxy objects from
Python.

The atom data is stored CSR-style — one flat ``atom_xyz`` array of
shape ``(T, 3)`` plus an ``atom_offsets`` array of length ``N + 1`` so
the atoms of residue ``i`` are ``atom_xyz[atom_offsets[i] : atom_offsets[i + 1]]``.
This is the layout that round-trips zero-copy to ``pyarrow.RecordBatch``
if we ever want a precomputed parquet store.

Determinism contract (versus exp34's reference parser):

* ``plddt_per_residue[i]`` is the left-fold mean of every non-hydrogen
  atom's ``b_iso`` for residue ``i`` (``-inf`` if no heavy atoms) —
  byte-identical Python FP to the legacy ``_residue_plddt``.
* ``cb_or_ca_xyz[i]`` is the first CB seen in residue ``i``, falling
  back to the *last* CA seen for non-GLY (matches the legacy
  ``cb_or_ca_position`` scan order where ``fallback_ca`` is
  overwritten); for GLY it's the *first* CA. ``NaN`` row when neither
  atom is present.
* ``global_plddt`` is computed as ``sum(values_list) / N`` over Python
  floats — *not* ``np.mean`` — so the FP reduction order matches
  exp34's ``ParsedStructure.global_plddt`` property exactly.

These three invariants are what keeps v2 docs byte-identical to the
exp34 reference (see ``tests/test_byte_identity.py``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Self

import fsspec
import numpy as np

from vocab import AMINO_ACIDS, ATOM_NAMES


# --------------------------------------------------------------------------
# Vocab lookups, hoisted once at module load
# --------------------------------------------------------------------------

_CANONICAL_20 = frozenset(AMINO_ACIDS)

# ATOM_NAMES is small (~37 entries) and fixed. Encoding each atom as an
# index into this tuple lets us store atom_name_id as uint8 and recover
# the string via O(1) tuple indexing in generate.py.
_ATOM_NAMES_TUPLE: tuple[str, ...] = tuple(ATOM_NAMES)
_ATOM_NAME_TO_ID: dict[str, int] = {n: i for i, n in enumerate(_ATOM_NAMES_TUPLE)}

_NAN3: tuple[float, float, float] = (float("nan"), float("nan"), float("nan"))


# --------------------------------------------------------------------------
# ParsedStructure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedStructure:
    """One polymer chain in a vectorization-friendly columnar layout.

    See module docstring for the determinism contract that keeps generated
    v2 docs byte-identical to the exp34 reference implementation.
    """

    entry_id: str  # canonical id, e.g. "AF-A0A090SHW3-F1" (matches afdb-1.6M manifest)
    source: str  # URI or path, for provenance

    # Per-residue arrays — all length N.
    sequence: tuple[str, ...]  # 3-letter canonical AA names, "UNK" for non-canonical
    plddt_per_residue: np.ndarray  # float64[N]
    cb_or_ca_xyz: np.ndarray  # float64[N, 3], NaN row when neither CB nor CA present

    # Flat CSR-style atom table — atoms for residue i live at
    # atom_xyz[atom_offsets[i] : atom_offsets[i+1]] (same slice for atom_name_id).
    atom_offsets: np.ndarray  # int32[N+1]
    atom_name_id: np.ndarray  # uint8[T] — index into ``parse._ATOM_NAMES_TUPLE``
    atom_xyz: np.ndarray  # float64[T, 3]

    # Precomputed once at parse time so generate.py doesn't have to choose
    # between FP-fidelity (sum/N over Python floats) and vectorization.
    global_plddt: float

    @classmethod
    def from_gemmi(cls, structure, *, entry_id: str, source: str, require_single_chain: bool) -> Self:
        """Single-pass extraction from a parsed gemmi ``Structure``."""
        structure.setup_entities()
        if len(structure) == 0:
            raise ValueError(f"{source}: no models in structure")
        model = structure[0]
        polymer_chains = [ch for ch in model if ch.get_polymer()]
        if not polymer_chains:
            raise ValueError(f"{source}: no polymer chain")
        if require_single_chain and len(polymer_chains) != 1:
            raise ValueError(
                f"{source}: expected single polymer chain, found {len(polymer_chains)}."
            )
        chain = polymer_chains[0]
        polymer = chain.get_polymer()

        # Fast-path detection (AFDB-shape): one chain in the model AND every
        # residue in that chain is a polymer residue (no waters / ligands). When
        # both hold, ``model.all()`` yields exactly the polymer atoms in order
        # — see the gemmi maintainer's advice in
        # https://github.com/project-gemmi/gemmi/issues/314 — and we can avoid
        # the nested-loop overhead. Otherwise fall back to polymer iteration so
        # we don't pick up solvent/ligand atoms.
        n_polymer = sum(1 for _ in polymer)
        fast_path = len(model) == 1 and sum(1 for _ in chain) == n_polymer

        sequence: list[str] = []
        plddt_values: list[float] = []
        cb_or_ca_rows: list[tuple[float, float, float]] = []
        atom_name_id_list: list[int] = []
        atom_xyz_list: list[tuple[float, float, float]] = []
        atom_offsets: list[int] = [0]

        def _flush_residue(name: str, b_values: list[float],
                           cb: tuple[float, float, float] | None,
                           ca_first: tuple[float, float, float] | None,
                           ca_last: tuple[float, float, float] | None) -> None:
            # FP-identical to exp34's _residue_plddt (left-fold sum over Python
            # floats divided by count; -inf if no heavy atoms).
            plddt = sum(b_values) / len(b_values) if b_values else float("-inf")
            if name == "GLY":
                chosen = ca_first
            else:
                chosen = cb if cb is not None else ca_last
            sequence.append(name)
            plddt_values.append(plddt)
            cb_or_ca_rows.append(chosen if chosen is not None else _NAN3)
            atom_offsets.append(len(atom_name_id_list))

        if fast_path:
            prev_res = None
            name = ""
            b_values: list[float] = []
            cb: tuple[float, float, float] | None = None
            ca_first: tuple[float, float, float] | None = None
            ca_last: tuple[float, float, float] | None = None
            for cra in model.all():
                if cra.residue is not prev_res:
                    if prev_res is not None:
                        _flush_residue(name, b_values, cb, ca_first, ca_last)
                    prev_res = cra.residue
                    raw = cra.residue.name.strip()
                    name = raw if raw in _CANONICAL_20 else "UNK"
                    b_values = []
                    cb = None
                    ca_first = None
                    ca_last = None
                atom = cra.atom
                if atom.is_hydrogen():
                    continue
                b_values.append(atom.b_iso)
                atom_name = atom.name.strip()
                name_id = _ATOM_NAME_TO_ID.get(atom_name)
                if name_id is None:
                    continue
                x, y, z = atom.pos.tolist()  # one nanobind call vs three attr lookups
                atom_name_id_list.append(name_id)
                atom_xyz_list.append((x, y, z))
                if atom_name == "CB":
                    if cb is None:
                        cb = (x, y, z)
                elif atom_name == "CA":
                    if ca_first is None:
                        ca_first = (x, y, z)
                    ca_last = (x, y, z)
            if prev_res is not None:
                _flush_residue(name, b_values, cb, ca_first, ca_last)
        else:
            for res in polymer:
                raw = res.name.strip()
                name = raw if raw in _CANONICAL_20 else "UNK"
                b_values = []
                cb = None
                ca_first = None
                ca_last = None
                for atom in res:
                    if atom.is_hydrogen():
                        continue
                    b_values.append(atom.b_iso)
                    atom_name = atom.name.strip()
                    name_id = _ATOM_NAME_TO_ID.get(atom_name)
                    if name_id is None:
                        continue
                    x, y, z = atom.pos.tolist()
                    atom_name_id_list.append(name_id)
                    atom_xyz_list.append((x, y, z))
                    if atom_name == "CB":
                        if cb is None:
                            cb = (x, y, z)
                    elif atom_name == "CA":
                        if ca_first is None:
                            ca_first = (x, y, z)
                        ca_last = (x, y, z)
                _flush_residue(name, b_values, cb, ca_first, ca_last)

        if not sequence:
            raise ValueError(f"{source}: no residues parsed")

        # global_plddt: sum/N over the Python float list (NOT np.mean) so the
        # FP reduction order matches exp34's ParsedStructure.global_plddt
        # property — keeps RNG-seeded doc content byte-identical.
        global_plddt = sum(plddt_values) / len(plddt_values)

        return cls(
            entry_id=entry_id,
            source=source,
            sequence=tuple(sequence),
            plddt_per_residue=np.asarray(plddt_values, dtype=np.float64),
            cb_or_ca_xyz=np.asarray(cb_or_ca_rows, dtype=np.float64),
            atom_offsets=np.asarray(atom_offsets, dtype=np.int32),
            atom_name_id=(
                np.asarray(atom_name_id_list, dtype=np.uint8)
                if atom_name_id_list else np.empty(0, dtype=np.uint8)
            ),
            atom_xyz=(
                np.asarray(atom_xyz_list, dtype=np.float64).reshape(-1, 3)
                if atom_xyz_list else np.empty((0, 3), dtype=np.float64)
            ),
            global_plddt=global_plddt,
        )

    @property
    def num_residues(self) -> int:
        return len(self.sequence)

    def atoms_for(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Atoms of residue ``i`` (0-based): ``(name_ids[K], xyz[K, 3])``."""
        start = int(self.atom_offsets[i])
        end = int(self.atom_offsets[i + 1])
        return self.atom_name_id[start:end], self.atom_xyz[start:end]


# --------------------------------------------------------------------------
# Public parsers
# --------------------------------------------------------------------------


def parse_cif_content(data, entry_id: str, *, source: str | None = None,
                      require_single_chain: bool = True) -> ParsedStructure:
    """Parse an mmCIF document held in memory (``str`` or ``bytes``)."""
    import gemmi

    structure = gemmi.read_structure_string(data, format=gemmi.CoorFormat.Mmcif)
    return ParsedStructure.from_gemmi(
        structure,
        entry_id=entry_id,
        source=source or f"<cif_content:{entry_id}>",
        require_single_chain=require_single_chain,
    )


def parse_cif_from_uri(uri: str, entry_id: str, *,
                       require_single_chain: bool = True) -> ParsedStructure:
    """Fetch ``uri`` (any fsspec URL) and parse the bytes as an mmCIF document.

    Transparent gzip via ``compression='infer'``. The ``entry_id`` arrives
    from the manifest (e.g. afdb-1.6M's ``entry_id`` column) and is the
    canonical id used to seed generation — it does NOT need to match the
    URI basename.
    """
    with fsspec.open(uri, "rb", compression="infer") as f:
        data = f.read()
    return parse_cif_content(data, entry_id=entry_id, source=uri,
                             require_single_chain=require_single_chain)


def try_parse_cif_content(data, entry_id: str, *, source: str | None = None
                          ) -> ParsedStructure | None:
    """``.map``-friendly :func:`parse_cif_content`: ``None`` on failure."""
    try:
        return parse_cif_content(data, entry_id=entry_id, source=source)
    except ValueError as exc:
        warnings.warn(f"skipping {source or entry_id}: {exc}", stacklevel=2)
        return None


def try_parse_cif_from_uri(uri: str, entry_id: str) -> ParsedStructure | None:
    """``.map``-friendly :func:`parse_cif_from_uri`: ``None`` on failure.

    Catches ``OSError`` (gs:// 404 / transient HTTP / DNS) in addition to
    ``ValueError`` so a single missing object can't kill a Zephyr worker.
    """
    try:
        return parse_cif_from_uri(uri, entry_id=entry_id)
    except (ValueError, OSError) as exc:
        warnings.warn(f"skipping {uri}: {exc}", stacklevel=2)
        return None

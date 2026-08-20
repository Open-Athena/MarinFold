# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The canonical structure-file contract shared by ground truth and predictions.

Everything in this experiment — the ground-truth bundle
(``prepare_gt_structures.py``), any predictor written against Component 1,
and the scorer (``score_structures.py``) — exchanges **one PDB file per
protein** obeying the contract below. Getting a coordinate file into this
shape is the whole interface; the scorer needs nothing else.

**The contract.**

- One file per protein, named ``<stem>.pdb``, where ``stem`` is the eval-set
  identifier from the exp74/exp78 manifests (e.g. ``5sbj_A``, ``1mj0_A``).
- A single chain, ``A``.
- ``resSeq`` is the **1-based index into the input sequence** the model was
  conditioned on — *not* the author numbering of the ground-truth structure.
  This is what makes prediction and ground truth directly comparable and
  lets US-align's ``-TMscore 1`` (equivalent residues = equal residue index)
  do the right thing with no alignment step at scoring time.
- ``resName`` is a canonical three-letter amino-acid name. It is
  informational: **atom identity is the pair ``(res_id, atom_name)``**, and
  that is what the scorer matches on.
- ``name`` (atom name) is one of the 37 heavy-atom names in the
  contacts-and-crops-v1 / contacts-and-distances-v1 vocabulary
  (:data:`ATOM_NAMES`). No hydrogens, no atoms outside the vocabulary —
  those are exactly the atoms the document format can express.
- Atoms the predictor did not place are simply **absent**. There are no
  placeholder coordinates and no occupancy-0 rows; absence is the signal.
- ``b_factor`` carries the predictor's **positional uncertainty in Å** for
  that atom (see :data:`UNCERTAINTY_UNPLACED` for the convention when a
  predictor has no estimate). It is what the scorer's ``--refined-max-sigma``
  threshold splits on to report "fraction of atoms at fine resolution vs
  coarse-box-only", the coverage story issue #174 asks for. Ground-truth
  files carry ``0.0``.

Coordinates live in whatever frame the producer chose. Every metric here is
either superposition-based (RMSD, TM-score) or superposition-free (lDDT), so
the random rotation + translation that contacts-and-crops-v1 applies to each
document is irrelevant — which is the point of the format's frame
augmentation.
"""

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from marinfold.document_structures.contacts_and_distances_v1.vocab import (
    AMINO_ACIDS,
    ATOM_NAMES,
)

# The heavy-atom names the document formats can express, as a set for
# membership tests. Anything else in a file is a contract violation.
VALID_ATOM_NAMES = frozenset(ATOM_NAMES)

# Canonical residue names, plus the "UNK" the contacts-v1 parse layer falls
# back to for residues it cannot canonicalize.
VALID_RES_NAMES = frozenset(AMINO_ACIDS) | {"UNK"}

# Every vocabulary atom name starts with its element symbol (C / N / O / S),
# so the element column is derivable and never has to be carried around.
_ELEMENTS = frozenset("CNOS")

# Predictors that place an atom but have no calibrated uncertainty should use
# this sentinel in the B-factor column: it is larger than any real per-atom
# sigma and sorts the atom into the "coarse" bucket. A Pass-1-only atom in
# contacts-and-crops-v1 is localized to a 10 Å box, so ~2.9 Å (the standard
# deviation of a uniform distribution over a 10 Å cell) is the natural value
# for it; this sentinel is for "we genuinely do not know".
UNCERTAINTY_UNPLACED = 99.0

CHAIN_ID = "A"

# Atom order within a residue: the backbone in the conventional PDB order
# first, then the remaining vocabulary atoms in vocabulary order. Purely
# cosmetic — atom identity is (res_id, atom_name) — but it keeps the files
# readable and matches what every other structure tool emits.
_ATOM_ORDER = {
    name: i
    for i, name in enumerate(
        ["N", "CA", "C", "O"]
        + [n for n in ATOM_NAMES if n not in {"N", "CA", "C", "O"}]
    )
}


def element_of(atom_name: str) -> str:
    """Element symbol of a vocabulary heavy-atom name (its first character)."""
    element = atom_name[0]
    if element not in _ELEMENTS:
        raise ValueError(f"atom name {atom_name!r} does not start with C/N/O/S")
    return element


def build_atom_array(
    atoms: list[tuple[int, str, str, float, float, float, float]],
) -> AtomArray:
    """Assemble a canonical :class:`AtomArray` from flat atom records.

    Args:
        atoms: one ``(res_id, res_name, atom_name, x, y, z, uncertainty)``
            tuple per atom. ``res_id`` is the 1-based input-sequence index;
            ``uncertainty`` is the per-atom positional sigma in Å that lands
            in the B-factor column.

    Returns:
        An :class:`AtomArray` with the annotations the contract requires,
        sorted by ``(res_id, backbone-first atom order)`` so two files listing
        the same atoms are byte-comparable.

    Raises:
        ValueError: an atom name is outside the vocabulary, a residue name is
            not canonical, or the same ``(res_id, atom_name)`` appears twice.
    """
    seen: set[tuple[int, str]] = set()
    for res_id, res_name, atom_name, *_ in atoms:
        if atom_name not in VALID_ATOM_NAMES:
            raise ValueError(f"atom name {atom_name!r} is outside the vocabulary")
        if res_name not in VALID_RES_NAMES:
            raise ValueError(f"residue name {res_name!r} is not canonical")
        key = (res_id, atom_name)
        if key in seen:
            raise ValueError(f"duplicate atom {key}")
        seen.add(key)

    ordered = sorted(atoms, key=lambda a: (a[0], _ATOM_ORDER[a[2]]))
    array = AtomArray(len(ordered))
    array.add_annotation("b_factor", float)
    array.add_annotation("occupancy", float)
    array.chain_id = np.array([CHAIN_ID] * len(ordered))
    array.res_id = np.array([a[0] for a in ordered], dtype=int)
    array.res_name = np.array([a[1] for a in ordered])
    array.atom_name = np.array([a[2] for a in ordered])
    array.element = np.array([element_of(a[2]) for a in ordered])
    array.hetero = np.zeros(len(ordered), dtype=bool)
    array.coord = np.array([[a[3], a[4], a[5]] for a in ordered], dtype=np.float32).reshape(
        len(ordered), 3
    )
    array.b_factor = np.array([a[6] for a in ordered], dtype=float)
    array.occupancy = np.ones(len(ordered), dtype=float)
    return array


def write_structure(array: AtomArray, path) -> None:
    """Write a canonical :class:`AtomArray` to ``path`` as a PDB file."""
    pdb = PDBFile()
    pdb.set_structure(array)
    pdb.write(str(path))


def read_structure(path) -> AtomArray:
    """Read a canonical PDB file, validating the contract.

    Filters nothing silently: an atom outside the vocabulary, a second chain,
    or a duplicated ``(res_id, atom_name)`` raises rather than being dropped,
    because each of those means the producer and the scorer disagree about
    what was predicted.

    Raises:
        ValueError: the file violates the contract in any of those ways.
    """
    array = PDBFile.read(str(path)).get_structure(
        model=1, extra_fields=["b_factor", "occupancy"]
    )
    chains = set(np.unique(array.chain_id))
    if chains - {CHAIN_ID}:
        raise ValueError(f"{path}: expected the single chain {CHAIN_ID!r}, got {sorted(chains)}")
    bad = sorted(set(np.unique(array.atom_name)) - VALID_ATOM_NAMES)
    if bad:
        raise ValueError(f"{path}: atom names outside the vocabulary: {bad}")
    keys = list(zip(array.res_id.tolist(), array.atom_name.tolist()))
    if len(set(keys)) != len(keys):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"{path}: duplicate (res_id, atom_name) entries: {duplicates[:10]}")
    return array


def atom_keys(array: AtomArray) -> list[tuple[int, str]]:
    """The ``(res_id, atom_name)`` identity of every atom, in file order."""
    return list(zip(array.res_id.tolist(), array.atom_name.tolist()))

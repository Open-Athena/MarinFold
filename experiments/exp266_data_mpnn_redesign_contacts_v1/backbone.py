# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Backbone extraction and sequence relabelling — the core of exp266.

The whole experiment rests on one property of pyconfind, verified in
``probe_pyconfind.py`` and pinned by ``tests/test_backbone.py``:

    confind's contact degree is a *rotamer-ensemble* quantity. It rebuilds
    side chains from the Dunbrack library rather than reading the ones in
    the input file, so a structure stripped to backbone atoms yields
    bit-identical contact degrees to the all-atom original.

That makes ``(backbone, residue-name assignment)`` a complete pyconfind
input, which is what lets us take an AFDB backbone, write a ProteinMPNN
sequence onto it, and compute a contacts-v1 document under *exactly* the
same contact operator as the parent corpus.

Nothing here is contacts-v1-specific; it is gemmi structure surgery.
"""

from __future__ import annotations

import gemmi

# The four mainchain atoms confind needs to place a rotamer. CB is
# deliberately absent: pyconfind's ``do_not_count_cb`` defaults to True and
# the probe confirms dropping CB changes nothing.
BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O"})

# contacts-v1 serializes the 20 canonical residues; anything else becomes
# <UNK> downstream. ProteinMPNN's alphabet is exactly these 20 plus 'X'.
AA1_TO_AA3: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AA3_TO_AA1: dict[str, str] = {v: k for k, v in AA1_TO_AA3.items()}


def prepare_structure(structure: gemmi.Structure) -> gemmi.Structure:
    """Apply the same gemmi cleanup contacts-v1 generation assumes.

    Mutates a *clone*; the caller's structure is untouched.
    """
    st = structure.clone()
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    return st


def strip_to_backbone(structure: gemmi.Structure) -> gemmi.Structure:
    """Delete every non-``BACKBONE_ATOMS`` atom, returning a new structure.

    Deletion walks each residue in reverse so index shifts can't skip an
    atom — the same class of bug as gemmi's chain-removal trap (see the
    root ``AGENTS.md`` note on ``remove_chain`` skipping half the model).
    """
    st = structure.clone()
    for model in st:
        for chain in model:
            for residue in chain:
                for i in range(len(residue) - 1, -1, -1):
                    if residue[i].name not in BACKBONE_ATOMS:
                        del residue[i]
    return st


def residue_sequence(structure: gemmi.Structure) -> str:
    """One-letter sequence in pyconfind's residue order (grouped by chain).

    Non-canonical residues become ``X``; contacts-v1 maps those to ``<UNK>``
    and ProteinMPNN cannot design them, so they are the reason a row can be
    filtered rather than silently mislabelled.
    """
    return "".join(
        AA3_TO_AA1.get(residue.name, "X")
        for chain in structure[0]
        for residue in chain
    )


def relabel_sequence(structure: gemmi.Structure, sequence: str) -> gemmi.Structure:
    """Write ``sequence`` (one-letter) onto ``structure``'s residues in order.

    Raises on a length mismatch or a non-canonical letter rather than
    truncating or substituting: a silently misaligned sequence would produce
    a document whose contacts belong to a different protein, which is
    exactly the corpus-corrupting failure the pipeline must never ship.
    """
    residues = [residue for chain in structure[0] for residue in chain]
    if len(sequence) != len(residues):
        raise ValueError(
            f"sequence length {len(sequence)} != {len(residues)} residues"
        )
    try:
        names = [AA1_TO_AA3[aa] for aa in sequence]
    except KeyError as exc:
        raise ValueError(f"non-canonical residue letter {exc} in sequence") from None

    st = structure.clone()
    for residue, name in zip(
        (r for chain in st[0] for r in chain), names, strict=True
    ):
        residue.name = name
    return st


def backbone_coords(structure: gemmi.Structure) -> tuple[list[str], list[list[list[float]]]]:
    """Extract ``(chain_ids, [[N, CA, C, O], ...])`` in pyconfind residue order.

    This is the hand-off to ProteinMPNN, which wants raw N/CA/C/O arrays. A
    residue missing any mainchain atom raises — ProteinMPNN would otherwise
    silently see a NaN and design against garbage geometry.
    """
    chain_ids: list[str] = []
    coords: list[list[list[float]]] = []
    for chain in structure[0]:
        for residue in chain:
            atoms = {atom.name: atom for atom in residue}
            try:
                frame = [
                    [atoms[name].pos.x, atoms[name].pos.y, atoms[name].pos.z]
                    for name in ("N", "CA", "C", "O")
                ]
            except KeyError as exc:
                raise ValueError(
                    f"residue {chain.name}/{residue.seqid.num} {residue.name} "
                    f"is missing mainchain atom {exc}"
                ) from None
            chain_ids.append(chain.name)
            coords.append(frame)
    return chain_ids, coords

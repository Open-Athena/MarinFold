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


# --- staged backbone encoding -------------------------------------------------
#
# CoreWeave task pods carry only CoreWeave S3 credentials (`iris-task-env`:
# AWS_*/CW_*/FSSPEC_S3, no GCP), so a CoreWeave worker cannot read AFDB's
# requester-pays GCS bucket. The backbones are therefore staged once from GCP
# into a compact parquet that CoreWeave can read from its own object store.
#
# Coordinates are stored as **int32 milli-angstroms**, not floats. AFDB mmCIF
# writes `Cartn_x` with 3 decimals, so `round(x * 1000)` is exact and
# `value / 1000.0` reproduces the very same double gemmi parsed — lossless,
# 4 bytes per number, and integers with spatial locality compress far better
# than float32. Storing float32 instead would perturb coordinates at the ~1e-3 A
# level, which is exactly the scale at which a marginal pyconfind contact can
# flip. `tests/test_backbone.py` asserts the round-trip is byte-identical at the
# document level; `stage_backbones.py` re-checks exactness per structure.

COORD_SCALE = 1000  # milli-angstroms


def encode_backbone(structure: gemmi.Structure) -> dict:
    """Serialize a prepared, backbone-only structure to plain parquet columns.

    Raises if the structure is not the single contiguous protein chain the
    AFDB monomer corpus is supposed to be, or if any coordinate is not exactly
    representable at 0.001 A — both are conditions the caller must see rather
    than silently encode wrong.
    """
    chains = list(structure[0])
    if len(chains) != 1:
        raise ValueError(f"expected 1 chain, got {len(chains)}")
    chain = chains[0]
    residues = list(chain)
    if not residues:
        raise ValueError("chain has no residues")

    resnums = [r.seqid.num for r in residues]
    if resnums != list(range(resnums[0], resnums[0] + len(resnums))):
        raise ValueError(
            f"residue numbers are not contiguous ({resnums[0]}..{resnums[-1]} "
            f"over {len(resnums)} residues)"
        )

    sequence = residue_sequence(structure)
    if "X" in sequence:
        # Non-canonical residue: ProteinMPNN cannot design it and decode has no
        # 3-letter name to write back. A designed-in filter for the caller, but
        # it must not be encodable.
        raise ValueError("structure contains non-canonical residues")

    coords: list[int] = []
    plddt: list[float] = []
    for residue in residues:
        atoms = {atom.name: atom for atom in residue}
        for name in ("N", "CA", "C", "O"):
            try:
                pos = atoms[name].pos
            except KeyError:
                raise ValueError(
                    f"residue {residue.seqid.num} {residue.name} is missing {name}"
                ) from None
            for value in (pos.x, pos.y, pos.z):
                scaled = round(value * COORD_SCALE)
                if scaled / COORD_SCALE != value:
                    raise ValueError(
                        f"coordinate {value!r} is not exact at 1/{COORD_SCALE} A"
                    )
                coords.append(scaled)
        plddt.append(atoms["CA"].b_iso)

    return {
        "chain_id": chain.name,
        "resnum_start": resnums[0],
        "sequence": sequence,
        "coords_milli": coords,   # [L * 4 * 3] int32, residue-major, N/CA/C/O
        "ca_plddt": plddt,        # [L] float32; rebuilt onto CA b_iso
    }


def decode_backbone(row: dict) -> gemmi.Structure:
    """Rebuild the gemmi structure `encode_backbone` serialized.

    The result is what pyconfind sees, so it has to carry everything
    ``analyze_structure`` reads: residue names, chain id, author residue
    numbers, N/CA/C/O positions, and the CA B-factors ``_mean_ca_bfactor``
    averages into ``global_plddt``.
    """
    sequence = row["sequence"]
    coords = row["coords_milli"]
    plddt = row["ca_plddt"]
    if len(coords) != len(sequence) * 12:
        raise ValueError(
            f"coords length {len(coords)} != 12 * {len(sequence)} residues"
        )
    if len(plddt) != len(sequence):
        raise ValueError(f"ca_plddt length {len(plddt)} != {len(sequence)} residues")

    structure = gemmi.Structure()
    structure.name = row.get("entry_id", "backbone")
    model = gemmi.Model("1")
    chain = gemmi.Chain(row["chain_id"])

    for i, letter in enumerate(sequence):
        residue = gemmi.Residue()
        residue.name = AA1_TO_AA3[letter]
        residue.seqid = gemmi.SeqId(row["resnum_start"] + i, " ")
        residue.het_flag = "A"
        for j, name in enumerate(("N", "CA", "C", "O")):
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element(name[0])
            base = i * 12 + j * 3
            atom.pos = gemmi.Position(*(coords[base + k] / COORD_SCALE for k in range(3)))
            # AFDB writes the residue's pLDDT into every atom's B-factor;
            # only CA is read back (_mean_ca_bfactor), but match the source.
            atom.b_iso = plddt[i]
            residue.add_atom(atom)
        chain.add_residue(residue)

    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    return structure


def backbone_coords_from_row(row: dict):
    """Staged ``coords_milli`` -> the ``[L, 4, 3]`` float array ProteinMPNN wants.

    Skips rebuilding a gemmi structure: the design step needs only numbers, and
    on the GPU worker this runs once per backbone in the hot loop. Returns
    float32 because that is what ``tied_featurize`` casts to anyway — the
    lossless int32 representation matters for *pyconfind*, which sees the
    rebuilt structure from :func:`decode_backbone`, not for the design model.
    """
    import numpy as np

    coords = np.asarray(row["coords_milli"], dtype=np.int32)
    expected = len(row["sequence"]) * 12
    if coords.size != expected:
        raise ValueError(f"coords length {coords.size} != {expected}")
    return (coords.reshape(-1, 4, 3) / COORD_SCALE).astype(np.float32)

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Production adapter: structure file/bytes -> bio2token model input tensors.

This is the gemmi-based, cif-native replacement for the biopython/pandas
oracle in ``reference_input.py``. It parses a structure directly with gemmi
(fast, reads mmCIF from bytes, no PDB round-trip) and reproduces bio2token's
``uniform_dataframe`` layout exactly, reusing the canonical atom/residue
conventions vendored under ``bio2token/`` as the source of truth:

    per residue, atoms in canonical order  [N, CA, C, O] + SC_ATOMS[res]
    token_class                            N/C/O -> BB(0), CA -> C_REF(1), SC(2)
    missing atoms                          NaN, then dropped before the model
    all coords                             barycenter-centered over known atoms

Unlike the oracle path (cif -> gemmi.write_pdb -> biopython, which rounds
coordinates to PDB's 3 decimals), this reads full-precision cif coordinates,
so it is the *more* faithful input. It is validated against the oracle by
round-trip RMSD and high token agreement (see tests/test_adapt.py).
"""

import os
from dataclasses import dataclass

import gemmi
import numpy as np
import torch

from bio2token.data.utils.molecule_conventions import (
    AA_ABRV_REVERSED,
    AA_C_REF,
    BB_ATOMS_AA,
    BB_ATOMS_RNA,
    RNA_ABRV_REVERSED,
    RNA_C_REF,
    SC_ATOMS_AA,
    SC_ATOMS_RNA,
)
from bio2token.data.utils.tokens import BB_CLASS, C_REF_CLASS, SC_CLASS

_NAN3 = (float("nan"), float("nan"), float("nan"))


@dataclass(frozen=True)
class Residue:
    """One kept residue: canonical name plus its heavy atoms by name.

    ``atoms`` maps atom name -> (x, y, z), first occurrence only (matching
    bio2token's single-match-or-missing convention), hydrogens excluded.
    """

    name: str          # 3-letter (aa) or 1-2 char (rna) canonical residue name
    res_type: str      # "aa" or "rna"
    atoms: dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class ParsedStructure:
    entry_id: str
    residues: list[Residue]

    @property
    def sequence(self) -> str:
        rev = {"aa": AA_ABRV_REVERSED, "rna": RNA_ABRV_REVERSED}
        return "".join(rev[r.res_type][r.name] for r in self.residues)


def parse_structure(source, *, entry_id: str | None = None) -> ParsedStructure:
    """Parse a structure into a :class:`ParsedStructure`.

    ``source`` is a filesystem path or raw cif/pdb bytes/str (the production
    path fetches cif bytes from GCS). Residue filtering mirrors bio2token's
    ``pdb_2_dict``: keep 3-char residues in the amino-acid table and shorter
    names in the RNA table; skip everything else (ligands, water, modified
    residues like MSE). Hydrogens (atom name starting with 'H') are dropped.
    """
    if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
        st = gemmi.read_structure(str(source))
        entry_id = entry_id or st.name or os.path.basename(str(source)).split(".")[0]
    else:
        data = source.decode("utf-8", "replace") if isinstance(source, (bytes, bytearray)) else source
        st = gemmi.read_structure_string(data)
        entry_id = entry_id or st.name or "structure"
    st.setup_entities()
    if len(st) == 0:
        raise ValueError(f"{entry_id}: no models in structure")

    residues: list[Residue] = []
    for chain in st[0]:
        for res in chain:
            resname = res.name.strip()
            if len(resname) == 3:
                rev, res_type = AA_ABRV_REVERSED, "aa"
            else:
                rev, res_type = RNA_ABRV_REVERSED, "rna"
            if resname not in rev:
                continue
            atoms: dict[str, tuple[float, float, float]] = {}
            for atom in res:
                name = atom.name.strip()
                if name[:1] == "H":  # bio2token drops atoms whose name starts with 'H'
                    continue
                if name not in atoms:  # keep first occurrence (ignore altloc dups)
                    atoms[name] = (atom.pos.x, atom.pos.y, atom.pos.z)
            residues.append(Residue(name=resname, res_type=res_type, atoms=atoms))
    if not residues:
        raise ValueError(f"{entry_id}: no standard residues parsed")
    return ParsedStructure(entry_id=entry_id, residues=residues)


def _canonical_layout(res: Residue):
    """Return (atom_names, token_classes) for one residue in canonical order."""
    if res.res_type == "aa":
        bb, sc, c_ref = BB_ATOMS_AA, SC_ATOMS_AA[res.name], AA_C_REF
    else:
        bb, sc, c_ref = BB_ATOMS_RNA, SC_ATOMS_RNA[res.name], RNA_C_REF
    names = list(bb) + list(sc)
    classes = [C_REF_CLASS if c_ref[i] else BB_CLASS for i in range(len(bb))]
    classes += [SC_CLASS] * len(sc)
    return names, classes


def to_bio2token_batch(parsed: ParsedStructure, *, add_batch_dim: bool = True) -> dict:
    """Build the model-ready input from a :class:`ParsedStructure`.

    Reproduces ``uniform_dataframe`` + ``compute_masks`` + the ``test_pdb.py``
    drop-unknown step: atoms are laid out in canonical per-residue order,
    missing atoms are NaN, all known coords are centered on the structure
    barycenter, and NaN (missing) atoms are dropped. Returns the kept-atom
    tensors plus per-atom provenance (residue ordinal + atom name).
    """
    coords: list[tuple[float, float, float]] = []
    token_class: list[int] = []
    residue_index: list[int] = []
    atom_name: list[str] = []
    for r_idx, res in enumerate(parsed.residues):
        names, classes = _canonical_layout(res)
        for name, cls in zip(names, classes, strict=True):
            coords.append(res.atoms.get(name, _NAN3))
            token_class.append(cls)
            residue_index.append(r_idx)
            atom_name.append(name)

    structure = np.asarray(coords, dtype=np.float64)          # (N_all, 3)
    known = ~np.isnan(structure).any(axis=1)                  # (N_all,)
    if not known.any():
        # Every heavy atom missing -> centering on an empty barycenter is NaN and
        # would emit a degenerate 0-atom document. Fail loud instead.
        raise ValueError(f"{parsed.entry_id}: no known heavy atoms to tokenize")
    structure[known] -= structure[known].mean(axis=0)         # center on known barycenter

    batch = {
        "structure": torch.tensor(structure[known], dtype=torch.float32),
        "token_class": torch.tensor(np.asarray(token_class)[known], dtype=torch.long),
        "residue_index": torch.tensor(np.asarray(residue_index)[known], dtype=torch.long),
        "atom_name": [a for a, k in zip(atom_name, known, strict=True) if k],
    }
    if add_batch_dim:
        for key in ("structure", "token_class", "residue_index"):
            batch[key] = batch[key][None]
    return batch

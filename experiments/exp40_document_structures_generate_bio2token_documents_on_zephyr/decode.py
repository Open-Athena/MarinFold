# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode a ``bio2token-v2`` document back to a structure — the inverse of
``generate.py``.

    document -> parse triples -> reconstruct the encoder's canonical atom order
    -> FSQ dequantize + decoder -> per-atom coordinates.

Because the document shuffles its atom triples, decoding must first put the
atoms back into the order the encoder saw them: per residue (in sequence order),
the canonical layout ``[N, CA, C, O] + sidechain(res)``, keeping only the atoms
the document actually carries. Each triple names its ``(residue position, atom
name)``, so that reordering is exact.

Used to validate that the format round-trips (``tests/test_roundtrip.py``) and
as the decode step for downstream evaluation (model rollout -> structure ->
lDDT / CA-RMSD / TM-score).
"""

import torch

from bio2token.data.utils.molecule_conventions import BB_ATOMS_AA, SC_ATOMS_AA


def _inside(token: str) -> str:
    """``<CA>`` -> ``CA``."""
    return token[1:-1]


def parse_document(document: str):
    """Parse a document into ``(sequence, atoms)``.

    ``sequence`` is ``[(position, res_name), ...]`` in chain order; ``atoms`` is
    ``[(position, atom_name, code), ...]`` in the document's (shuffled) order.
    """
    toks = document.split(" ")
    i_seq = toks.index("<begin_sequence>")
    i_stmt = toks.index("<begin_statements>")
    i_end = toks.index("<end>")

    seq_toks = toks[i_seq + 1:i_stmt]
    sequence = [
        (int(_inside(seq_toks[k])[1:]), _inside(seq_toks[k + 1]))
        for k in range(0, len(seq_toks), 2)
    ]

    atom_toks = toks[i_stmt + 1:i_end]
    atoms = [
        (int(_inside(atom_toks[k])[1:]),          # <p5>   -> 5
         _inside(atom_toks[k + 1]),               # <CA>   -> CA
         int(_inside(atom_toks[k + 2])[2:]))      # <bt508>-> 508
        for k in range(0, len(atom_toks), 3)
    ]
    return sequence, atoms


def canonical_indices(sequence, atoms):
    """Reconstruct the encoder's canonical atom order.

    Returns ``(indices, order)``: the bio2token codes reordered to match how the
    encoder saw the atoms, and the aligned ``[(position, atom_name), ...]`` (so a
    caller can pick out, e.g., the CA atoms after decoding).
    """
    code_at = {(pos, atom): code for pos, atom, code in atoms}
    present: dict[int, set[str]] = {}
    for pos, atom, _ in atoms:
        present.setdefault(pos, set()).add(atom)

    indices, order = [], []
    for pos, res in sequence:
        canonical = list(BB_ATOMS_AA) + list(SC_ATOMS_AA.get(res, ()))
        for atom in canonical:
            if atom in present.get(pos, ()):
                indices.append(code_at[(pos, atom)])
                order.append((pos, atom))
    return indices, order


@torch.no_grad()
def decode_document(document: str, model):
    """Decode a document to ``(coords, order)``.

    ``coords`` is an ``(N, 3)`` tensor of per-atom coordinates (bio2token's
    barycenter-centered frame); ``order`` is the aligned ``[(position, atom
    name), ...]``. Coordinates come back on CPU.
    """
    sequence, atoms = parse_document(document)
    indices, order = canonical_indices(sequence, atoms)
    device = next(model.parameters()).device
    idx = torch.tensor([indices], dtype=torch.long, device=device)
    coords = model.decode_indices(idx)[0].to("cpu")
    return coords, order

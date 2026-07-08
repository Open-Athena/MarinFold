# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference bio2token model input, via bio2token's own data pipeline.

This is the *oracle* path: it builds the ``structure`` / ``token_class`` /
mask tensors exactly the way upstream ``scripts/test_pdb.py`` does — through
vendored ``pdb_2_dict`` + ``uniform_dataframe`` + ``compute_masks``. It is
what the round-trip test tokenizes and what our own (gemmi-based) production
adapter in a later phase must match.

bio2token's ``pdb_2_dict`` is biopython/PDB based, so a cif input is first
converted to PDB with gemmi.
"""

import tempfile

import gemmi
import torch

from bio2token.data.utils.utils import compute_masks, pdb_2_dict, uniform_dataframe


def build_reference_batch(structure_path: str) -> dict:
    """Return a single-example batch (leading batch dim) ready for the model.

    Missing-coordinate atoms (``unknown_structure``) are dropped before the
    model, matching upstream ``test_pdb.py``.
    """
    if structure_path.endswith((".cif", ".mmcif")):
        st = gemmi.read_structure(structure_path)
        st.setup_entities()
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            structure_path = f.name
        st.write_pdb(structure_path)

    d = pdb_2_dict(structure_path, chains=None)
    structure, unknown, _res_name, _res_ids, token_class, _atom_names = uniform_dataframe(
        d["seq"], d["res_types"], d["coords_groundtruth"], d["atom_names"],
        d["res_atom_start"], d["res_atom_end"],
    )
    batch = {
        "structure": torch.tensor(structure).float(),
        "token_class": torch.tensor(token_class).long(),
        "unknown_structure": torch.tensor(unknown).bool(),
    }
    batch = {k: v[~batch["unknown_structure"]] for k, v in batch.items()}
    batch = compute_masks(batch, structure_track=True)
    return {k: (v[None] if torch.is_tensor(v) else v) for k, v in batch.items()}


def kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor) -> float:
    """RMSD between two ``(N, 3)`` point sets after optimal rigid alignment."""
    P = P - P.mean(0, keepdim=True)
    Q = Q - Q.mean(0, keepdim=True)
    U, _, Vt = torch.linalg.svd(P.T @ Q)
    d = torch.sign(torch.det(Vt.T @ U.T))
    R = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=P.dtype)) @ U.T
    return torch.sqrt(((P @ R.T - Q) ** 2).sum(-1).mean()).item()

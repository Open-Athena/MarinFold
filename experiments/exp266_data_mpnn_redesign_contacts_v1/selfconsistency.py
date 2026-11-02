# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-consistency between a designed sequence's refold and its backbone.

The one assumption exp266's design does not otherwise test: a ProteinMPNN
sequence is written onto an AFDB backbone and the contacts are computed there,
but nothing checks that the sequence *would actually fold* to that backbone.

Because a design has the same length and residue order as its parent backbone,
the residue correspondence is the identity map — no structural alignment search
is needed, and both metrics reduce to a Kabsch superposition:

* **scRMSD** — CA RMSD after optimal superposition. The field's designability
  gate is < 2 Å.
* **scTM** — TM-score under that same correspondence, length-normalised so it
  is comparable across proteins. > 0.5 means "same fold".

Numpy only; no TM-align binary, no alignment heuristics.
"""

from __future__ import annotations

import numpy as np


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    """RMSD of ``p`` onto ``q`` after optimal rotation, plus the superposed ``p``."""
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {q.shape}")
    pc, qc = p - p.mean(0), q - q.mean(0)
    # Kabsch: SVD of the covariance, with a reflection guard so we get a proper
    # rotation rather than a mirror image (which would flatten the RMSD of a
    # mirrored fold and silently call it a match).
    v, _s, wt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(v @ wt))
    rot = v @ np.diag([1.0, 1.0, d]) @ wt
    aligned = pc @ rot
    return float(np.sqrt(((aligned - qc) ** 2).sum(1).mean())), aligned + q.mean(0)


def tm_score(p: np.ndarray, q: np.ndarray) -> float:
    """TM-score of ``p`` vs ``q`` under the identity correspondence.

    Uses the standard d0 normalisation. This is the superposition-fixed
    TM-score, not TM-align's optimised one: with a 1:1 correspondence and a
    Kabsch fit it is the right quantity and is a slight *under*-estimate of
    what TM-align would report, so it will not overstate agreement.
    """
    n = len(p)
    if n != len(q):
        raise ValueError("length mismatch")
    _rmsd, aligned = kabsch_rmsd(p, q)
    d0 = 1.24 * (max(n, 19) - 15) ** (1 / 3) - 1.8
    d = np.sqrt(((aligned - q) ** 2).sum(1))
    return float((1.0 / (1.0 + (d / d0) ** 2)).mean())


def ca_coords_from_staged(row: dict) -> np.ndarray:
    """CA coordinates of a staged backbone row, in ångströms."""
    from backbone import COORD_SCALE

    coords = np.asarray(row["coords_milli"], dtype=np.float64).reshape(-1, 4, 3)
    return coords[:, 1, :] / COORD_SCALE          # index 1 == CA in N/CA/C/O


def ca_coords_from_structure(structure) -> np.ndarray:
    """CA coordinates of a gemmi structure's first chain."""
    out = []
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                if atom.name == "CA":
                    out.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    break
    return np.asarray(out, dtype=np.float64)

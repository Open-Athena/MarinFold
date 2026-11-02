# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Superposition-based and superposition-free metrics for one prediction.

The array half of the scoring harness: everything that can be computed from
two :class:`~biotite.structure.AtomArray` objects in the ``canonical_pdb``
contract. TM-score needs a file pair and lives in ``usalign.py``;
``score_structures.py`` runs both and joins the results.

Atom identity
-------------
An atom is the pair ``(res_id, atom_name)`` — the 1-based input-sequence
residue index and a name from the 37-atom heavy-atom vocabulary. The
**covered set** is the intersection of the ground truth's atoms and the
prediction's. Predicted atoms outside the ground truth (a residue the
structure never resolved, or an atom name the residue does not have) are
counted as ``n_pred_extra`` and otherwise ignored: they have nothing to be
scored against.

The partial-prediction convention
---------------------------------
contacts-and-crops-v1 documents are budget-filling and ~96% are truncated, so
partial predictions are the *normal* case, not an edge case. Two families of
metric handle it differently, and mixing them up is the easiest way to
misread this eval:

* **RMSD is covered-only.** A superposition needs a common atom set, so
  ``rmsd_ca`` / ``rmsd_all`` are computed over covered atoms and are blind to
  everything the predictor declined to place. A model that emits three atoms
  perfectly scores 0.0 Å. **Never read an RMSD from this harness without the
  coverage column next to it.**
* **lDDT and TM-score are coverage-penalized.** Their denominators come from
  the *ground truth*: every reference contact (lDDT) and every reference
  residue (TM-score) counts, whether or not the predictor placed it. Missing
  atoms therefore push both scores down, exactly as a missing loop does in
  CASP. These are the numbers to compare models on.

For lDDT the harness reports both readings — ``lddt_all`` (penalized, the
headline) and ``lddt_all_covered`` (restricted to covered atoms, i.e. "how
good is what it did emit") — because the gap between them *is* the coverage
story.

Implementation note on the penalized lDDT: :func:`biotite.structure.lddt`
takes reference contacts from the reference structure and needs a subject
coordinate for every reference atom. We give unpredicted atoms a coordinate
far from everything *and from each other* (:data:`_SENTINEL_SPACING`), so
every reference contact touching an unpredicted atom is scored as broken
while still counting in the denominator. That is precisely the penalized
definition, computed by the library's own tested code path rather than by a
re-derivation of it here.
"""

import numpy as np
from biotite.structure import AtomArray, lddt, rmsd, superimpose

from canonical_pdb import atom_keys

# lDDT convention (Mariani et al. 2013), the same one exp12/exp20/exp26 pinned
# for their distogram lDDT and the one CASP and AlphaFold report.
LDDT_INCLUSION_RADIUS_A: float = 15.0
LDDT_DISTANCE_BINS_A: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

# Unpredicted atoms are parked at ``(k * _SENTINEL_SPACING, 0, 0)``. The
# spacing must exceed the inclusion radius plus the widest distance bin by a
# wide margin so that *every* pair involving a sentinel — including
# sentinel-to-sentinel pairs, whose reference distance can be short — lands
# outside the 4 Å bin and scores 0.
_SENTINEL_SPACING = 1.0e4

# Minimum atoms for a meaningful least-squares superposition.
_MIN_SUPERPOSITION_ATOMS = 3


def _sentinel_coords(n: int) -> np.ndarray:
    """``n`` mutually distant coordinates, far from any real structure."""
    out = np.zeros((n, 3), dtype=np.float64)
    out[:, 0] = (np.arange(n, dtype=np.float64) + 1.0) * _SENTINEL_SPACING
    return out


def _kabsch_rmsd(reference: AtomArray, subject_coord: np.ndarray) -> float:
    """Least-squares-superposed RMSD between a reference and matching coords.

    Returns ``nan`` when there are too few atoms to define a superposition.
    """
    if len(reference) < _MIN_SUPERPOSITION_ATOMS:
        return float("nan")
    fitted = reference.copy()
    fitted.coord = subject_coord.astype(np.float32)
    fitted, _ = superimpose(reference, fitted)
    return float(rmsd(reference, fitted))


def _lddt(reference: AtomArray, subject_coord: np.ndarray, *, atom_mask=None) -> float:
    """lDDT of ``subject_coord`` against ``reference``, CASP convention."""
    if len(reference) < 2:
        return float("nan")
    value = lddt(
        reference,
        subject_coord,
        inclusion_radius=LDDT_INCLUSION_RADIUS_A,
        distance_bins=LDDT_DISTANCE_BINS_A,
        exclude_same_residue=True,
        atom_mask=atom_mask,
    )
    return float(value)


def score_prediction(
    gt: AtomArray,
    pred: AtomArray,
    *,
    refined_max_sigma: float = 1.0,
) -> dict[str, float]:
    """Score one prediction against its ground truth.

    Args:
        gt: ground-truth structure in the ``canonical_pdb`` contract.
        pred: predicted structure, same contract. May cover any subset of the
            ground truth's atoms, in any frame.
        refined_max_sigma: atoms whose B-factor (predicted positional sigma in
            Å) is at or below this are counted as *refined*; the rest are
            coarse. For contacts-and-crops-v1 this separates Pass-2 crop atoms
            (0.1 Å tenths) from Pass-1 box-only atoms (a 10 Å cell).

    Returns:
        A flat metric dict. Coverage first, then the coverage-penalized
        metrics (``lddt_*``), then the covered-only ones (``lddt_*_covered``,
        ``rmsd_*``). TM-score is added by ``score_structures.py``.
    """
    gt_keys = atom_keys(gt)
    pred_keys = atom_keys(pred)
    pred_row = {key: i for i, key in enumerate(pred_keys)}

    covered = np.array([key in pred_row for key in gt_keys], dtype=bool)
    pred_rows = np.array(
        [pred_row[key] for key in gt_keys if key in pred_row], dtype=int
    )

    is_ca = gt.atom_name == "CA"
    n_gt = len(gt_keys)
    n_covered = int(covered.sum())

    # Subject coordinates for every ground-truth atom: the predicted position
    # where there is one, a mutually-distant sentinel where there is not.
    subject = _sentinel_coords(n_gt)
    if n_covered:
        subject[covered] = pred.coord[pred_rows].astype(np.float64)

    gt_residues = set(gt.res_id.tolist())
    covered_residues = {rid for rid, keep in zip(gt.res_id.tolist(), covered) if keep}

    sigma = pred.b_factor[pred_rows] if n_covered else np.empty(0)
    n_refined = int((sigma <= refined_max_sigma).sum())
    refined_mask = np.zeros(n_gt, dtype=bool)
    if n_covered:
        refined_mask[np.flatnonzero(covered)[sigma <= refined_max_sigma]] = True

    metrics: dict[str, float] = {
        "n_gt_atoms": float(n_gt),
        "n_gt_ca": float(is_ca.sum()),
        "n_gt_residues": float(len(gt_residues)),
        "n_pred_atoms": float(len(pred_keys)),
        "n_pred_extra": float(len(pred_keys) - n_covered),
        "n_covered_atoms": float(n_covered),
        "n_covered_ca": float((covered & is_ca).sum()),
        "atom_coverage": n_covered / n_gt if n_gt else float("nan"),
        "ca_coverage": (
            float((covered & is_ca).sum()) / float(is_ca.sum())
            if is_ca.any()
            else float("nan")
        ),
        "residue_coverage": (
            len(covered_residues) / len(gt_residues) if gt_residues else float("nan")
        ),
        "frac_refined_of_covered": n_refined / n_covered if n_covered else float("nan"),
        "frac_refined_of_gt": n_refined / n_gt if n_gt else float("nan"),
    }

    # --- Coverage-penalized (denominator from the ground truth) ---
    metrics["lddt_all"] = _lddt(gt, subject)
    metrics["lddt_ca"] = (
        _lddt(gt[is_ca], subject[is_ca]) if is_ca.sum() >= 2 else float("nan")
    )
    # lDDT of the refined atoms only, scored against *all* ground-truth
    # partners: "when the model claims 0.1 Å precision, is it right?"
    metrics["lddt_all_refined"] = (
        _lddt(gt, subject, atom_mask=refined_mask) if refined_mask.any() else float("nan")
    )

    # --- Covered-only (denominator restricted to what was predicted) ---
    gt_covered = gt[covered]
    subject_covered = subject[covered]
    metrics["lddt_all_covered"] = (
        _lddt(gt_covered, subject_covered) if n_covered >= 2 else float("nan")
    )
    covered_ca = gt_covered.atom_name == "CA"
    metrics["lddt_ca_covered"] = (
        _lddt(gt_covered[covered_ca], subject_covered[covered_ca])
        if covered_ca.sum() >= 2
        else float("nan")
    )
    metrics["rmsd_all"] = _kabsch_rmsd(gt_covered, subject_covered)
    metrics["rmsd_ca"] = _kabsch_rmsd(gt_covered[covered_ca], subject_covered[covered_ca])
    return metrics

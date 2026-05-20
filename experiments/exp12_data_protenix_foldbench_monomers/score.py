# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the best Protenix sample per (protein, mode) against GT.

Five per-protein metrics (one row per (protein, mode) in
``scores.csv``):

- **mae_distogram_cb_angstrom** — mean absolute error between
  Protenix's distogram-derived expected distance
  (``Σ p_bin · midpoint`` over the 64 bins) and the GT **CB-CB
  distance** (CA-CA for glycine, per the AF3 / Protenix
  representative-atom convention), averaged over residue pairs where
  both representative atoms are resolved in the GT. *This is the
  apples-to-apples comparison for the distogram head's output.*

- **mae_structure_ca_angstrom** — same shape as the above, but the
  predicted distances come from the predicted **structure**'s CA-CA
  pairwise distance matrix (i.e. the mmCIF output, not the distogram).
  GT is also CA-CA. Lets us compare how well the *structure* itself
  reproduces the GT distance map.

- **drmsd_ca_angstrom** — RMSE of the same (pred_CA-CA - GT_CA-CA)
  pairwise distance differences (i.e. dRMSD over CA, no sequence-
  separation filter).

- **rmsd_ca_angstrom** — superposition-based RMSD over CA atoms, after
  Kabsch alignment (via ``gemmi.superpose_positions``). Uses the
  residues where both pred and GT have a resolved CA.

- **rmsd_all_heavy_angstrom** — superposition-based RMSD over all
  heavy-atom matches between pred and GT, keyed by ``(label_seq_id,
  atom_name)``. Hydrogens are excluded (Protenix outputs heavy atoms
  only by default).

Why distogram MAE uses CB but dRMSD/RMSD use CA: Protenix's distogram
head represents each token by its CB (CA for glycine) — see
``add_distogram_rep_atom_mask`` in bytedance/Protenix
protenix/data/core/parser.py. Comparing the distogram-derived
expected distance against a GT CA-CA distance would be apples-to-
oranges. The structure-derived metrics (dRMSD + the two RMSDs + the
structure-derived MAE) all use CA, since CA-CA is the canonical
backbone-geometry comparison.

Residue index alignment: Protenix's output sequence is length-N (same
as input), 1-indexed. The GT CIF's residues are mapped to the
canonical sequence via gemmi's ``label_seq_id`` (which matches the
canonical 1..N order). Unresolved residues — those in
``_entity_poly_seq`` but with no corresponding ``_atom_site`` row for
the relevant atom — get masked out per-metric.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np


# CSV schema shared by the top-1 (``score``) and all-samples
# (``score-all-samples``) entry points. In the top-1 CSV every row has
# selected_as_best=1; in the all-samples CSV exactly 200 of 8000 rows
# (one per (mode, stem)) get selected_as_best=1.
#
# Distogram-derived metric columns (per-seed redundancy: the 8 samples
# within a seed share the same distogram → identical values):
#   * `mae_distogram_cb_angstrom` / `drmsd_distogram_cb_angstrom` —
#     filtered to GT pairs in the distogram's expressible range
#     (2.31 ≤ gt ≤ 21.84 Å for Protenix v2). This removes the clipping
#     bias of the unfiltered version.
#   * `mae_distogram_cb_contact_angstrom` / `drmsd_distogram_cb_contact_angstrom`
#     — same formula but on the contact-regime pair set (GT ≤ 8 Å,
#     CASP convention).
#   * `prec_{short,medium,long}_L{,_2,_5}` — CASP contact precision @
#     top-L / top-L/2 / top-L/5 per separation class. Score: contact
#     probability summed over distogram bins with center ≤ 8 Å.
_FIELDS = [
    "pdb_id", "chain_id", "mode", "seed", "sample_idx", "ranking_score",
    "selected_as_best",
    "n_residues",
    # Distogram MAE/dRMSD on the in-range pair set (option B).
    "mae_distogram_cb_angstrom", "drmsd_distogram_cb_angstrom",
    "n_mae_distogram_pairs",
    # Distogram MAE/dRMSD on the contact-regime pair set (option C1).
    "mae_distogram_cb_contact_angstrom", "drmsd_distogram_cb_contact_angstrom",
    "n_mae_distogram_contact_pairs",
    # Structure-derived (CA-based, unaffected by distogram range).
    "mae_structure_ca_angstrom",
    "drmsd_ca_angstrom", "n_ca_pairs",
    "rmsd_ca_angstrom", "n_ca_atoms",
    "rmsd_all_heavy_angstrom", "n_heavy_atoms",
    # CASP contact precision (option C2). One column per (range, k).
    "prec_short_L", "prec_short_L_2", "prec_short_L_5",
    "prec_medium_L", "prec_medium_L_2", "prec_medium_L_5",
    "prec_long_L", "prec_long_L_2", "prec_long_L_5",
    "n_short_contacts", "n_medium_contacts", "n_long_contacts",
]


# Distogram bin centers — must match Protenix v2's distogram head config.
# From bytedance/Protenix configs/configs_base.py's loss.distogram block:
#   min_bin = 2.3125, max_bin = 21.6875, no_bins = 64
# And bytedance/Protenix protenix/model/sample_confidence.py:get_bin_centers:
#   bin_width = (max_bin - min_bin) / no_bins
#   boundaries = linspace(min_bin, max_bin - bin_width, no_bins)
#   centers    = boundaries + 0.5 * bin_width
# Out-of-range distances (>21.84 Å) get implicit-clipped to the last
# bin's center on the model side (the expected distance can never
# exceed centers[-1] ≈ 21.54 Å), which is why the distance-based
# metrics (MAE / dRMSD) filter pairs to GT in [min, max].
_DISTOGRAM_MIN_A = 2.3125
_DISTOGRAM_MAX_A = 21.6875
_DISTOGRAM_N_BINS = 64
_DISTOGRAM_BIN_WIDTH = (_DISTOGRAM_MAX_A - _DISTOGRAM_MIN_A) / _DISTOGRAM_N_BINS
_DISTOGRAM_BIN_MIDPOINTS = np.linspace(
    _DISTOGRAM_MIN_A,
    _DISTOGRAM_MAX_A - _DISTOGRAM_BIN_WIDTH,
    _DISTOGRAM_N_BINS,
    dtype=np.float64,
) + 0.5 * _DISTOGRAM_BIN_WIDTH


# CASP / contact-evaluation constants.
# Contact = CB-CB ≤ 8 Å (CA for GLY) — standard since CASP9, and the
# definition Protenix's distogram is trained against (see the v2 config).
_CONTACT_CUTOFF_A = 8.0

# Mask of distogram bins whose centers fall under the contact cutoff —
# used to derive the per-pair contact probability from the per-pair
# 64-bin distribution. Bins 0..18 for Protenix v2 (centers 2.46-7.91 Å).
_CONTACT_BIN_MASK = _DISTOGRAM_BIN_MIDPOINTS <= _CONTACT_CUTOFF_A

# CASP14-onwards sequence-separation classes. `(sep_lo, sep_hi)` with
# inclusive bounds; sep_hi=None means no upper limit. Pairs with
# |i - j| ≤ 5 are intentionally excluded (too easy).
_CASP_SEPARATIONS: dict[str, tuple[int, int | None]] = {
    "short": (6, 11),
    "medium": (12, 23),
    "long": (24, None),
}

# Precision @ top L/k convention. CASP14 reports L, L/2, L/5 (and
# sometimes L/10). We track the three headline ones.
_CASP_TOP_K: tuple[int, ...] = (1, 2, 5)


@dataclass(frozen=True)
class ProteinCoords:
    """CA + CB-or-CA-for-GLY (distogram representative) coords, 1-indexed.

    Both ``ca`` and ``rep`` lists have length ``n_residues``. An entry
    is None for residues with no resolved atom of that kind in the
    structure. ``ca[i]`` is the CA at canonical residue position
    ``i+1``; ``rep[i]`` is the AF3-style distogram representative atom
    (CB for non-GLY std residues, CA for GLY, None otherwise).
    """

    n_residues: int
    ca: list[tuple[float, float, float] | None]
    rep: list[tuple[float, float, float] | None]


def _read_protein_coords_from_cif(cif_path: Path) -> ProteinCoords:
    """Read CA + distogram-representative coords for the polypeptide chain.

    Uses gemmi's ``label_seq_id`` to align ``atom_site`` rows back to
    the canonical 1..N residue numbering. Unresolved residues — those
    in ``_entity_poly_seq`` but missing from ``_atom_site`` for the
    relevant atom — appear as None in the returned lists.

    The "rep" atom matches Protenix's
    ``add_distogram_rep_atom_mask``: CB for std L-peptide residues
    except glycine, CA for glycine. Non-std residues get None for rep.
    """
    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()

    peptide_entities = [
        e for e in structure.entities
        if e.entity_type == gemmi.EntityType.Polymer
        and e.polymer_type == gemmi.PolymerType.PeptideL
    ]
    if not peptide_entities:
        raise ValueError(f"No polypeptide(L) entity in {cif_path}")
    if len(peptide_entities) > 1:
        raise ValueError(
            f"{cif_path} has {len(peptide_entities)} polypeptide(L) entities; "
            f"score.py only handles monomers."
        )
    entity = peptide_entities[0]
    subchains = set(entity.subchains)
    n_residues = len(entity.full_sequence)

    ca_by_seq: dict[int, tuple[float, float, float]] = {}
    rep_by_seq: dict[int, tuple[float, float, float]] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.subchain not in subchains:
                    continue
                label_seq = residue.label_seq
                if label_seq is None:
                    continue
                ca = residue.find_atom("CA", "\0")
                if ca is not None:
                    ca_by_seq[label_seq] = (ca.pos.x, ca.pos.y, ca.pos.z)
                # Representative atom for the distogram: CB for non-GLY
                # std residues, CA for GLY. We trust the residue name.
                res_name = residue.name.upper()
                if res_name == "GLY":
                    if ca is not None:
                        rep_by_seq[label_seq] = (ca.pos.x, ca.pos.y, ca.pos.z)
                else:
                    cb = residue.find_atom("CB", "\0")
                    if cb is not None:
                        rep_by_seq[label_seq] = (cb.pos.x, cb.pos.y, cb.pos.z)
        break  # only first model

    return ProteinCoords(
        n_residues=n_residues,
        ca=[ca_by_seq.get(i) for i in range(1, n_residues + 1)],
        rep=[rep_by_seq.get(i) for i in range(1, n_residues + 1)],
    )


def _read_heavy_atom_positions(cif_path: Path) -> dict[tuple[int, str], tuple[float, float, float]]:
    """Read every heavy atom keyed by ``(label_seq_id, atom_name)``.

    Used for all-heavy-atom RMSD: we intersect the keys between pred
    and GT, then superpose on the matched subset. Hydrogens are
    excluded (Protenix outputs heavy-only by default; GT may have H
    in some entries and not others — easier to ignore them uniformly).
    """
    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    peptide_entities = [
        e for e in structure.entities
        if e.entity_type == gemmi.EntityType.Polymer
        and e.polymer_type == gemmi.PolymerType.PeptideL
    ]
    if not peptide_entities:
        raise ValueError(f"No polypeptide(L) entity in {cif_path}")
    subchains = set(peptide_entities[0].subchains)
    out: dict[tuple[int, str], tuple[float, float, float]] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.subchain not in subchains:
                    continue
                label_seq = residue.label_seq
                if label_seq is None:
                    continue
                for atom in residue:
                    # Skip hydrogens (H / D) — element check beats atom-name
                    # heuristics like startswith("H") since e.g. "HG" might
                    # be Mercury in a non-standard residue.
                    element = atom.element.name if atom.element else ""
                    if element in ("H", "D"):
                        continue
                    key = (int(label_seq), atom.name)
                    out[key] = (atom.pos.x, atom.pos.y, atom.pos.z)
        break  # first model only
    return out


def _kabsch_rmsd(
    pred_xyz: list[tuple[float, float, float]],
    gt_xyz: list[tuple[float, float, float]],
) -> float:
    """Superposition-based RMSD via gemmi (Kabsch). Returns NaN if <3 points."""
    if len(pred_xyz) < 3:
        return float("nan")
    pred_pos = [gemmi.Position(*p) for p in pred_xyz]
    gt_pos = [gemmi.Position(*p) for p in gt_xyz]
    res = gemmi.superpose_positions(pred_pos, gt_pos)
    return float(res.rmsd)


def _pairwise_distance_matrix(
    positions: list[tuple[float, float, float] | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(distances, mask)`` from a 1-indexed list of (optional) positions.

    ``distances[i, j]`` is the Euclidean distance between positions
    ``i+1`` and ``j+1`` in Å (NaN if either is None). ``mask[i, j]`` is
    True iff both positions are present.
    """
    n = len(positions)
    arr = np.full((n, 3), np.nan, dtype=np.float64)
    for i, p in enumerate(positions):
        if p is not None:
            arr[i] = p
    diffs = arr[:, None, :] - arr[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    mask = np.isfinite(distances)
    return distances, mask


@dataclass(frozen=True)
class _DistogramDerived:
    """Outputs we derive from a single distogram .npz."""

    expected_distances: np.ndarray   # [N, N] — Σ p_bin · center_bin
    contact_probabilities: np.ndarray  # [N, N] — Σ p_bin over bins with center ≤ 8 Å


def _load_distogram_derivatives(distogram_path: Path) -> _DistogramDerived:
    """Load distogram (.npz with key 'probs' shape [N, N, 64]).

    Returns both the expected-distance matrix and the contact-probability
    matrix (= mass on bins with center ≤ 8 Å). Both are needed for the
    full per-protein scoring suite; computing them in one pass avoids
    reloading the .npz.

    If the .npz key is 'logits' instead, softmax along the last axis
    first. The Modal-side hook saves probabilities directly (after
    softmax) to avoid this branch, but we handle both for forward-compat.
    """
    data = np.load(distogram_path)
    if "probs" in data:
        probs = data["probs"].astype(np.float64)
    elif "logits" in data:
        logits = data["logits"].astype(np.float64)
        m = logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits - m)
        probs = exp / exp.sum(axis=-1, keepdims=True)
    else:
        raise KeyError(f"{distogram_path}: expected 'probs' or 'logits', got {list(data.keys())}")
    if probs.shape[-1] != _DISTOGRAM_N_BINS:
        raise ValueError(
            f"{distogram_path}: expected last dim = {_DISTOGRAM_N_BINS}, got {probs.shape[-1]}"
        )
    expected = (probs * _DISTOGRAM_BIN_MIDPOINTS).sum(axis=-1)
    contact = probs[..., _CONTACT_BIN_MASK].sum(axis=-1)
    return _DistogramDerived(expected_distances=expected, contact_probabilities=contact)


def _expected_distances_from_distogram(distogram_path: Path) -> np.ndarray:
    """Back-compat wrapper around :func:`_load_distogram_derivatives` for callers
    that only need the expected-distance matrix.
    """
    return _load_distogram_derivatives(distogram_path).expected_distances


def _casp_contact_precisions(
    *,
    contact_probs: np.ndarray,   # [N, N], in [0, 1]
    gt_rep_d: np.ndarray,        # [N, N], NaN at unresolved positions
    gt_rep_mask: np.ndarray,     # [N, N], True iff both rep atoms resolved
    n_residues: int,
) -> dict[str, float | int]:
    """CASP-style top-L/k contact precision per separation class.

    For each (short / medium / long) sequence-separation class:
      * Restrict to pairs (i, j) with i < j, |j - i| in the class's
        range, AND both rep atoms resolved in GT.
      * Rank those pairs by predicted ``contact_probs[i, j]`` descending.
      * For k ∈ {1, 2, 5} take the top ``L // k`` (at least 1) pairs;
        precision = (# true contacts in top-k) / (# returned).
      * Also report the total true-contact count in the class.

    Returns a dict keyed by the corresponding ``_FIELDS`` columns
    (``prec_<range>_L``, ``prec_<range>_L_2``, ``prec_<range>_L_5``,
    ``n_<range>_contacts``). Precision is NaN if the class has no
    eligible pairs (i.e. the protein is too short to have any).
    """
    is_contact_pair = (gt_rep_d <= _CONTACT_CUTOFF_A) & gt_rep_mask
    out: dict[str, float | int] = {}
    for range_name, (sep_lo, sep_hi) in _CASP_SEPARATIONS.items():
        col_template = f"prec_{range_name}_L{{}}"
        # Build (i, j) pair list for the class, only resolved pairs.
        rows: list[int] = []
        cols: list[int] = []
        for i in range(n_residues):
            j_start = i + sep_lo
            j_end = n_residues if sep_hi is None else min(n_residues, i + sep_hi + 1)
            for j in range(j_start, j_end):
                if gt_rep_mask[i, j]:
                    rows.append(i)
                    cols.append(j)
        if not rows:
            for k in _CASP_TOP_K:
                col = col_template.format("" if k == 1 else f"_{k}")
                out[col] = float("nan")
            out[f"n_{range_name}_contacts"] = 0
            continue
        rows_arr = np.asarray(rows)
        cols_arr = np.asarray(cols)
        scores = contact_probs[rows_arr, cols_arr]
        gt_binary = is_contact_pair[rows_arr, cols_arr].astype(np.int64)
        # Sort by predicted contact probability descending. We use
        # mergesort for stability across runs (CASP-friendly).
        order = np.argsort(-scores, kind="mergesort")
        sorted_gt = gt_binary[order]
        for k in _CASP_TOP_K:
            top_n = max(1, n_residues // k)
            top_n = min(top_n, len(sorted_gt))
            precision = float(sorted_gt[:top_n].sum()) / top_n
            col = col_template.format("" if k == 1 else f"_{k}")
            out[col] = precision
        out[f"n_{range_name}_contacts"] = int(gt_binary.sum())
    return out


@dataclass(frozen=True)
class ProteinScore:
    pdb_id: str
    chain_id: str
    mode: str
    seed: int
    sample_idx: int
    ranking_score: float
    n_residues: int
    # Distogram-derived MAE + dRMSD on CB-CB (CA-for-GLY), filtered to
    # pairs whose GT is in the distogram's expressible range.
    mae_distogram_cb_angstrom: float
    drmsd_distogram_cb_angstrom: float
    n_mae_distogram_pairs: int
    # Same, filtered to the contact-regime (GT ≤ 8 Å).
    mae_distogram_cb_contact_angstrom: float
    drmsd_distogram_cb_contact_angstrom: float
    n_mae_distogram_contact_pairs: int
    # Structure-derived MAE on CA-CA (predicted vs GT pairwise distances).
    mae_structure_ca_angstrom: float
    # dRMSD on CA-CA (RMSE of pred-vs-GT pairwise CA distances).
    drmsd_ca_angstrom: float
    n_ca_pairs: int
    # Kabsch RMSD over CA atoms.
    rmsd_ca_angstrom: float
    n_ca_atoms: int
    # Kabsch RMSD over all matching heavy atoms.
    rmsd_all_heavy_angstrom: float
    n_heavy_atoms: int
    # CASP-style contact precision per separation class @ top L / L/2 / L/5.
    prec_short_L: float
    prec_short_L_2: float
    prec_short_L_5: float
    prec_medium_L: float
    prec_medium_L_2: float
    prec_medium_L_5: float
    prec_long_L: float
    prec_long_L_2: float
    prec_long_L_5: float
    n_short_contacts: int
    n_medium_contacts: int
    n_long_contacts: int


@dataclass(frozen=True)
class _MetricResult:
    """The metric outputs of comparing one (pred, GT) pair. No ID fields."""

    n_residues: int
    # In-range distogram MAE/dRMSD (option B).
    mae_distogram_cb_angstrom: float
    drmsd_distogram_cb_angstrom: float
    n_mae_distogram_pairs: int
    # Contact-regime distogram MAE/dRMSD (option C1).
    mae_distogram_cb_contact_angstrom: float
    drmsd_distogram_cb_contact_angstrom: float
    n_mae_distogram_contact_pairs: int
    # Structure metrics.
    mae_structure_ca_angstrom: float
    drmsd_ca_angstrom: float
    n_ca_pairs: int
    rmsd_ca_angstrom: float
    n_ca_atoms: int
    rmsd_all_heavy_angstrom: float
    n_heavy_atoms: int
    # CASP contact precision (option C2).
    prec_short_L: float
    prec_short_L_2: float
    prec_short_L_5: float
    prec_medium_L: float
    prec_medium_L_2: float
    prec_medium_L_5: float
    prec_long_L: float
    prec_long_L_2: float
    prec_long_L_5: float
    n_short_contacts: int
    n_medium_contacts: int
    n_long_contacts: int


def _compute_metrics(
    *,
    pred_cif: Path,
    gt_cif: Path,
    distogram_npz: Path,
    pred_coords_cache: ProteinCoords | None = None,
    gt_coords_cache: ProteinCoords | None = None,
    gt_atoms_cache: dict[tuple[int, str], tuple[float, float, float]] | None = None,
    distogram_derived_cache: _DistogramDerived | None = None,
) -> _MetricResult:
    """Compute the full metric suite for one (pred_cif, gt_cif, distogram_npz).

    Per-sample loops (e.g. ``score-all-samples``) re-use the GT coordinates
    + heavy-atom positions + (per-seed) distogram across the 40 samples
    that share them. Caches let callers avoid re-parsing the same CIF /
    .npz 8 times per seed. ``None`` for any cache means "parse from disk".
    """
    gt_coords = gt_coords_cache or _read_protein_coords_from_cif(gt_cif)
    pred_coords = pred_coords_cache or _read_protein_coords_from_cif(pred_cif)
    if gt_coords.n_residues != pred_coords.n_residues:
        # Truncate to the shorter chain (rare; only seen with malformed
        # CIFs). The downstream pair sets are masked by what's resolved
        # in each.
        pass  # logged once at the caller level
    n = min(gt_coords.n_residues, pred_coords.n_residues)

    derived = distogram_derived_cache or _load_distogram_derivatives(distogram_npz)
    expected = derived.expected_distances
    contact_probs = derived.contact_probabilities
    if expected.shape != (n, n):
        m = min(expected.shape[0], n)
        expected = expected[:m, :m]
        contact_probs = contact_probs[:m, :m]
        n = m

    iu = np.triu_indices(n, k=1)
    gt_rep_d, gt_rep_mask = _pairwise_distance_matrix(gt_coords.rep[:n])

    # 1a. Distogram MAE + dRMSD — in-range pair set (option B).
    # Filter: pair resolved in GT AND 2.31 <= gt <= 21.84 (Protenix v2's
    # distogram range). This removes pairs whose GT distance the model
    # literally cannot express; without the filter the metric is
    # dominated by the clipping bias on far pairs.
    gt_rep_d_iu = gt_rep_d[iu]
    inrange_mask = (
        gt_rep_mask[iu]
        & (gt_rep_d_iu >= _DISTOGRAM_MIN_A)
        & (gt_rep_d_iu <= _DISTOGRAM_MAX_A)
    )
    n_mae_disto_pairs = int(inrange_mask.sum())
    if n_mae_disto_pairs == 0:
        mae_disto = float("nan")
        drmsd_disto = float("nan")
    else:
        disto_diffs = expected[iu][inrange_mask] - gt_rep_d_iu[inrange_mask]
        mae_disto = float(np.mean(np.abs(disto_diffs)))
        drmsd_disto = float(np.sqrt(np.mean(disto_diffs ** 2)))

    # 1b. Distogram MAE + dRMSD — contact-regime pair set (option C1).
    # CASP contact convention: GT <= 8 Å. Same atom (CB / CA-for-GLY).
    contact_mask = gt_rep_mask[iu] & (gt_rep_d_iu <= _CONTACT_CUTOFF_A)
    n_disto_contact_pairs = int(contact_mask.sum())
    if n_disto_contact_pairs == 0:
        mae_disto_contact = float("nan")
        drmsd_disto_contact = float("nan")
    else:
        disto_diffs_c = expected[iu][contact_mask] - gt_rep_d_iu[contact_mask]
        mae_disto_contact = float(np.mean(np.abs(disto_diffs_c)))
        drmsd_disto_contact = float(np.sqrt(np.mean(disto_diffs_c ** 2)))

    # 1c. CASP contact precision (option C2).
    casp = _casp_contact_precisions(
        contact_probs=contact_probs,
        gt_rep_d=gt_rep_d,
        gt_rep_mask=gt_rep_mask,
        n_residues=n,
    )

    # 2 + 3. Structure-derived MAE + dRMSD on CA-CA (same pair set).
    gt_ca_d, gt_ca_mask = _pairwise_distance_matrix(gt_coords.ca[:n])
    pred_ca_d, pred_ca_mask = _pairwise_distance_matrix(pred_coords.ca[:n])
    ca_usable = gt_ca_mask[iu] & pred_ca_mask[iu]
    n_ca_pairs = int(ca_usable.sum())
    if n_ca_pairs == 0:
        mae_struct = float("nan")
        drmsd = float("nan")
    else:
        ca_diffs = pred_ca_d[iu][ca_usable] - gt_ca_d[iu][ca_usable]
        mae_struct = float(np.mean(np.abs(ca_diffs)))
        drmsd = float(np.sqrt(np.mean(ca_diffs ** 2)))

    # 4. Kabsch RMSD over CA atoms.
    pred_ca_xyz: list[tuple[float, float, float]] = []
    gt_ca_xyz: list[tuple[float, float, float]] = []
    for i in range(n):
        if pred_coords.ca[i] is not None and gt_coords.ca[i] is not None:
            pred_ca_xyz.append(pred_coords.ca[i])
            gt_ca_xyz.append(gt_coords.ca[i])
    rmsd_ca = _kabsch_rmsd(pred_ca_xyz, gt_ca_xyz)
    n_ca_atoms = len(pred_ca_xyz)

    # 5. Kabsch RMSD over all matching heavy atoms.
    pred_atoms = _read_heavy_atom_positions(pred_cif)
    gt_atoms = gt_atoms_cache or _read_heavy_atom_positions(gt_cif)
    shared_keys = sorted(set(pred_atoms) & set(gt_atoms))
    pred_heavy = [pred_atoms[k] for k in shared_keys]
    gt_heavy = [gt_atoms[k] for k in shared_keys]
    rmsd_heavy = _kabsch_rmsd(pred_heavy, gt_heavy)
    n_heavy_atoms = len(shared_keys)

    return _MetricResult(
        n_residues=n,
        mae_distogram_cb_angstrom=mae_disto,
        drmsd_distogram_cb_angstrom=drmsd_disto,
        n_mae_distogram_pairs=n_mae_disto_pairs,
        mae_distogram_cb_contact_angstrom=mae_disto_contact,
        drmsd_distogram_cb_contact_angstrom=drmsd_disto_contact,
        n_mae_distogram_contact_pairs=n_disto_contact_pairs,
        mae_structure_ca_angstrom=mae_struct,
        drmsd_ca_angstrom=drmsd,
        n_ca_pairs=n_ca_pairs,
        rmsd_ca_angstrom=rmsd_ca,
        n_ca_atoms=n_ca_atoms,
        rmsd_all_heavy_angstrom=rmsd_heavy,
        n_heavy_atoms=n_heavy_atoms,
        prec_short_L=casp["prec_short_L"],
        prec_short_L_2=casp["prec_short_L_2"],
        prec_short_L_5=casp["prec_short_L_5"],
        prec_medium_L=casp["prec_medium_L"],
        prec_medium_L_2=casp["prec_medium_L_2"],
        prec_medium_L_5=casp["prec_medium_L_5"],
        prec_long_L=casp["prec_long_L"],
        prec_long_L_2=casp["prec_long_L_2"],
        prec_long_L_5=casp["prec_long_L_5"],
        n_short_contacts=casp["n_short_contacts"],
        n_medium_contacts=casp["n_medium_contacts"],
        n_long_contacts=casp["n_long_contacts"],
    )


def score(
    *,
    best_dir: Path,
    inputs_dir: Path,
    out_csv: Path,
    modes: list[str],
) -> list[ProteinScore]:
    """For each (mode, stem) under ``best_dir``, compute MAE + dRMSD vs the GT in ``inputs_dir/gt/``."""
    import csv as _csv
    manifest_path = inputs_dir / "manifest.csv"
    with manifest_path.open() as f:
        manifest = list(_csv.DictReader(f))

    rows: list[ProteinScore] = []
    for mode in modes:
        for entry in manifest:
            stem = entry["stem"]
            best_subdir = best_dir / mode / stem
            if not best_subdir.exists():
                print(f"WARN: missing best/{mode}/{stem}; skipping.")
                continue
            gt_cif = inputs_dir / "gt" / f"{stem}.cif"
            pred_cif = best_subdir / "structure.cif"
            distogram_npz = best_subdir / "distogram.npz"
            provenance = json.loads((best_subdir / "provenance.json").read_text())

            m = _compute_metrics(
                pred_cif=pred_cif, gt_cif=gt_cif, distogram_npz=distogram_npz,
            )
            rows.append(ProteinScore(
                pdb_id=entry["pdb_id"],
                chain_id=entry["chain_id"],
                mode=mode,
                seed=int(provenance["seed"]),
                sample_idx=int(provenance["sample_idx"]),
                ranking_score=float(provenance["ranking_score"]),
                n_residues=m.n_residues,
                mae_distogram_cb_angstrom=m.mae_distogram_cb_angstrom,
                drmsd_distogram_cb_angstrom=m.drmsd_distogram_cb_angstrom,
                n_mae_distogram_pairs=m.n_mae_distogram_pairs,
                mae_distogram_cb_contact_angstrom=m.mae_distogram_cb_contact_angstrom,
                drmsd_distogram_cb_contact_angstrom=m.drmsd_distogram_cb_contact_angstrom,
                n_mae_distogram_contact_pairs=m.n_mae_distogram_contact_pairs,
                mae_structure_ca_angstrom=m.mae_structure_ca_angstrom,
                drmsd_ca_angstrom=m.drmsd_ca_angstrom,
                n_ca_pairs=m.n_ca_pairs,
                rmsd_ca_angstrom=m.rmsd_ca_angstrom,
                n_ca_atoms=m.n_ca_atoms,
                rmsd_all_heavy_angstrom=m.rmsd_all_heavy_angstrom,
                n_heavy_atoms=m.n_heavy_atoms,
                prec_short_L=m.prec_short_L,
                prec_short_L_2=m.prec_short_L_2,
                prec_short_L_5=m.prec_short_L_5,
                prec_medium_L=m.prec_medium_L,
                prec_medium_L_2=m.prec_medium_L_2,
                prec_medium_L_5=m.prec_medium_L_5,
                prec_long_L=m.prec_long_L,
                prec_long_L_2=m.prec_long_L_2,
                prec_long_L_5=m.prec_long_L_5,
                n_short_contacts=m.n_short_contacts,
                n_medium_contacts=m.n_medium_contacts,
                n_long_contacts=m.n_long_contacts,
            ))
            print(
                f"{mode}/{stem}: n_res={m.n_residues} "
                f"MAE_disto[in]={m.mae_distogram_cb_angstrom:.3f} "
                f"MAE_disto[ct]={m.mae_distogram_cb_contact_angstrom:.3f} "
                f"dRMSD[in]={m.drmsd_distogram_cb_angstrom:.3f} "
                f"MAE_struct={m.mae_structure_ca_angstrom:.3f} "
                f"RMSD_CA={m.rmsd_ca_angstrom:.3f} "
                f"P_long_L={m.prec_long_L:.3f} P_long_L/5={m.prec_long_L_5:.3f}"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_FIELDS)
        for r in rows:
            # selected_as_best=1 always: this is the top-1 picker.
            writer.writerow(_format_csv_row(r, selected_as_best=1))
    print(f"Wrote {out_csv} with {len(rows)} rows.")
    return rows


def _format_csv_row(r, *, selected_as_best: int) -> list:
    """One CSV-row formatter shared by score() and score_all_samples()."""
    def _f(x: float) -> str:
        return f"{x:.4f}" if x == x else "nan"  # NaN passthrough
    return [
        r.pdb_id, r.chain_id, r.mode, r.seed, r.sample_idx, r.ranking_score,
        selected_as_best,
        r.n_residues,
        _f(r.mae_distogram_cb_angstrom), _f(r.drmsd_distogram_cb_angstrom),
        r.n_mae_distogram_pairs,
        _f(r.mae_distogram_cb_contact_angstrom), _f(r.drmsd_distogram_cb_contact_angstrom),
        r.n_mae_distogram_contact_pairs,
        _f(r.mae_structure_ca_angstrom),
        _f(r.drmsd_ca_angstrom), r.n_ca_pairs,
        _f(r.rmsd_ca_angstrom), r.n_ca_atoms,
        _f(r.rmsd_all_heavy_angstrom), r.n_heavy_atoms,
        _f(r.prec_short_L), _f(r.prec_short_L_2), _f(r.prec_short_L_5),
        _f(r.prec_medium_L), _f(r.prec_medium_L_2), _f(r.prec_medium_L_5),
        _f(r.prec_long_L), _f(r.prec_long_L_2), _f(r.prec_long_L_5),
        r.n_short_contacts, r.n_medium_contacts, r.n_long_contacts,
    ]


def score_all_samples(
    *,
    runs_dir: Path,
    inputs_dir: Path,
    out_csv: Path,
    modes: list[str],
    stems_filter: list[str] | None = None,
    append: bool = False,
) -> int:
    """For each (mode, stem) under ``runs_dir``, score ALL 40 samples per
    (seed × sample) — not just the top-1 — and append rows to ``out_csv``.

    Expects the raw Protenix output layout under ``runs_dir``::

        runs_dir/{mode}/{stem}/seed_<S>/
            {stem}_sample_<I>.cif
            {stem}_summary_confidence_sample_<I>.json
            {stem}_distogram.npz   # one per seed; shared across the 8 samples

    Per-seed caching: the GT coords, GT heavy atoms, and the distogram
    expected-distance matrix are parsed once per seed and reused across
    the 8 samples in that seed (the distogram is per-seed by construction
    of Protenix's trunk loop).

    Idempotent via the ``append`` flag — when True, ``out_csv`` is opened
    in append mode and existing rows are skipped via a (mode, stem,
    seed, sample_idx) key.

    ``stems_filter`` (if given) restricts to those stems; otherwise all
    stems in ``inputs_dir/manifest.csv`` are processed.
    """
    import csv as _csv

    manifest = list(_csv.DictReader((inputs_dir / "manifest.csv").open()))
    if stems_filter is not None:
        wanted = set(stems_filter)
        manifest = [r for r in manifest if r["stem"] in wanted]

    # Idempotency: load existing (mode, stem, seed, sample_idx) tuples.
    seen: set[tuple[str, str, int, int]] = set()
    write_header = True
    if append and out_csv.exists():
        with out_csv.open() as f:
            reader = _csv.DictReader(f)
            for row in reader:
                seen.add((
                    row["mode"], f"{row['pdb_id']}_{row['chain_id']}",
                    int(row["seed"]), int(row["sample_idx"]),
                ))
        write_header = False
        print(f"resuming from {out_csv}: {len(seen)} existing rows")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mode_str = "a" if append and out_csv.exists() else "w"
    f_out = out_csv.open(mode_str, newline="")
    writer = csv.writer(f_out)
    if write_header:
        writer.writerow(_FIELDS)

    n_written = 0
    try:
        for entry in manifest:
            stem = entry["stem"]
            gt_cif = inputs_dir / "gt" / f"{stem}.cif"
            # Parse GT once per stem; reused across the 40 samples in each mode.
            gt_coords = _read_protein_coords_from_cif(gt_cif)
            gt_atoms = _read_heavy_atom_positions(gt_cif)
            # Buffer this stem's rows so we can stamp ``selected_as_best``
            # before writing — top-1 by ranking_score *per mode*.
            stem_rows_by_mode: dict[str, list[dict]] = {m: [] for m in modes}
            for mode in modes:
                mode_stem_dir = runs_dir / mode / stem
                if not mode_stem_dir.exists():
                    continue
                for seed_dir in sorted(mode_stem_dir.glob("seed_*")):
                    seed = int(seed_dir.name.removeprefix("seed_"))
                    distogram_npz = seed_dir / f"{stem}_distogram.npz"
                    if not distogram_npz.exists():
                        print(f"WARN: missing distogram {distogram_npz}; skipping seed.")
                        continue
                    # Parse the distogram once per seed (gets both
                    # expected-distance and contact-probability matrices).
                    derived = _load_distogram_derivatives(distogram_npz)
                    # Iterate samples within the seed.
                    conf_prefix = f"{stem}_summary_confidence_sample_"
                    for conf_path in sorted(seed_dir.glob(f"{conf_prefix}*.json")):
                        idx_part = conf_path.stem[len(conf_prefix):]
                        if not idx_part.isdigit():
                            continue
                        sample_idx = int(idx_part)
                        if (mode, stem, seed, sample_idx) in seen:
                            continue
                        cif = seed_dir / f"{stem}_sample_{sample_idx}.cif"
                        if not cif.exists():
                            continue
                        ranking_score = float(
                            json.loads(conf_path.read_text())
                            .get("ranking_score", float("nan"))
                        )
                        m = _compute_metrics(
                            pred_cif=cif, gt_cif=gt_cif, distogram_npz=distogram_npz,
                            gt_coords_cache=gt_coords,
                            gt_atoms_cache=gt_atoms,
                            distogram_derived_cache=derived,
                        )
                        stem_rows_by_mode[mode].append({
                            "ranking_score": ranking_score, "seed": seed,
                            "sample_idx": sample_idx, "metrics": m,
                        })
            # Mark top-1 per mode (by ranking_score; ties broken by lower
            # seed then lower sample_idx for determinism).
            for mode, mode_rows in stem_rows_by_mode.items():
                if not mode_rows:
                    continue
                best_idx = max(
                    range(len(mode_rows)),
                    key=lambda i: (
                        mode_rows[i]["ranking_score"],
                        -mode_rows[i]["seed"],
                        -mode_rows[i]["sample_idx"],
                    ),
                )
                for i, row in enumerate(mode_rows):
                    row["selected_as_best"] = 1 if i == best_idx else 0
            # Write all rows for this stem in (mode, seed, sample_idx) order.
            # We need a "ProteinScore-like" record for _format_csv_row;
            # build a tiny adapter that carries everything from the
            # _MetricResult plus the id fields.
            class _Adapter:
                __slots__ = ("pdb_id", "chain_id", "mode", "seed", "sample_idx",
                             "ranking_score", *_MetricResult.__dataclass_fields__.keys())
            for mode in modes:
                for row in stem_rows_by_mode[mode]:
                    rm = row["metrics"]
                    a = _Adapter()
                    a.pdb_id = entry["pdb_id"]
                    a.chain_id = entry["chain_id"]
                    a.mode = mode
                    a.seed = row["seed"]
                    a.sample_idx = row["sample_idx"]
                    # Use full repr so downstream "top-1 by ranking_score"
                    # picks deterministically; 6-dp rounding produces
                    # spurious ties (seen for 5sbj_A msa seeds 1 vs 3).
                    a.ranking_score = repr(row["ranking_score"])
                    for f in _MetricResult.__dataclass_fields__:
                        setattr(a, f, getattr(rm, f))
                    writer.writerow(_format_csv_row(a, selected_as_best=row["selected_as_best"]))
                    n_written += 1
            f_out.flush()  # crash-safe under streaming sync
            print(f"{stem}: scored ({n_written} rows so far)")
    finally:
        f_out.close()
    print(f"Wrote {n_written} new rows to {out_csv}.")
    return n_written


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "score",
        help="MAE (distogram + structure) + dRMSD + RMSD (CA + all-heavy) per (protein, mode).",
    )
    p.add_argument("--best", type=Path, required=True, help="Dir produced by select-best (with {mode}/{stem}/ subtree).")
    p.add_argument("--inputs", type=Path, required=True, help="Dir produced by prepare-inputs (contains manifest.csv + gt/).")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path (typically data/scores.csv).")
    p.add_argument("--modes", default="single_seq,msa", help="Comma-separated mode names.")
    p.set_defaults(func=lambda args: score(
        best_dir=args.best, inputs_dir=args.inputs, out_csv=args.out, modes=args.modes.split(","),
    ))

    # --- score-all-samples ---
    p2 = subparsers.add_parser(
        "score-all-samples",
        help="Score every sample (not just top-1) under a raw outputs/ tree. One row per (mode, stem, seed, sample_idx).",
    )
    p2.add_argument(
        "--runs", type=Path, required=True,
        help="Dir mirroring the Modal output Volume layout: {mode}/{stem}/seed_*/{stem}_sample_*.cif ...",
    )
    p2.add_argument("--inputs", type=Path, required=True, help="Dir produced by prepare-inputs.")
    p2.add_argument("--out", type=Path, required=True, help="Output CSV path (typically data/scores_all_samples.csv).")
    p2.add_argument("--modes", default="single_seq,msa", help="Comma-separated mode names.")
    p2.add_argument("--stems-file", default=None, help="Optional path with one stem per line; restricts to those stems.")
    p2.add_argument("--append", action="store_true",
                    help="Append to an existing CSV; skip (mode, stem, seed, sample_idx) tuples already present. "
                         "Useful for streaming runs that score per-stem then delete the local copy.")
    p2.set_defaults(func=lambda args: score_all_samples(
        runs_dir=args.runs, inputs_dir=args.inputs, out_csv=args.out,
        modes=args.modes.split(","),
        stems_filter=(Path(args.stems_file).read_text().split() if args.stems_file else None),
        append=args.append,
    ))

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the best Protenix sample per (protein, mode) against GT.

Two per-protein metrics (one row per (protein, mode) in ``scores.csv``):

- **MAE** — mean absolute error between Protenix's expected distance
  (``Σ p_bin · midpoint`` over the 64 bins of its distogram) and the GT
  **CB-CB distance** (CA-CA for glycine, per the AF3 / Protenix
  representative-atom convention), averaged over residue pairs where
  both representative atoms are resolved in the GT.

- **dRMSD** — sqrt of mean squared difference between the predicted
  **CA-CA** pairwise distance matrix (from the best mmCIF) and the GT
  CA-CA pairwise distance matrix, over the same pair set (resolved-in-GT
  CA pairs). No sequence-separation filter (per agreement with author).

Why different atoms: Protenix's distogram head represents each token
by its CB (CA for glycine) — see ``add_distogram_rep_atom_mask`` in
bytedance/Protenix protenix/data/core/parser.py. Comparing the
distogram-derived expected distance against a GT CA-CA distance would
be a biased apples-to-oranges comparison, so MAE uses the matching
atom set. dRMSD is a property of the predicted *structure*, not the
distogram — the user specified CA-CA for that.

Residue index alignment: Protenix's output sequence is length-N (same
as input), 1-indexed. The GT CIF's residues are mapped to the
canonical sequence via gemmi's ``label_seq_id`` (which matches the
canonical 1..N order). Unresolved residues — those in
``_entity_poly_seq`` but with no corresponding ``_atom_site`` row for
the relevant atom — get masked out.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np


# Distogram bin centers — must match Protenix v2's distogram head config.
# From bytedance/Protenix configs/configs_base.py's loss.distogram block:
#   min_bin = 2.3125, max_bin = 21.6875, no_bins = 64
# And bytedance/Protenix protenix/model/sample_confidence.py:get_bin_centers:
#   bin_width = (max_bin - min_bin) / no_bins
#   boundaries = linspace(min_bin, max_bin - bin_width, no_bins)
#   centers    = boundaries + 0.5 * bin_width
# Out-of-range distances (>21.84 Å) go into the last bin; the
# corresponding bin center underestimates them, which biases MAE
# slightly upward on far pairs. This is the AF3-standard convention.
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


def _expected_distances_from_distogram(distogram_path: Path) -> np.ndarray:
    """Load distogram (.npz with key 'probs' shape [N, N, 64]); return expected-distance matrix [N, N].

    If the .npz key is 'logits' instead, softmax along the last axis
    first. The Modal-side hook saves probabilities directly (after
    softmax) to avoid this branch, but we handle both for forward-compat.
    """
    data = np.load(distogram_path)
    if "probs" in data:
        probs = data["probs"].astype(np.float64)
    elif "logits" in data:
        logits = data["logits"].astype(np.float64)
        # Stable softmax along last axis.
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
    return expected


@dataclass(frozen=True)
class ProteinScore:
    pdb_id: str
    chain_id: str
    mode: str
    seed: int
    sample_idx: int
    ranking_score: float
    n_residues: int
    n_pairs_scored: int
    mae_angstrom: float
    drmsd_angstrom: float


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

            gt_coords = _read_protein_coords_from_cif(gt_cif)
            pred_coords = _read_protein_coords_from_cif(pred_cif)
            if gt_coords.n_residues != pred_coords.n_residues:
                print(
                    f"WARN: {mode}/{stem}: GT has {gt_coords.n_residues} residues, "
                    f"prediction has {pred_coords.n_residues}; truncating to min."
                )
            n = min(gt_coords.n_residues, pred_coords.n_residues)

            expected = _expected_distances_from_distogram(distogram_npz)
            if expected.shape != (n, n):
                # Truncate to the smaller dimension if the distogram
                # exceeds GT/pred length (e.g. token padding round-trip).
                m = min(expected.shape[0], n)
                expected = expected[:m, :m]
                n = m

            # MAE: expected distance vs GT CB-or-CA-for-GLY distance.
            gt_rep_d, gt_rep_mask = _pairwise_distance_matrix(gt_coords.rep[:n])
            iu = np.triu_indices(n, k=1)
            mae_usable = gt_rep_mask[iu]
            n_mae_pairs = int(mae_usable.sum())
            if n_mae_pairs == 0:
                print(f"WARN: {mode}/{stem}: no usable GT representative-atom pairs; skipping MAE.")
                mae = float("nan")
            else:
                mae = float(np.mean(np.abs(
                    expected[iu][mae_usable] - gt_rep_d[iu][mae_usable]
                )))

            # dRMSD: predicted CA-CA vs GT CA-CA.
            gt_ca_d, gt_ca_mask = _pairwise_distance_matrix(gt_coords.ca[:n])
            pred_ca_d, pred_ca_mask = _pairwise_distance_matrix(pred_coords.ca[:n])
            drmsd_usable = gt_ca_mask[iu] & pred_ca_mask[iu]
            n_drmsd_pairs = int(drmsd_usable.sum())
            if n_drmsd_pairs == 0:
                print(f"WARN: {mode}/{stem}: no usable CA pairs; skipping dRMSD.")
                drmsd = float("nan")
            else:
                drmsd = float(np.sqrt(np.mean(
                    (pred_ca_d[iu][drmsd_usable] - gt_ca_d[iu][drmsd_usable]) ** 2
                )))

            rows.append(ProteinScore(
                pdb_id=entry["pdb_id"],
                chain_id=entry["chain_id"],
                mode=mode,
                seed=int(provenance["seed"]),
                sample_idx=int(provenance["sample_idx"]),
                ranking_score=float(provenance["ranking_score"]),
                n_residues=n,
                n_pairs_scored=n_drmsd_pairs,
                mae_angstrom=mae,
                drmsd_angstrom=drmsd,
            ))
            print(
                f"{mode}/{stem}: n_res={n} "
                f"mae_pairs={n_mae_pairs} drmsd_pairs={n_drmsd_pairs} "
                f"MAE={mae:.3f} dRMSD={drmsd:.3f}"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pdb_id", "chain_id", "mode", "seed", "sample_idx", "ranking_score",
            "n_residues", "n_pairs_scored", "mae_angstrom", "drmsd_angstrom",
        ])
        for r in rows:
            writer.writerow([
                r.pdb_id, r.chain_id, r.mode, r.seed, r.sample_idx, r.ranking_score,
                r.n_residues, r.n_pairs_scored,
                f"{r.mae_angstrom:.4f}", f"{r.drmsd_angstrom:.4f}",
            ])
    print(f"Wrote {out_csv} with {len(rows)} rows.")
    return rows


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("score", help="MAE (vs expected distance) + dRMSD (vs CA-CA distance) per (protein, mode).")
    p.add_argument("--best", type=Path, required=True, help="Dir produced by select-best (with {mode}/{stem}/ subtree).")
    p.add_argument("--inputs", type=Path, required=True, help="Dir produced by prepare-inputs (contains manifest.csv + gt/).")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path (typically data/scores.csv).")
    p.add_argument("--modes", default="single_seq,msa", help="Comma-separated mode names.")
    p.set_defaults(func=lambda args: score(
        best_dir=args.best, inputs_dir=args.inputs, out_csv=args.out, modes=args.modes.split(","),
    ))

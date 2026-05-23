# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score MarinFold 1B distograms against the FoldBench GT.

Mirror of the distogram-only portion of exp12's ``score.py``, with the
bin scheme made a parameter so we can plug in either model's bins.
Structure-derived metrics (CA RMSD, lDDT-CA-from-structure, etc.) are
NOT computed here — 1B doesn't emit a structure in this eval and the
issue calls out that this non-parity is acceptable.

For cross-model fairness, the *in-range* distogram pair filter uses
the **intersection** of MarinFold's range (0..32 Å) and Protenix's
(2.3125..21.6875 Å). Per exp12's cross-model section that's
``[2.3125, 21.6875]`` — Protenix is the narrower one. The same filter
is what exp12 used for the Protenix scores in
``protenix_data/scores.csv``, which keeps the headline comparison
honest.

Output CSV schema (one row per protein):

    pdb_id, chain_id, method, n_residues,
    mae_distogram_cb_angstrom, drmsd_distogram_cb_angstrom,
    n_mae_distogram_pairs,
    mae_distogram_cb_contact_angstrom, drmsd_distogram_cb_contact_angstrom,
    n_mae_distogram_contact_pairs,
    prec_short_L, prec_short_L_2, prec_short_L_5,
    prec_medium_L, prec_medium_L_2, prec_medium_L_5,
    prec_long_L, prec_long_L_2, prec_long_L_5,
    n_short_contacts, n_medium_contacts, n_long_contacts,
    lddt_distogram_cb, lddt_distogram_cb_soft

``method`` is fixed to ``"marinfold_1b"`` here. The Protenix-side rows
are produced separately in ``score_comparison.py``.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np

from canonical_sequence import normalize_residue_name, representative_atom_name


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


# CASP contact convention. Same as exp12.
_CONTACT_CUTOFF_A = 8.0
_CASP_SEPARATIONS: dict[str, tuple[int, int | None]] = {
    "short": (6, 11),
    "medium": (12, 23),
    "long": (24, None),
}
_CASP_TOP_K: tuple[int, ...] = (1, 2, 5)

# LDDT convention. Same as exp12.
_LDDT_INCLUSION_RADIUS_A: float = 15.0
_LDDT_THRESHOLDS_A: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
_LDDT_MIN_SEPARATION: int = 1

# Cross-model in-range pair filter — intersection of MarinFold (0..32 Å)
# and Protenix (2.3125..21.6875 Å). Per exp12's "Cross-model comparison"
# section: score MarinFold on the same Protenix-defined pair set.
_INRANGE_MIN_A = 2.3125
_INRANGE_MAX_A = 21.6875


@dataclass(frozen=True)
class BinScheme:
    """Distogram bin scheme for ONE model.

    Per-pair predicted distance = ``(probs * midpoints).sum(axis=-1)``.
    The MAE/dRMSD pair filter uses the *intersection range*
    (``_INRANGE_MIN_A``..``_INRANGE_MAX_A``) regardless of which model
    this is for — that's the cross-model fairness rule.
    """

    n_bins: int
    midpoints_A: np.ndarray  # shape [n_bins], in Å

    @property
    def contact_bin_mask(self) -> np.ndarray:
        """Bins whose center is ≤ 8 Å (used to derive contact probs)."""
        return self.midpoints_A <= _CONTACT_CUTOFF_A


# The MarinFold 1B scheme. 0.5 Å bins, midpoints 0.25..31.75 Å.
# Matches exp1's vocab.py DISTANCE_BINS construction.
MARINFOLD_BINS = BinScheme(
    n_bins=64,
    midpoints_A=np.array(
        [(k + 1) * 0.5 - 0.25 for k in range(64)], dtype=np.float64,
    ),
)


# --------------------------------------------------------------------------
# GT-side CIF parsing (CB-CB / CA-for-GLY/UNK, label_seq_id indexed)
# --------------------------------------------------------------------------


def _read_gt_rep_coords(cif_path: Path) -> tuple[int, list[tuple[float, float, float] | None]]:
    """Return (n_residues, rep_xyz_list) — same as exp12.

    ``rep_xyz_list[i]`` is the CB-or-CA representative position of residue
    i+1, or None if unresolved in the GT. ``n_residues`` comes from
    ``entity.full_sequence`` so the indexing aligns with the 1B
    distogram (which is also driven by the canonical 1..N sequence).
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
        raise ValueError(f"{cif_path}: multiple polypeptide(L) entities.")
    entity = peptide_entities[0]
    subchains = set(entity.subchains)
    sequence_names = tuple(
        normalize_residue_name(str(raw_name)) for raw_name in entity.full_sequence
    )
    n_residues = len(sequence_names)

    rep_by_seq: dict[int, tuple[float, float, float]] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.subchain not in subchains:
                    continue
                label_seq = residue.label_seq
                if label_seq is None:
                    continue
                if label_seq < 1 or label_seq > n_residues:
                    continue
                if representative_atom_name(sequence_names[label_seq - 1]) == "CA":
                    ca = residue.find_atom("CA", "\0")
                    if ca is not None:
                        rep_by_seq[label_seq] = (ca.pos.x, ca.pos.y, ca.pos.z)
                else:
                    cb = residue.find_atom("CB", "\0")
                    if cb is not None:
                        rep_by_seq[label_seq] = (cb.pos.x, cb.pos.y, cb.pos.z)
        break
    rep = [rep_by_seq.get(i) for i in range(1, n_residues + 1)]
    return n_residues, rep


def _pairwise_distance_matrix(
    positions: list[tuple[float, float, float] | None],
) -> tuple[np.ndarray, np.ndarray]:
    """[N, N] distance matrix + boolean mask of resolved pairs."""
    n = len(positions)
    arr = np.full((n, 3), np.nan, dtype=np.float64)
    for i, p in enumerate(positions):
        if p is not None:
            arr[i] = p
    diffs = arr[:, None, :] - arr[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    mask = np.isfinite(distances)
    return distances, mask


# --------------------------------------------------------------------------
# LDDT (CASP convention: 15 Å inclusion, thresholds 0.5/1/2/4 Å)
# --------------------------------------------------------------------------


def _lddt_inclusion_mask(
    gt_d: np.ndarray, gt_mask: np.ndarray, *, min_separation: int = _LDDT_MIN_SEPARATION,
) -> np.ndarray:
    n = gt_d.shape[0]
    i_idx = np.arange(n)[:, None]
    j_idx = np.arange(n)[None, :]
    sep_ok = np.abs(i_idx - j_idx) >= min_separation
    return gt_mask & sep_ok & (gt_d < _LDDT_INCLUSION_RADIUS_A)


def _lddt_point(pred_d: np.ndarray, gt_d: np.ndarray, *, pair_mask: np.ndarray) -> float:
    """LDDT given pred + GT pairwise distance matrices (point estimate)."""
    diffs = np.abs(pred_d - gt_d)
    diffs = np.where(np.isnan(diffs), np.inf, diffs)
    preservation = np.mean(
        [(diffs < t).astype(np.float64) for t in _LDDT_THRESHOLDS_A],
        axis=0,
    )
    mask = pair_mask.copy()
    np.fill_diagonal(mask, False)
    denom = mask.sum(axis=1)
    if denom.sum() == 0:
        return float("nan")
    numer = (preservation * mask).sum(axis=1)
    per_residue = np.where(denom > 0, numer / np.maximum(denom, 1), np.nan)
    finite = per_residue[np.isfinite(per_residue)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _lddt_soft(
    probs: np.ndarray,        # [N, N, n_bins]
    gt_d: np.ndarray,         # [N, N]
    *, pair_mask: np.ndarray,
    bins: BinScheme,
) -> float:
    """Probabilistic LDDT: per-pair score = Σ p_bin over bins in (gt-t, gt+t)."""
    centers = bins.midpoints_A
    gt_d_safe = np.where(np.isfinite(gt_d), gt_d, 0.0)
    bin_diff = np.abs(centers[None, None, :] - gt_d_safe[:, :, None])  # [N, N, n_bins]
    preservation_per_threshold = []
    for t in _LDDT_THRESHOLDS_A:
        bin_in_window = (bin_diff < t).astype(np.float64)
        soft = (probs * bin_in_window).sum(axis=-1)
        preservation_per_threshold.append(soft)
    preservation = np.mean(preservation_per_threshold, axis=0)
    mask = pair_mask.copy()
    np.fill_diagonal(mask, False)
    denom = mask.sum(axis=1)
    if denom.sum() == 0:
        return float("nan")
    numer = (preservation * mask).sum(axis=1)
    per_residue = np.where(denom > 0, numer / np.maximum(denom, 1), np.nan)
    finite = per_residue[np.isfinite(per_residue)]
    return float(np.mean(finite)) if finite.size else float("nan")


# --------------------------------------------------------------------------
# CASP contact precision
# --------------------------------------------------------------------------


def _casp_contact_precisions(
    *,
    contact_probs: np.ndarray,
    gt_rep_d: np.ndarray,
    gt_rep_mask: np.ndarray,
    n_residues: int,
) -> dict[str, float | int]:
    is_contact_pair = (gt_rep_d <= _CONTACT_CUTOFF_A) & gt_rep_mask
    out: dict[str, float | int] = {}
    for range_name, (sep_lo, sep_hi) in _CASP_SEPARATIONS.items():
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
                col = f"prec_{range_name}_L" + ("" if k == 1 else f"_{k}")
                out[col] = float("nan")
            out[f"n_{range_name}_contacts"] = 0
            continue
        rows_arr = np.asarray(rows)
        cols_arr = np.asarray(cols)
        scores = contact_probs[rows_arr, cols_arr]
        gt_binary = is_contact_pair[rows_arr, cols_arr].astype(np.int64)
        order = np.argsort(-scores, kind="mergesort")
        sorted_gt = gt_binary[order]
        for k in _CASP_TOP_K:
            top_n = max(1, n_residues // k)
            top_n = min(top_n, len(sorted_gt))
            precision = float(sorted_gt[:top_n].sum()) / top_n
            col = f"prec_{range_name}_L" + ("" if k == 1 else f"_{k}")
            out[col] = precision
        out[f"n_{range_name}_contacts"] = int(gt_binary.sum())
    return out


# --------------------------------------------------------------------------
# Per-protein scoring
# --------------------------------------------------------------------------


_CSV_FIELDS = [
    "pdb_id", "chain_id", "method", "n_residues",
    "mae_distogram_cb_angstrom", "drmsd_distogram_cb_angstrom",
    "n_mae_distogram_pairs",
    "mae_distogram_cb_contact_angstrom", "drmsd_distogram_cb_contact_angstrom",
    "n_mae_distogram_contact_pairs",
    "prec_short_L", "prec_short_L_2", "prec_short_L_5",
    "prec_medium_L", "prec_medium_L_2", "prec_medium_L_5",
    "prec_long_L", "prec_long_L_2", "prec_long_L_5",
    "n_short_contacts", "n_medium_contacts", "n_long_contacts",
    "lddt_distogram_cb", "lddt_distogram_cb_soft",
]


def score_one(
    *,
    distogram_npz: Path,
    gt_cif: Path,
    pdb_id: str,
    chain_id: str,
    method: str,
    bins: BinScheme,
) -> dict[str, float | int | str]:
    """Compute the full distogram-only metric suite for one protein."""
    with np.load(distogram_npz) as data:
        probs = data["probs"].astype(np.float64)
    expected = (probs * bins.midpoints_A).sum(axis=-1)
    contact_probs = probs[..., bins.contact_bin_mask].sum(axis=-1)

    n_gt, rep = _read_gt_rep_coords(gt_cif)
    n = min(n_gt, probs.shape[0])
    if n != probs.shape[0] or n != n_gt:
        # exp12 logs and truncates; do the same.
        print(
            f"WARN: {pdb_id}_{chain_id}: GT n={n_gt}, distogram n={probs.shape[0]} — truncating to {n}."
        )
    expected = expected[:n, :n]
    contact_probs = contact_probs[:n, :n]
    probs = probs[:n, :n, :]
    rep = rep[:n]
    gt_rep_d, gt_rep_mask = _pairwise_distance_matrix(rep)

    iu = np.triu_indices(n, k=1)
    gt_rep_d_iu = gt_rep_d[iu]

    # In-range MAE/dRMSD (option B) — uses intersection range, NOT
    # MarinFold's full range. This is the cross-model fairness rule.
    inrange_mask = (
        gt_rep_mask[iu]
        & (gt_rep_d_iu >= _INRANGE_MIN_A)
        & (gt_rep_d_iu <= _INRANGE_MAX_A)
    )
    n_inrange = int(inrange_mask.sum())
    if n_inrange == 0:
        mae_disto = float("nan")
        drmsd_disto = float("nan")
    else:
        diffs = expected[iu][inrange_mask] - gt_rep_d_iu[inrange_mask]
        mae_disto = float(np.mean(np.abs(diffs)))
        drmsd_disto = float(np.sqrt(np.mean(diffs ** 2)))

    # Contact-regime MAE/dRMSD (option C1).
    contact_mask = gt_rep_mask[iu] & (gt_rep_d_iu <= _CONTACT_CUTOFF_A)
    n_contacts = int(contact_mask.sum())
    if n_contacts == 0:
        mae_contact = float("nan")
        drmsd_contact = float("nan")
    else:
        diffs_c = expected[iu][contact_mask] - gt_rep_d_iu[contact_mask]
        mae_contact = float(np.mean(np.abs(diffs_c)))
        drmsd_contact = float(np.sqrt(np.mean(diffs_c ** 2)))

    # CASP contact precision (option C2).
    casp = _casp_contact_precisions(
        contact_probs=contact_probs,
        gt_rep_d=gt_rep_d,
        gt_rep_mask=gt_rep_mask,
        n_residues=n,
    )

    # LDDT (point + soft). Inclusion mask is gt < 15 Å AND |i-j| >= 1 AND resolved.
    inclusion = _lddt_inclusion_mask(gt_d=gt_rep_d, gt_mask=gt_rep_mask)
    lddt_point = _lddt_point(pred_d=expected, gt_d=gt_rep_d, pair_mask=inclusion)
    lddt_soft = _lddt_soft(probs=probs, gt_d=gt_rep_d, pair_mask=inclusion, bins=bins)

    return {
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "method": method,
        "n_residues": n,
        "mae_distogram_cb_angstrom": mae_disto,
        "drmsd_distogram_cb_angstrom": drmsd_disto,
        "n_mae_distogram_pairs": n_inrange,
        "mae_distogram_cb_contact_angstrom": mae_contact,
        "drmsd_distogram_cb_contact_angstrom": drmsd_contact,
        "n_mae_distogram_contact_pairs": n_contacts,
        "prec_short_L": casp["prec_short_L"],
        "prec_short_L_2": casp["prec_short_L_2"],
        "prec_short_L_5": casp["prec_short_L_5"],
        "prec_medium_L": casp["prec_medium_L"],
        "prec_medium_L_2": casp["prec_medium_L_2"],
        "prec_medium_L_5": casp["prec_medium_L_5"],
        "prec_long_L": casp["prec_long_L"],
        "prec_long_L_2": casp["prec_long_L_2"],
        "prec_long_L_5": casp["prec_long_L_5"],
        "n_short_contacts": casp["n_short_contacts"],
        "n_medium_contacts": casp["n_medium_contacts"],
        "n_long_contacts": casp["n_long_contacts"],
        "lddt_distogram_cb": lddt_point,
        "lddt_distogram_cb_soft": lddt_soft,
    }


def _format_value(x: float | int | str) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return str(x)
    if x != x:  # NaN
        return "nan"
    return f"{x:.4f}"


def score_all(
    *,
    protenix_dir: Path,
    outputs_dir: Path,
    out_csv: Path,
    method: str = "marinfold_1b",
    bins: BinScheme = MARINFOLD_BINS,
) -> int:
    """Score every protein in ``outputs_dir`` that has a distogram.npz."""
    with (protenix_dir / "manifest.csv").open() as f:
        manifest = list(csv.DictReader(f))

    rows: list[dict] = []
    for entry in manifest:
        stem = entry["stem"]
        pdb_id = entry["pdb_id"]
        chain_id = entry["chain_id"]
        distogram_npz = outputs_dir / stem / "distogram.npz"
        if not distogram_npz.exists():
            continue
        gt_cif = protenix_dir / "gt" / f"{stem}.cif"
        row = score_one(
            distogram_npz=distogram_npz,
            gt_cif=gt_cif,
            pdb_id=pdb_id,
            chain_id=chain_id,
            method=method,
            bins=bins,
        )
        rows.append(row)
        print(
            f"{stem}: n={row['n_residues']} "
            f"lDDT_disto={row['lddt_distogram_cb']:.3f} "
            f"lDDT_disto[soft]={row['lddt_distogram_cb_soft']:.3f} "
            f"MAE_disto={row['mae_distogram_cb_angstrom']:.3f}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_FIELDS)
        for row in rows:
            writer.writerow([_format_value(row[col]) for col in _CSV_FIELDS])
    print(f"Wrote {out_csv} with {len(rows)} rows.")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protenix-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "protenix_data" / "data" / "protenix-foldbench-monomers",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Dir containing <stem>/distogram.npz files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "marinfold_scores.csv",
    )
    parser.add_argument("--method", default="marinfold_1_5b")
    args = parser.parse_args()
    score_all(
        protenix_dir=args.protenix_dir,
        outputs_dir=args.outputs_dir,
        out_csv=args.out,
        method=args.method,
    )


if __name__ == "__main__":
    main()

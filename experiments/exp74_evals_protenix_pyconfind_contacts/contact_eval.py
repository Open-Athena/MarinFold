# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score Protenix contact prediction against pyconfind ground truth (4 configs).

Issue #74. Ground truth = pyconfind side-chain contacts on the
experimental structure (degree >= 0.001, primary-sequence separation
>= 6; see ``pyconfind_contacts.py``). Each protein is scored in **four
configurations** = {single_seq, msa} x {distogram, structure}:

- **distogram** predictor: rank candidate pairs by the distogram's
  cumulative probability that the representative atoms (CB, CA-for-GLY)
  are within ``DEFAULT_CONTACT_THRESHOLD_A`` (8 Å) — i.e. the mass on
  Protenix-v2 distogram bins with center <= 8 Å (exp12's contact prob).
- **structure** predictor: run pyconfind on the predicted top-1 structure
  (``native_only=True``, same knobs as the ground truth) and rank
  candidate pairs by predicted contact degree.

Metric: **contacts @ L** (precision among the top-L predicted pairs,
L = sequence length), also @ L/2 and L/5, reported **in aggregate**
(separation >= 6) and **split by range** short [6,11] / medium [12,23]
/ long [>=24]. The candidate-pair universe is restricted to residues
resolved in the ground-truth structure, identically across all four
configs, so the numbers are comparable.

Outputs (tidy, long-form, easy to plot + stratify):
  - ``contact_precision.csv``  one row per (stem, mode, predictor, range, k)
  - ``contacts_raw.parquet``   every degree>0 contact (gt + predicted) — the
                               "save all contacts" deliverable
  - ``contact_eval_meta.csv``  per-stem alignment / resolution diagnostics
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from pyconfind_contacts import compute_contacts

# --- Protenix v2 distogram bin scheme (pinned; matches exp12/score.py) -----
# From bytedance/Protenix configs/configs_base.py loss.distogram:
#   min_bin = 2.3125, max_bin = 21.6875, no_bins = 64
_DISTOGRAM_MIN_A = 2.3125
_DISTOGRAM_MAX_A = 21.6875
_DISTOGRAM_N_BINS = 64
_BIN_WIDTH = (_DISTOGRAM_MAX_A - _DISTOGRAM_MIN_A) / _DISTOGRAM_N_BINS
_BIN_CENTERS = (
    np.linspace(_DISTOGRAM_MIN_A, _DISTOGRAM_MAX_A - _BIN_WIDTH, _DISTOGRAM_N_BINS, dtype=np.float64)
    + 0.5 * _BIN_WIDTH
)

# Ground-truth contact thresholds (match contacts_v1 GenerationConfig).
MIN_CONTACT_DEGREE = 0.001
MIN_SEQ_SEP = 6

# Default distogram contact threshold: P(rep atoms within 8 Å). Tunable
# (the issue invited input) — the distogram is saved, so a threshold sweep
# is cheap.
DEFAULT_CONTACT_THRESHOLD_A = 8.0

# CASP separation classes (inclusive). "all" = the sep>=6 aggregate.
RANGES: dict[str, tuple[int, int | None]] = {
    "all": (6, None),
    "short": (6, 11),
    "medium": (12, 23),
    "long": (24, None),
}
TOP_K: tuple[int, ...] = (1, 2, 5)

MODES = ("single_seq", "msa")
PREDICTORS = ("distogram", "structure")


def contact_probs_from_distogram(probs: np.ndarray, *, threshold_a: float = DEFAULT_CONTACT_THRESHOLD_A) -> np.ndarray:
    """[N,N] cumulative contact probability = mass on bins with center <= threshold."""
    bin_mask = _BIN_CENTERS <= threshold_a
    return probs[..., bin_mask].sum(axis=-1)


def _true_matrix(L: int, contacts: list[tuple[int, int, float]]) -> np.ndarray:
    """Boolean [L,L] (upper tri) of true contacts: degree>=0.001 AND sep>=6."""
    m = np.zeros((L, L), dtype=bool)
    for i, j, d in contacts:
        if d >= MIN_CONTACT_DEGREE and (j - i) >= MIN_SEQ_SEP and i < j < L:
            m[i, j] = True
    return m


def _degree_matrix(L: int, contacts: list[tuple[int, int, float]]) -> np.ndarray:
    """[L,L] (upper tri) predicted contact degree; 0 where pyconfind emitted nothing."""
    m = np.zeros((L, L), dtype=np.float64)
    for i, j, d in contacts:
        if i < j < L:
            m[i, j] = max(m[i, j], d)
    return m


def _resolved_pairs(resolved: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-triangular (i, j, sep) over the resolved-residue index set."""
    a, b = np.triu_indices(len(resolved), k=1)
    i = resolved[a]
    j = resolved[b]
    return i, j, (j - i)


def _precision_rows(
    *,
    score: np.ndarray,        # [L,L] predictor score (higher = more likely contact)
    true_mat: np.ndarray,     # [L,L] bool, true contacts
    pair_i: np.ndarray,       # candidate pair endpoints (resolved upper-tri)
    pair_j: np.ndarray,
    pair_sep: np.ndarray,
    L: int,
) -> list[dict]:
    """precision @ {L, L/2, L/5} per range, given a predictor score matrix."""
    cand_scores_all = score[pair_i, pair_j]
    cand_gt_all = true_mat[pair_i, pair_j]
    rows: list[dict] = []
    for rng, (lo, hi) in RANGES.items():
        in_range = pair_sep >= lo
        if hi is not None:
            in_range &= pair_sep <= hi
        scores = cand_scores_all[in_range]
        gt = cand_gt_all[in_range].astype(np.int64)
        n_cand = int(scores.size)
        n_true = int(gt.sum())
        if n_cand == 0:
            for k in TOP_K:
                rows.append(dict(range=rng, k=k, precision=float("nan"),
                                 n_candidate=0, n_true=n_true, n_top=0))
            continue
        order = np.argsort(-scores, kind="mergesort")  # stable, deterministic ties
        gt_sorted = gt[order]
        for k in TOP_K:
            top_n = min(max(1, L // k), n_cand)
            precision = float(gt_sorted[:top_n].sum()) / top_n
            rows.append(dict(range=rng, k=k, precision=precision,
                             n_candidate=n_cand, n_true=n_true, n_top=top_n))
    return rows


def evaluate_protein(
    *,
    stem: str,
    input_seq: str,
    gt_cif: Path,
    gt_chain: str | None,
    best_dir: Path,
    modes: tuple[str, ...] = MODES,
    distogram_threshold_a: float = DEFAULT_CONTACT_THRESHOLD_A,
) -> tuple[list[dict], list[dict], dict]:
    """Score one protein in all (mode x predictor) configs.

    Returns ``(precision_rows, raw_contact_rows, meta)``. ``precision_rows``
    carry only the metric fields (dataset / strata are stamped by the
    caller). Missing predictions for a mode are skipped (logged in meta).
    """
    gt = compute_contacts(gt_cif, input_seq, stem=stem, prefer_chain=gt_chain)
    L = gt.n_input_residues
    resolved = np.asarray(gt.resolved_positions, dtype=np.int64)
    true_mat = _true_matrix(L, list(gt.contacts))
    pair_i, pair_j, pair_sep = _resolved_pairs(resolved)

    raw: list[dict] = [
        dict(stem=stem, role="gt", mode="na", i=i, j=j, degree=d, sep=j - i)
        for (i, j, d) in gt.contacts
    ]
    meta = dict(
        stem=stem, L=L,
        gt_chain=gt.chain,
        gt_n_resolved=gt.n_resolved_residues,
        gt_n_mapped=gt.n_mapped_residues,
        gt_align_identity=round(gt.alignment_identity, 4),
        n_true_contacts=int(true_mat.sum()),
    )

    rows: list[dict] = []
    for mode in modes:
        sub = best_dir / mode / stem
        dpath = sub / "distogram.npz"
        cpath = sub / "structure.cif"
        if not dpath.exists() or not cpath.exists():
            meta[f"{mode}_present"] = 0
            continue
        meta[f"{mode}_present"] = 1

        # --- distogram predictor ---
        probs = np.load(dpath)["probs"]
        if probs.shape[0] != L:
            # Defensive: distogram should be L x L. Truncate to the overlap
            # and record it (alignment is by canonical index, so truncation
            # at the tail is the safe restriction).
            meta[f"{mode}_distogram_L"] = int(probs.shape[0])
        cprob = contact_probs_from_distogram(probs, threshold_a=distogram_threshold_a)
        score_d = np.zeros((L, L), dtype=np.float64)
        m = min(L, cprob.shape[0])
        score_d[:m, :m] = cprob[:m, :m]
        for r in _precision_rows(score=score_d, true_mat=true_mat,
                                 pair_i=pair_i, pair_j=pair_j, pair_sep=pair_sep, L=L):
            rows.append(dict(mode=mode, predictor="distogram", **r))

        # --- structure predictor (pyconfind on the predicted CIF) ---
        pred = compute_contacts(cpath, input_seq, stem=stem, prefer_chain=None)
        meta[f"{mode}_pred_align_identity"] = round(pred.alignment_identity, 4)
        meta[f"{mode}_pred_n_contacts"] = len(pred.contacts)
        score_s = _degree_matrix(L, list(pred.contacts))
        for r in _precision_rows(score=score_s, true_mat=true_mat,
                                 pair_i=pair_i, pair_j=pair_j, pair_sep=pair_sep, L=L):
            rows.append(dict(mode=mode, predictor="structure", **r))
        raw.extend(
            dict(stem=stem, role="pred", mode=mode, i=i, j=j, degree=d, sep=j - i)
            for (i, j, d) in pred.contacts
        )

    return rows, raw, meta


# --- dataset driver --------------------------------------------------------

# Eval-manifest columns the driver requires; any *other* columns are carried
# through onto each precision row as strata (neff_tier, fold_verdict, ...).
_REQUIRED = {"dataset", "stem", "gt_cif", "input_seq"}


def evaluate_dataset(
    *,
    manifest_csv: Path,
    best_dir: Path,
    out_dir: Path,
    modes: tuple[str, ...] = MODES,
    distogram_threshold_a: float = DEFAULT_CONTACT_THRESHOLD_A,
    gt_root: Path | None = None,
    limit: int | None = None,
) -> None:
    """Score every protein in ``manifest_csv``; write the three output tables.

    ``gt_cif`` paths in the manifest are resolved relative to ``gt_root``
    (default: the manifest's directory). Strata columns (anything beyond
    the required set) ride along onto the precision rows.
    """
    manifest_csv = Path(manifest_csv)
    gt_root = Path(gt_root) if gt_root else manifest_csv.parent
    df = pd.read_csv(manifest_csv, dtype=str).fillna("")
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_csv}: manifest missing columns {sorted(missing)}")
    strata_cols = [c for c in df.columns if c not in _REQUIRED and c != "gt_chain"]
    if limit:
        df = df.iloc[:limit]

    prec_rows: list[dict] = []
    raw_rows: list[dict] = []
    meta_rows: list[dict] = []
    for n, row in enumerate(df.itertuples(index=False), start=1):
        rec = row._asdict()
        stem = rec["stem"]
        gt_cif = (gt_root / rec["gt_cif"]) if not Path(rec["gt_cif"]).is_absolute() else Path(rec["gt_cif"])
        if not gt_cif.exists():
            print(f"[{n}/{len(df)}] {stem}: missing GT {gt_cif}; skipping.")
            continue
        try:
            pr, raw, meta = evaluate_protein(
                stem=stem, input_seq=rec["input_seq"], gt_cif=gt_cif,
                gt_chain=(rec.get("gt_chain") or None), best_dir=best_dir,
                modes=modes, distogram_threshold_a=distogram_threshold_a,
            )
        except Exception as e:  # noqa: BLE001 — one bad structure shouldn't kill the run
            print(f"[{n}/{len(df)}] {stem}: ERROR {type(e).__name__}: {e}")
            continue
        strata = {c: rec[c] for c in strata_cols}
        for r in pr:
            prec_rows.append(dict(dataset=rec["dataset"], stem=stem, **strata, **r))
        for r in raw:
            raw_rows.append(dict(dataset=rec["dataset"], **r))
        meta_rows.append(dict(dataset=rec["dataset"], **strata, **meta))
        long_prec = next((r["precision"] for r in pr
                          if r["mode"] == "msa" and r["predictor"] == "structure"
                          and r["range"] == "long" and r["k"] == 1), float("nan"))
        print(f"[{n}/{len(df)}] {stem}: L={meta['L']} resolved={meta['gt_n_resolved']} "
              f"id={meta['gt_align_identity']:.3f} true={meta['n_true_contacts']} "
              f"| msa/structure P@L(long)={long_prec}")

    out_dir.mkdir(parents=True, exist_ok=True)
    prec_df = pd.DataFrame(prec_rows)
    prec_df.to_csv(out_dir / "contact_precision.csv", index=False)
    pd.DataFrame(raw_rows).to_parquet(out_dir / "contacts_raw.parquet", index=False)
    pd.DataFrame(meta_rows).to_csv(out_dir / "contact_eval_meta.csv", index=False)
    print(f"\nWrote {out_dir}/contact_precision.csv ({len(prec_df)} rows), "
          f"contacts_raw.parquet ({len(raw_rows)} rows), "
          f"contact_eval_meta.csv ({len(meta_rows)} rows).")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("contact-eval", help="Score Protenix contacts vs pyconfind GT (4 configs).")
    p.add_argument("--manifest", type=Path, required=True, help="Eval manifest CSV (dataset, stem, gt_cif, input_seq, gt_chain, +strata).")
    p.add_argument("--best", type=Path, required=True, help="best/ tree: {mode}/{stem}/{structure.cif,distogram.npz}.")
    p.add_argument("--out", type=Path, required=True, help="Output dir for the three tables.")
    p.add_argument("--gt-root", type=Path, default=None, help="Root for relative gt_cif paths (default: manifest dir).")
    p.add_argument("--modes", default="single_seq,msa", help="Comma-separated modes.")
    p.add_argument("--distogram-threshold", type=float, default=DEFAULT_CONTACT_THRESHOLD_A, help="Å threshold for distogram contact prob.")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N proteins.")
    p.set_defaults(func=lambda args: evaluate_dataset(
        manifest_csv=args.manifest, best_dir=args.best, out_dir=args.out,
        modes=tuple(args.modes.split(",")), distogram_threshold_a=args.distogram_threshold,
        gt_root=args.gt_root, limit=args.limit,
    ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_subparser(sub)
    args = ap.parse_args()
    args.func(args)

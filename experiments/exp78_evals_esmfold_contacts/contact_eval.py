# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score ESMFold / ESMFold2 contact prediction against pyconfind ground truth.

Issue #78 — extends exp74's eval to two more *structure* predictors,
ESMFold (``facebook/esmfold_v1``) and ESMFold2 (``biohub/ESMFold2``), on
the **same 554-protein eval set** (FoldBench-100 + the exp65 low-MSA /
novel-fold set), with the **same pyconfind ground truth and metric** that
exp74 used for Protenix v2.

Ground truth = pyconfind side-chain contacts on the experimental
structure (degree >= 0.001, primary-sequence separation >= 6; see
``pyconfind_contacts.py``, copied verbatim from exp74). Both ESM models
are **single-sequence structure predictors**, so each is scored with the
"structure" predictor only: run pyconfind on the *predicted* structure
(``native_only=True``, same knobs as the ground truth) and rank candidate
pairs by predicted contact degree. There is no distogram config here
(the issue asks to score from the predicted structures).

This module is deliberately **model-agnostic**: it scores any predicted
structure under ``{pred_root}/{model}/{stem}/structure.cif`` and stamps a
``model`` column on every row, so the output concatenates cleanly with
exp74's Protenix ``contact_precision_all.csv`` (see ``combine_scores.py``).

Metric: **contacts @ L** (precision among the top-L predicted pairs,
L = sequence length), also @ L/2, L/5, and **R-precision** (precision@R
where R = the bin's ground-truth contact count, so the ceiling is 1.0 for
every protein), reported **in aggregate** (separation >= 6) and **split by
range** short [6,11] / medium [12,23] / long [>=24]. The candidate-pair
universe is restricted to residues resolved in the ground-truth structure,
identically across all models, so the numbers are comparable to exp74's.

Outputs (tidy, long-form, easy to plot + stratify):
  - ``contact_precision.csv``  one row per (stem, model, predictor, range, k)
  - ``contacts_raw.parquet``   every degree>0 contact (gt + predicted) — the
                               "save all contacts" deliverable
  - ``contact_eval_meta.csv``  per-stem alignment / resolution diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pyconfind_contacts import compute_contacts

# Ground-truth contact thresholds (match contacts_v1 GenerationConfig;
# identical to exp74 so the two experiments' numbers are comparable).
MIN_CONTACT_DEGREE = 0.001
MIN_SEQ_SEP = 6

# CASP separation classes (inclusive). "all" = the sep>=6 aggregate.
RANGES: dict[str, tuple[int, int | None]] = {
    "all": (6, None),
    "short": (6, 11),
    "medium": (12, 23),
    "long": (24, None),
}

# Prediction-count cuts for precision@<cut>. L/L2/L5 are the CASP top-L/k
# cuts. "R" is R-precision (cutoff R = the ground-truth contact count for
# the bin, so a perfect ranker scores 1.0 for every protein regardless of
# contact density — the density-robust complement to precision@L).
CUTS: tuple[tuple[str, object], ...] = (
    ("L", lambda L, c: L),
    ("L/2", lambda L, c: max(1, L // 2)),
    ("L/5", lambda L, c: max(1, L // 5)),
    ("R", lambda L, c: c),
)

# Default models scored here. Both are single-sequence; predictor is always
# "structure" (pyconfind on the predicted CIF).
MODELS = ("esmfold", "esmfold2")


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
    """precision @ {L, L/2, L/5, R} per range, given a predictor score matrix.

    "R" = R-precision (precision@R): cutoff R = the bin's ground-truth contact
    count, so the ceiling is 1.0 for every protein regardless of contact
    density (at this cutoff precision == recall). NaN when the bin has no
    candidate pairs or no true contacts.
    """
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
        order = np.argsort(-scores, kind="mergesort") if n_cand else None  # stable ties
        gt_sorted = gt[order] if n_cand else None
        for cut, target_fn in CUTS:
            target = int(target_fn(L, n_true))
            if n_cand == 0 or target <= 0:
                rows.append(dict(range=rng, cut=cut, precision=float("nan"),
                                 n_candidate=n_cand, n_true=n_true, n_top=0))
                continue
            top_n = min(target, n_cand)
            precision = float(gt_sorted[:top_n].sum()) / top_n
            rows.append(dict(range=rng, cut=cut, precision=precision,
                             n_candidate=n_cand, n_true=n_true, n_top=top_n))
    return rows


def evaluate_protein(
    *,
    stem: str,
    input_seq: str,
    gt_cif: Path,
    gt_chain: str | None,
    pred_root: Path,
    models: tuple[str, ...] = MODELS,
) -> tuple[list[dict], list[dict], dict]:
    """Score one protein for every model whose predicted structure is present.

    Predicted structures are read from
    ``{pred_root}/{model}/{stem}/structure.cif``. Returns
    ``(precision_rows, raw_contact_rows, meta)``; ``precision_rows`` carry
    only the metric fields plus ``model`` / ``predictor`` (dataset + strata
    are stamped by the caller). Missing predictions for a model are skipped
    (logged in meta).
    """
    gt = compute_contacts(gt_cif, input_seq, stem=stem, prefer_chain=gt_chain)
    L = gt.n_input_residues
    resolved = np.asarray(gt.resolved_positions, dtype=np.int64)
    true_mat = _true_matrix(L, list(gt.contacts))
    pair_i, pair_j, pair_sep = _resolved_pairs(resolved)

    raw: list[dict] = [
        dict(stem=stem, role="gt", model="na", i=i, j=j, degree=d, sep=j - i)
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
    for model in models:
        cpath = pred_root / model / stem / "structure.cif"
        if not cpath.exists():
            meta[f"{model}_present"] = 0
            continue
        meta[f"{model}_present"] = 1

        # structure predictor: pyconfind on the predicted CIF, rank by degree.
        pred = compute_contacts(cpath, input_seq, stem=stem, prefer_chain=None)
        meta[f"{model}_pred_align_identity"] = round(pred.alignment_identity, 4)
        meta[f"{model}_pred_n_contacts"] = len(pred.contacts)
        score_s = _degree_matrix(L, list(pred.contacts))
        for r in _precision_rows(score=score_s, true_mat=true_mat,
                                 pair_i=pair_i, pair_j=pair_j, pair_sep=pair_sep, L=L):
            rows.append(dict(model=model, mode="single_seq", predictor="structure", **r))
        raw.extend(
            dict(stem=stem, role="pred", model=model, i=i, j=j, degree=d, sep=j - i)
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
    pred_root: Path,
    out_dir: Path,
    models: tuple[str, ...] = MODELS,
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
                gt_chain=(rec.get("gt_chain") or None), pred_root=pred_root,
                models=models,
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
        # Progress line: report ESMFold2 long-range R-precision when present.
        long_r = next((r["precision"] for r in pr
                       if r["model"] == "esmfold2" and r["range"] == "long" and r["cut"] == "R"),
                      float("nan"))
        present = ",".join(m for m in models if meta.get(f"{m}_present"))
        print(f"[{n}/{len(df)}] {stem}: L={meta['L']} resolved={meta['gt_n_resolved']} "
              f"id={meta['gt_align_identity']:.3f} true={meta['n_true_contacts']} "
              f"present=[{present}] | esmfold2 R-prec(long)={long_r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    prec_df = pd.DataFrame(prec_rows)
    prec_df.to_csv(out_dir / "contact_precision.csv", index=False)
    pd.DataFrame(raw_rows).to_parquet(out_dir / "contacts_raw.parquet", index=False)
    pd.DataFrame(meta_rows).to_csv(out_dir / "contact_eval_meta.csv", index=False)
    print(f"\nWrote {out_dir}/contact_precision.csv ({len(prec_df)} rows), "
          f"contacts_raw.parquet ({len(raw_rows)} rows), "
          f"contact_eval_meta.csv ({len(meta_rows)} rows).")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("contact-eval", help="Score ESM contacts vs pyconfind GT (structure predictor).")
    p.add_argument("--manifest", type=Path, required=True, help="Eval manifest CSV (dataset, stem, gt_cif, input_seq, gt_chain, +strata).")
    p.add_argument("--pred-root", type=Path, required=True, help="Predicted structures: {model}/{stem}/structure.cif.")
    p.add_argument("--out", type=Path, required=True, help="Output dir for the three tables.")
    p.add_argument("--gt-root", type=Path, default=None, help="Root for relative gt_cif paths (default: manifest dir).")
    p.add_argument("--models", default=",".join(MODELS), help="Comma-separated models to score.")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N proteins.")
    p.set_defaults(func=lambda args: evaluate_dataset(
        manifest_csv=args.manifest, pred_root=args.pred_root, out_dir=args.out,
        models=tuple(args.models.split(",")), gt_root=args.gt_root, limit=args.limit,
    ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_subparser(sub)
    args = ap.parse_args()
    args.func(args)

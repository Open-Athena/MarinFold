# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Idea 5: post-process distogram sharpening sweep.

Loads the on-disk distogram for each protein in ``data/train_proteins.csv``,
applies a temperature sharpen ``probs' = softmax(log(probs + eps) / T)``
to non-empty pair rows, scores the sharpened distogram against the GT,
and reports mean / median ``lddt_distogram_cb`` for each T in the sweep.

Inputs:
  outputs/<stem>/distogram.npz — whatever was last written (here:
    gt_filtered_naive); the LDDT-shell pair rows have the same
    probabilities as a baseline_naive distogram so sharpening + scoring
    gives the same hard LDDT either way.

Outputs:
  outputs/<stem>/distogram_sharp_T{T}.npz — temp per (stem, T) pairs
    (gitignored; can be deleted after we pick the best T).
  data/sharpen_sweep.csv — one row per (stem, T) with the LDDT.
  Best T's row appended to data/experiments.tsv.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_EXP1 = _THIS.parent / "exp1_document_structures_contacts_and_distances_v1"
if str(_EXP1) not in sys.path:
    sys.path.insert(0, str(_EXP1))

from score_marinfold import MARINFOLD_BINS, score_one  # noqa: E402
from append_experiment_row import upsert_experiment_row  # noqa: E402

EPS = 1e-12


def sharpen(probs: np.ndarray, T: float) -> np.ndarray:
    """Sharpen pair probability rows by temperature scaling.

    For pairs with ``probs.sum(axis=-1) > 0``: ``p' = softmax(log(p+ε)/T)``.
    For pairs with all-zero rows (skipped during inference): leave as zero.
    """
    out = np.zeros_like(probs)
    row_sums = probs.sum(axis=-1)
    nonzero = row_sums > 0
    if not nonzero.any():
        return out
    p = probs[nonzero]                       # [K, n_bins]
    logp = np.log(p + EPS)
    z = logp / T
    z = z - z.max(axis=-1, keepdims=True)
    expz = np.exp(z)
    sharp = expz / expz.sum(axis=-1, keepdims=True)
    out[nonzero] = sharp.astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv", type=Path,
        default=_THIS / "data" / "train_proteins.csv",
    )
    parser.add_argument(
        "--protenix-dir", type=Path,
        default=_THIS / "protenix_data" / "data" / "protenix-foldbench-monomers",
    )
    parser.add_argument("--out", type=Path, default=_THIS / "outputs")
    parser.add_argument(
        "--sweep-csv", type=Path,
        default=_THIS / "data" / "sharpen_sweep.csv",
    )
    parser.add_argument(
        "--experiments-tsv", type=Path,
        default=_THIS / "data" / "experiments.tsv",
    )
    parser.add_argument(
        "--temps", nargs="+", type=float,
        default=[1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05],
    )
    parser.add_argument(
        "--base-algorithm", default="gt_filtered_naive",
        help="The on-disk distograms' algorithm name (for provenance).",
    )
    args = parser.parse_args()

    train_rows = list(csv.DictReader(args.train_csv.open()))
    n_proteins = len(train_rows)

    t_start = time.time()
    per_t: dict[float, list[dict]] = {T: [] for T in args.temps}

    for T in args.temps:
        for entry in train_rows:
            stem = entry["stem"]
            base_npz = args.out / stem / "distogram.npz"
            sharp_npz = args.out / stem / f"distogram_sharp_T{T}.npz"
            d = np.load(base_npz)
            sharp = sharpen(d["probs"], T)
            np.savez_compressed(sharp_npz, probs=sharp)
            row = score_one(
                distogram_npz=sharp_npz,
                gt_cif=args.protenix_dir / "gt" / f"{stem}.cif",
                pdb_id=entry["pdb_id"],
                chain_id=entry["chain_id"],
                method=f"sharpen_T{T}",
                bins=MARINFOLD_BINS,
            )
            row["T"] = T
            row["stem"] = stem
            per_t[T].append(row)

    elapsed = time.time() - t_start

    # Write the sweep CSV.
    args.sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.sweep_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "T", "stem", "lddt_distogram_cb", "lddt_distogram_cb_soft",
        ])
        for T in args.temps:
            for row in per_t[T]:
                writer.writerow([
                    T, row["stem"],
                    f"{row['lddt_distogram_cb']:.6f}",
                    f"{row['lddt_distogram_cb_soft']:.6f}",
                ])

    # Aggregate + report.
    print(f"\n{'T':>5}  {'mean_lddt':>10}  {'median_lddt':>12}")
    best_T = None
    best_mean = -1.0
    summary: dict[float, tuple[float, float]] = {}
    for T in args.temps:
        lddts = sorted(r["lddt_distogram_cb"] for r in per_t[T]
                       if r["lddt_distogram_cb"] == r["lddt_distogram_cb"])
        mean = sum(lddts) / len(lddts)
        median = lddts[len(lddts) // 2]
        summary[T] = (mean, median)
        marker = "  *" if mean > best_mean else ""
        print(f"{T:>5}  {mean:>10.4f}  {median:>12.4f}{marker}")
        if mean > best_mean:
            best_mean = mean
            best_T = T

    print(f"\nBest T = {best_T} (mean LDDT {best_mean:.4f})")
    print(f"Sweep wall: {elapsed:.1f} s (no GPU; just numpy + scoring)")

    # Append best T to experiments.tsv. Wall is conservatively the
    # underlying gt_filtered_naive wall + the sharpening overhead.
    # The post-process itself is essentially free.
    best_mean_final, best_median_final = summary[best_T]
    upsert_experiment_row(
        experiment_id=f"sharpen_T{best_T}_on_{args.base_algorithm}",
        description=(
            f"Post-process temperature sharpening (T={best_T}) of the "
            f"{args.base_algorithm} distogram. probs' = softmax(log(p+eps)/T) "
            f"applied to non-empty pair rows; empty rows left as zero. "
            f"No new inference."
        ),
        mean_lddt=best_mean_final,
        median_lddt=best_median_final,
        total_wall_seconds=215.3 + elapsed,  # gt_filtered base + sharpen
        tsv_path=args.experiments_tsv,
    )

    print(f"\nWrote {args.sweep_csv}")


if __name__ == "__main__":
    main()

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only: seed the model with GROUND-TRUTH contacts and read.

NOT a candidate algorithm — uses GT structure to pick seeds, which
violates the issue's no-cheating rule. Purpose: establish the ceiling
of the seeded-prefix family. If even *perfect* contact seeds can't
push mean LDDT past some value, then any honest seeded variant is
bounded below that.

Algorithm: from GT CB-CB distance matrix, take every pair with
``gt_d < 8 Å`` and `|i-j| >= 6` as a true contact; bucket by CASP
separation (short/medium/long); use the full set as seeds in
``predict_distogram_seeded``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_EXP1 = _THIS.parent / "exp1_document_structures_contacts_and_distances_v1"
if str(_EXP1) not in sys.path:
    sys.path.insert(0, str(_EXP1))

from naive_inference import (  # noqa: E402
    BIN_MIDPOINTS, DISTANCE_MAX_A, NUM_DISTANCE_BINS,
    Runtime, load_runtime,
)
from canonical_sequence import read_canonical_sequence  # noqa: E402
from gt_filtered_inference import build_gt_shell_mask  # noqa: E402
from score_marinfold import _read_gt_rep_coords, _pairwise_distance_matrix  # noqa: E402
from seeded_contacts_inference import predict_distogram_seeded, _RANGE_BUCKETS  # noqa: E402

_CONTACT_CUTOFF_A = 8.0


def gt_contacts(gt_cif: Path, n_residues_seq: int) -> list[tuple[int, int, str]]:
    n_gt, rep = _read_gt_rep_coords(gt_cif)
    if n_gt != n_residues_seq:
        raise ValueError(f"len mismatch for {gt_cif}: gt={n_gt}, seq={n_residues_seq}")
    d, mask = _pairwise_distance_matrix(rep)
    n = n_gt
    out: list[tuple[int, int, str]] = []
    for name, sep_lo, sep_hi, token in _RANGE_BUCKETS:
        for i in range(n):
            j_start = i + sep_lo
            j_end = n if sep_hi is None else min(n, i + sep_hi + 1)
            for j in range(j_start, j_end):
                if not mask[i, j]:
                    continue
                if d[i, j] >= _CONTACT_CUTOFF_A:
                    continue
                out.append((i + 1, j + 1, token))
    return out


def predict_one(
    *,
    rt: Runtime,
    stem: str,
    protenix_dir: Path,
    out_dir: Path,
    batch_size: int = 128,
    algorithm: str = "gt_oracle_seeded",
    max_seeds: int | None = None,
) -> float:
    gt_cif = protenix_dir / "gt" / f"{stem}.cif"
    seq = read_canonical_sequence(gt_cif)
    pair_mask = build_gt_shell_mask(gt_cif, seq.n_residues)
    seeds = gt_contacts(gt_cif, seq.n_residues)
    if max_seeds is not None:
        seeds = seeds[:max_seeds]

    t_start = time.time()
    probs, n_pairs_queried, prefix_token_count = predict_distogram_seeded(
        rt=rt, residue_names=seq.residue_names, pair_mask=pair_mask,
        seeds=seeds, batch_size=batch_size,
    )
    elapsed = time.time() - t_start

    out_path = out_dir / stem / "distogram.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, probs=probs)
    (out_path.parent / "provenance.json").write_text(json.dumps({
        "stem": stem,
        "n_residues": seq.n_residues,
        "n_pairs": seq.n_residues * (seq.n_residues - 1) // 2,
        "n_pairs_queried": n_pairs_queried,
        "n_gt_contacts": len(seeds),
        "max_seeds_cap": max_seeds,
        "prefix_token_count": prefix_token_count,
        "algorithm": algorithm,
        "model_nickname": rt.model_nickname,
        "model_path": rt.model_path,
        "atom_convention": "CB-CB (CA for GLY/UNK)",
        "bin_scheme": {
            "min_A": 0.0, "max_A": DISTANCE_MAX_A,
            "n_bins": NUM_DISTANCE_BINS, "midpoints_A": BIN_MIDPOINTS.tolist(),
        },
        "elapsed_seconds": round(elapsed, 3),
        "model_load_seconds": round(rt.model_load_seconds, 3),
        "batch_size": batch_size,
        "hardware": rt.hardware,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
    return elapsed

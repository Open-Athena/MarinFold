# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1, step 1 — cache MarinFold's amino-acid conditionals for every assay.

This is the only step that touches a GPU. For each scorable ProteinGym target
sequence it runs ``K`` orderings through the contacts-v1 any-order readout and
writes the resulting ``(K, L, 20)`` log-probabilities to an ``.npz``. Every
scoring rule in :mod:`analyze` is then a cheap re-read of these files — the
ordering-count sweep, the context-fraction sweep, and any rule invented later
all come from one pass over the model.

Caching rather than scoring inline is deliberate. The tensors *are* the
experiment's reusable artifact: they let anyone re-derive a variant-effect
score, or a different one, without a GPU and without this code.

One file per assay, resumable: an existing ``.npz`` with the right ``K`` is
skipped, so an interrupted run continues where it stopped.

Usage::

    uv run python cache_conditionals.py --orderings 200
    uv run python cache_conditionals.py --orderings 200 --limit 4   # smoke test
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import proteingym
from marinfold.document_structures.contacts_v1 import sequence_likelihood as sl
from marinfold.document_structures.contacts_v1.parse import residues_from_sequence
from marinfold.inference.core import load_backend

HERE = Path(__file__).resolve().parent


def cache_path(out_dir: Path, dms_id: str) -> Path:
    return out_dir / f"{dms_id}.npz"


def already_done(path: Path, orderings: int) -> bool:
    """True if ``path`` holds at least ``orderings`` orderings.

    More than asked for is fine — the scorer slices. Fewer is not, so the
    assay is recomputed.
    """
    if not path.exists():
        return False
    with np.load(path) as data:
        return int(data["logprobs"].shape[0]) >= orderings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orderings",
        type=int,
        default=200,
        help="K. A residue lands in the top decile of context about K/10 times.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default=None, help="MODELS.yaml nickname")
    parser.add_argument("--backend", default="transformers")
    parser.add_argument(
        "--device",
        default=None,
        help="torch device; default picks CUDA when free. Use cpu to share the box.",
    )
    parser.add_argument("--limit", type=int, default=None, help="first N assays only")
    parser.add_argument(
        "--out-dir", type=Path, default=HERE / "data" / "conditionals"
    )
    args = parser.parse_args()

    reference = proteingym.reference()
    scorable = reference[reference.scorable].reset_index(drop=True)
    if args.limit is not None:
        scorable = scorable.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    skipped = reference[~reference.scorable]
    print(
        f"{len(scorable)} scorable assays, K={args.orderings}; "
        f"{len(skipped)} skipped ({', '.join(skipped.DMS_id)})"
    )

    backend = load_backend(
        args.backend,
        model=args.model,
        **({"device": args.device} if args.device else {}),
    )
    timings = []
    for index, row in scorable.iterrows():
        path = cache_path(args.out_dir, row.DMS_id)
        if already_done(path, args.orderings):
            continue
        started = time.time()
        conditionals = sl.amino_acid_conditionals(
            backend,
            residues_from_sequence(row.target_seq),
            # Seeding on the assay id makes the orderings reproducible and
            # assay-specific; two assays on the same protein get their own.
            entry_id=row.DMS_id,
            num_orderings=args.orderings,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - started
        np.savez_compressed(
            path,
            logprobs=conditionals.logprobs,
            context_sizes=conditionals.context_sizes,
            target_mass=conditionals.target_mass,
            target_seq=np.array(row.target_seq),
        )
        timings.append(
            {
                "stem": row.DMS_id,
                "n_residues": int(row.seq_len),
                "orderings": args.orderings,
                "seconds": round(elapsed, 3),
                "tokens": int(args.orderings * (2 * row.seq_len + 7)),
            }
        )
        print(
            f"[{index + 1:3d}/{len(scorable)}] {row.DMS_id:<45s} "
            f"L={row.seq_len:4d} {elapsed:7.1f}s "
            f"mass={conditionals.target_mass.mean():.4f}"
        )

    if timings:
        frame = pd.DataFrame(timings)
        path = HERE / "data" / "timings.csv"
        if path.exists():
            frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
        frame.to_csv(path, index=False)
        print(f"\n{len(timings)} assays cached; timings -> {path}")
        print(f"total GPU time: {frame.seconds.sum() / 60:.1f} min")


if __name__ == "__main__":
    main()

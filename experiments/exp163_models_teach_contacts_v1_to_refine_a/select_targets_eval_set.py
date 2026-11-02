# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""exp163 Phase 3 target selection: the **exp89 held-out eval set** (issue #163).

Phase 2 trained on the 10k ESM-Atlas refinement corpus, which has **no held-out
protein split** — all 9,375 rollout proteins went into training. So the headline
comparison (refiner@K vs refiner@K0 vs base matrix vs consensus) has to be
measured on the exp89 eval set, which needs its own base-model rollouts.

This script emits that eval set in the exact target schema the rollout worker and
``build_refinement_corpus.py`` consume::

    entry_id, L, sequence, n_gt, gt_contacts, global_plddt, ptm

Sources (both already in the repo / on the public bucket):

* **sequences** — the exp74/exp78 eval manifests
  (``eval_manifest_foldbench.csv`` + ``eval_manifest_exp65.csv``: 100 + 454 = 554
  rows, columns ``dataset, stem, gt_cif, gt_chain, input_seq, n_residues``);
* **ground truth** — exp89's ``gt_universe.jsonl`` (554 records, one per manifest
  row), whose ``contacts`` are ``[i, j, degree]`` triples in **0-based
  input-sequence coordinates**. We apply exp89's own definition of a true contact,
  ``degree >= 0.001 and (j - i) >= 6``, so the GT here is bit-identical to what
  ``compute_metrics.py`` scores against.

``entry_id`` is ``"{dataset}__{stem}"``: the two sources share 2 stems across
datasets (554 units / 552 distinct stems), and the worker writes one
``prompts/<entry_id>.parquet`` per target, so the id must be unique AND
path-safe.

``global_plddt`` / ``ptm`` are carried as NaN — they exist only to keep the
schema identical to the ESM-Atlas targets; nothing downstream filters on them
here (these are experimental structures, not predictions).

    uv run python select_targets_eval_set.py \\
        --gt-universe gt_universe.jsonl --out targets_eval.parquet

Fetch the GT universe (public bucket, anonymous) with::

    uv run --with 'huggingface_hub>=1.5' python -c "
    from huggingface_hub import HfFileSystem
    HfFileSystem(token=False).get(
      'buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp89/gt_universe.jsonl',
      'gt_universe.jsonl')"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXP78_DATA = REPO / "experiments/exp78_evals_esmfold_contacts/data"

# exp89's contact definition (compute_metrics.MIN_DEG / MIN_SEP), applied here so
# the target GT matches the metric's GT exactly.
MIN_DEG = 0.001
MIN_SEP = 6


def load_manifests(manifest_dir: Path) -> pd.DataFrame:
    """[(dataset, stem, input_seq)] over both eval manifests.

    Only these three columns are shared: foldbench carries ``n_residues`` while
    exp65 calls the same thing ``length``. We take the length from the sequence
    itself and cross-check it against the GT record, so neither is needed.
    """
    frames = []
    for name in ("eval_manifest_foldbench.csv", "eval_manifest_exp65.csv"):
        path = manifest_dir / name
        if not path.exists():
            raise SystemExit(f"missing manifest {path}")
        frames.append(pd.read_csv(path)[["dataset", "stem", "input_seq"]])
    df = pd.concat(frames, ignore_index=True)
    if df.duplicated(["dataset", "stem"]).any():
        raise SystemExit("manifest has duplicate (dataset, stem) keys")
    return df


def load_gt(path: Path) -> dict[tuple[str, str], dict]:
    """gt_universe.jsonl keyed by (dataset, stem)."""
    out: dict[tuple[str, str], dict] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[(rec["dataset"], rec["stem"])] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-universe", required=True, type=Path)
    ap.add_argument("--manifest-dir", type=Path, default=EXP78_DATA)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=None, help="drop targets longer than this")
    a = ap.parse_args()

    man = load_manifests(a.manifest_dir)
    gt = load_gt(a.gt_universe)
    print(f"[exp163] {len(man)} manifest rows, {len(gt)} GT records", flush=True)

    rows, skipped = [], []
    for r in man.itertuples(index=False):
        key = (r.dataset, r.stem)
        rec = gt.get(key)
        if rec is None:
            skipped.append((key, "no GT record"))
            continue
        seq = str(r.input_seq)
        L = len(seq)
        if int(rec["L"]) != L:
            skipped.append((key, f"L mismatch manifest={L} gt={rec['L']}"))
            continue
        if a.max_len and L > a.max_len:
            skipped.append((key, f"L={L} > max_len"))
            continue
        pairs = sorted(
            {
                (int(i), int(j))
                for i, j, d in rec["contacts"]
                if float(d) >= MIN_DEG and (int(j) - int(i)) >= MIN_SEP and int(i) < int(j) < L
            }
        )
        if not pairs:
            skipped.append((key, "no GT contacts after filter"))
            continue
        rows.append(
            dict(
                entry_id=f"{r.dataset}__{r.stem}",
                L=L,
                sequence=seq,
                n_gt=len(pairs),
                gt_contacts=[list(p) for p in pairs],
                global_plddt=np.nan,
                ptm=np.nan,
            )
        )

    df = pd.DataFrame(rows)
    df.to_parquet(a.out, index=False)
    print(
        f"[exp163] wrote {len(df)} targets -> {a.out}\n"
        f"         L: min={df.L.min()} median={int(df.L.median())} max={df.L.max()}\n"
        f"         n_gt: min={df.n_gt.min()} median={int(df.n_gt.median())} max={df.n_gt.max()}"
        f" (total {int(df.n_gt.sum()):,})",
        flush=True,
    )
    if skipped:
        print(f"         skipped {len(skipped)}: {skipped[:5]}{' ...' if len(skipped) > 5 else ''}", flush=True)


if __name__ == "__main__":
    main()

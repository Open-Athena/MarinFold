# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step A — extract the ground-truth contact universe for the eval set.

For each protein in the exp74/exp78 eval manifests (FoldBench-100 + exp65), run
pyconfind on the GT structure exactly as exp74/exp78 did and record, in
input-sequence coordinates:

* ``L``               — number of input-sequence residues
* ``resolved``        — input-seq indices resolved in the GT structure
                        (**the candidate-pair universe**, identical across all
                        predictors)
* ``contacts``        — every degree>0 pyconfind contact ``(i, j, degree)``

This decouples the (pyconfind, structures) GT computation from model scoring so
the two run in their own venvs. The output ``gt_universe.jsonl`` feeds the
metric step (MarinFold precision/AUC) and lets us compute AUC for the existing
predictors over the *same* resolved universe.

Run in an env with ``pyconfind`` (e.g. the exp78 venv)::

    /home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts/.venv/bin/python \
        prepare_gt_universe.py --out data/gt_universe.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pyconfind_contacts import compute_contacts

# Defaults point at the exp78 checkout, which has the manifests + staged GT
# structures (FoldBench GT under _scratch/gt_foldbench, exp65 GT under the exp65
# experiment dir).
EXP78 = Path("/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts")
EXP65 = Path("/home/bizon/git/MarinFold-exp78/experiments/exp65_evals_low_msa_depth_proteins")


def iter_manifest(manifest: Path, gt_root: Path):
    df = pd.read_csv(manifest)
    strata_cols = [c for c in df.columns
                   if c not in {"gt_cif", "input_seq", "n_residues"}]
    for _, rec in df.iterrows():
        gt_cif = gt_root / rec["gt_cif"]
        strata = {c: (None if pd.isna(rec[c]) else rec[c]) for c in strata_cols}
        yield rec["stem"], rec["input_seq"], gt_cif, rec.get("gt_chain"), strata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--foldbench-manifest", type=Path, default=EXP78 / "data/eval_manifest_foldbench.csv")
    ap.add_argument("--exp65-manifest", type=Path, default=EXP78 / "data/eval_manifest_exp65.csv")
    ap.add_argument("--foldbench-gt-root", type=Path, default=EXP78 / "_scratch/gt_foldbench")
    ap.add_argument("--exp65-gt-root", type=Path, default=EXP65)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sources = [
        (args.foldbench_manifest, args.foldbench_gt_root),
        (args.exp65_manifest, args.exp65_gt_root),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    with args.out.open("w") as fh:
        for manifest, gt_root in sources:
            for stem, input_seq, gt_cif, gt_chain, strata in iter_manifest(manifest, gt_root):
                if not gt_cif.exists():
                    print(f"  {stem}: MISSING GT {gt_cif}", file=sys.stderr)
                    n_fail += 1
                    continue
                try:
                    chain = None if (gt_chain is None or pd.isna(gt_chain)) else gt_chain
                    gt = compute_contacts(gt_cif, input_seq, stem=stem, prefer_chain=chain)
                except Exception as e:  # noqa: BLE001
                    print(f"  {stem}: compute_contacts FAILED: {e!r}", file=sys.stderr)
                    n_fail += 1
                    continue
                rec = dict(
                    dataset=strata.get("dataset"),
                    stem=stem,
                    L=int(gt.n_input_residues),
                    n_resolved=int(gt.n_resolved_residues),
                    gt_chain=gt.chain,
                    gt_align_identity=round(float(gt.alignment_identity), 4),
                    resolved=[int(p) for p in gt.resolved_positions],
                    contacts=[[int(i), int(j), float(d)] for (i, j, d) in gt.contacts],
                    strata={k: v for k, v in strata.items() if k != "dataset"},
                )
                fh.write(json.dumps(rec) + "\n")
                n_ok += 1
                if n_ok % 50 == 0:
                    print(f"  ...{n_ok} proteins done", flush=True)
    print(f"[gt] wrote {n_ok} proteins to {args.out} ({n_fail} failed/missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the FoldBench *dev* subset for the inference-algorithm search.

We pick the winning inference algorithm (pairwise / rollout / iterative) on a
small **dev** set so we never hill-climb the held-out contacts-v1 ``test`` split.
The dev pool is **FoldBench-100** (the cleanest named slice of the #74/#78 eval
set), restricted to a tractable mid-size band (``100 <= L <= 250``, so rollout's
100 samples/protein stay fast and every protein still has plenty of long-range
GT); a **fixed seed-0 sample of 16** is drawn. Everything else — the other 84
FoldBench proteins, the 454 denovo/CASP/CAMEO eval proteins, and the entire
contacts-v1 test split — is held out for a single final confirmation.

Sources (read once; the *output* is committed so the eval is self-contained and
doesn't depend on these external checkouts at run time):

* GT universe — pyconfind side-chain contacts (degree>0, sep>=6) over the
  resolved residues, in input-sequence coordinates — from exp89's
  ``prepare_gt_universe`` (``gt_universe.jsonl``), the canonical eval-set GT.
* Input sequences — the exp78 FoldBench manifest.

Output:
* ``data/foldbench_dev.jsonl`` — one protein/line: ``stem, L, n_resolved,
  resolved, contacts (i,j), input_seq``.
* ``data/foldbench_dev_proteins.csv`` — the human-readable selection summary.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Canonical eval-set GT (exp89) + FoldBench sequences (exp78). Read once.
GT_UNIVERSE = Path(
    "/home/bizon/git/MarinFold/.claude/worktrees/vibrant-hermann-12cd27/"
    "experiments/exp89_evals_contacts_v1_model_on_eval_set/data/gt_universe.jsonl"
)
MANIFEST = Path(
    "/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts/"
    "data/eval_manifest_foldbench.csv"
)
DATASET = "foldbench100"
LMIN, LMAX, N_DEV, SEED = 100, 250, 16, 0
MIN_SEP = 6  # contacts-v1 min_seq_separation; align the GT with the candidate universe.


def contact_ij(c):
    """Pull (i, j) out of a gt_universe contact (stored as [i, j, degree])."""
    return int(c[0]), int(c[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-universe", type=Path, default=GT_UNIVERSE)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    args = ap.parse_args()

    # 1. FoldBench-100 GT, keyed by stem.
    fb: dict[str, dict] = {}
    for line in args.gt_universe.open():
        d = json.loads(line)
        if d["dataset"] != DATASET:
            continue
        fb[d["stem"]] = d

    # 2. Sequences from the manifest.
    seqs = {r["stem"]: r["input_seq"] for r in csv.DictReader(args.manifest.open())}

    # 3. Deterministic selection: mid-size band, fixed-seed sample.
    pool = sorted(s for s, d in fb.items() if LMIN <= d["L"] <= LMAX)
    rng = np.random.default_rng(SEED)
    dev = sorted(rng.choice(pool, size=N_DEV, replace=False).tolist())
    print(f"FoldBench-100 -> band [{LMIN},{LMAX}] = {len(pool)} -> seed-{SEED} sample of {len(dev)}")

    # 4. Emit the self-contained dev set + summary.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.out_dir / "foldbench_dev.jsonl"
    summ = args.out_dir / "foldbench_dev_proteins.csv"
    with jsonl.open("w") as jf, summ.open("w", newline="") as sf:
        w = csv.writer(sf)
        w.writerow(["stem", "L", "n_resolved", "n_contacts", "n_long_range", "n_seq"])
        for stem in dev:
            d = fb[stem]
            seq = seqs[stem]
            assert len(seq) == d["L"], f"{stem}: seq len {len(seq)} != L {d['L']}"
            contacts = sorted({contact_ij(c) for c in d["contacts"]
                               if abs(contact_ij(c)[0] - contact_ij(c)[1]) >= MIN_SEP})
            n_long = sum(1 for (i, j) in contacts if abs(i - j) >= 24)
            jf.write(json.dumps({
                "stem": stem, "L": d["L"], "n_resolved": d["n_resolved"],
                "resolved": sorted(d["resolved"]), "contacts": contacts, "input_seq": seq,
            }) + "\n")
            w.writerow([stem, d["L"], d["n_resolved"], len(contacts), n_long, len(seq)])
    print(f"wrote {jsonl} and {summ} ({len(dev)} proteins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

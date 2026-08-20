# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 7 — the input files every predictor needs for the 23 net-new proteins.

eval2's other 284 proteins already have scores from every predictor. The 23 do
not, so each has to be run — and each driver wants the inputs in its own shape:

* ``eval2_new_targets.parquet`` — ``dataset, stem, L, input_seq``. What
  #82's ``score_rollout_worker.py`` consumes, matching the ``eval_targets.parquet``
  #212 scored the 554 with.
* ``eval2_new_predictor_manifest.csv`` — ``dataset, stem, gt_cif, gt_chain,
  input_seq, n_residues``. The eval-manifest shape exp74/exp78 use for pyconfind
  scoring and the ESMFold / ESMFold2 / Protenix Modal drivers use for fan-out.

Both are derived from :mod:`build_gt_contacts`'s output, so the sequence and the
chain here are exactly the ones the ground truth was computed against — the
failure mode this avoids is scoring a prediction for one chain against contacts
derived from another.

    uv run python build_predictor_inputs.py
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: Where build_gt_contacts.py cached the assembly mmCIFs. The manifest's
#: ``gt_cif`` is relative to this, matching exp78's ``--gt-root`` convention.
DEFAULT_GT_ROOT = Path("/data/exp226_gt/cif")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-manifest", type=Path, default=DATA / "eval2_new_gt_manifest.csv")
    ap.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    ap.add_argument("--out-targets", type=Path, default=DATA / "eval2_new_targets.parquet")
    ap.add_argument("--out-manifest", type=Path,
                    default=DATA / "eval2_new_predictor_manifest.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.gt_manifest.open()))
    missing = [r["gt_cif"] for r in rows if not (args.gt_root / r["gt_cif"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} GT structures missing under {args.gt_root}: "
                         f"{missing[:3]}; rerun build_gt_contacts.py")

    targets = pd.DataFrame([
        {"dataset": r["dataset"], "stem": r["stem"],
         "L": int(r["n_residues"]), "input_seq": r["input_seq"]}
        for r in rows
    ])
    if (targets["L"] != targets["input_seq"].str.len()).any():
        raise SystemExit("L disagrees with len(input_seq) for at least one target")
    args.out_targets.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(args.out_targets, index=False)

    with args.out_manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "stem", "gt_cif", "gt_chain",
                                                "input_seq", "n_residues"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "dataset": r["dataset"], "stem": r["stem"], "gt_cif": r["gt_cif"],
                # The chain pyconfind actually analysed for the ground truth,
                # not FoldBench's (sometimes label) chain id.
                "gt_chain": r["gt_chain"], "input_seq": r["input_seq"],
                "n_residues": r["n_residues"],
            })

    print(f"[targets] {len(targets)} rows, sum(L)={int(targets['L'].sum())}, "
          f"L in [{int(targets['L'].min())}, {int(targets['L'].max())}] "
          f"-> {args.out_targets}", flush=True)
    print(f"[manifest] {len(rows)} rows -> {args.out_manifest}", flush=True)
    print(f"[manifest] gt-root for scoring: {args.gt_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

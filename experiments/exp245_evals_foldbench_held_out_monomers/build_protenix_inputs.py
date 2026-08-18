# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 5b -- the exp12-shaped Protenix input tree for the proteins we must run.

exp12's ``prepare-inputs`` builds its tree from the *first N rows* of FoldBench's
monomer list, which cannot select a scattered subset. This emits the identical
tree -- ``jobs/<stem>.json``, ``gt/<stem>.cif``, ``manifest.csv`` -- for exactly
the units ``build_predictor_inputs.py`` marked as needing prediction, reusing
exp12's own :func:`build_protenix_job` so each job JSON is byte-for-byte what
exp12 would have emitted. Same approach #226 took for its 23.

The sequence written into each job is the one the ground truth was computed
against: a prediction indexed to a different sequence would be scored against
contacts in a different coordinate space.

    uv run python build_protenix_inputs.py
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import upstream as U

EXP12_DIR = U.EXPERIMENTS / "exp12_data_protenix_foldbench_monomers"
if not EXP12_DIR.is_dir():  # pragma: no cover - branch-layout guard
    raise SystemExit(f"exp12 directory not found at {EXP12_DIR}")
sys.path.insert(0, str(EXP12_DIR))

from prepare_inputs import MonomerTarget, build_protenix_job  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=U.DATA / "predictor_manifest_new.csv")
    parser.add_argument("--sets", type=Path, default=U.DATA / "eval_sets.csv")
    parser.add_argument("--cif-cache", type=Path, default=U.WORK / "cif")
    parser.add_argument("--out", type=Path, default=U.WORK / "protenix_inputs")
    args = parser.parse_args()

    by_stem = {row["stem"]: row for row in csv.DictReader(args.sets.open())}
    rows = list(csv.DictReader(args.manifest.open()))
    (args.out / "jobs").mkdir(parents=True, exist_ok=True)
    (args.out / "gt").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for row in rows:
        source = by_stem[row["stem"]]
        target = MonomerTarget(pdb_id=source["pdb_id"], assembly=1,
                               chain_id=source["chain_id"])
        if target.stem != row["stem"]:
            raise SystemExit(f"stem mismatch: {target.stem} vs {row['stem']}")
        job = build_protenix_job(target, row["input_seq"])
        (args.out / "jobs" / f"{target.stem}.json").write_text(
            json.dumps([job], indent=2))
        shutil.copyfile(args.cif_cache / row["gt_cif"],
                        args.out / "gt" / f"{target.stem}.cif")
        manifest_rows.append({
            "pdb_id": target.pdb_id, "chain_id": target.chain_id, "assembly": 1,
            "stem": target.stem, "n_residues": int(row["n_residues"]),
            "gt_cif": f"gt/{target.stem}.cif", "job_json": f"jobs/{target.stem}.json",
        })

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[protenix] {len(manifest_rows)} jobs + GT -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 7b — the exp12-shaped Protenix input tree for just the 23 new proteins.

exp12's ``prepare-inputs`` builds its tree from *the first N rows* of FoldBench's
monomer list. The 23 proteins here are a scattered subset of the other 234, so
that entry point cannot select them — and running it over all 334 would
re-download 334 assembly mmCIFs to use 23 of them.

This produces the identical tree (``jobs/<stem>.json``, ``gt/<stem>.cif``,
``manifest.csv``) for exactly the 23, reusing exp12's own
:func:`build_protenix_job` so the job JSON is byte-for-byte what exp12 would have
emitted, and the mmCIFs :mod:`build_gt_contacts` already fetched.

The sequence written into each job is the one the ground truth was computed
against, not a re-derivation — a Protenix prediction indexed to a different
sequence would be scored against contacts in a different coordinate space.

    uv run python build_protenix_inputs.py --out /data/exp226_gt/protenix_inputs
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# exp12 owns the Protenix job format; import it rather than restate it.
EXP12_DIR = HERE.parent / "exp12_data_protenix_foldbench_monomers"
if not EXP12_DIR.is_dir():  # pragma: no cover - branch-layout guard
    raise SystemExit(f"exp12 directory not found at {EXP12_DIR}")
sys.path.insert(0, str(EXP12_DIR))

from prepare_inputs import MonomerTarget, build_protenix_job  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=DATA / "eval2_new_predictor_manifest.csv")
    ap.add_argument("--targets", type=Path, default=DATA / "foldbench_targets.csv")
    ap.add_argument("--cif-cache", type=Path, default=Path("/data/exp226_gt/cif"))
    ap.add_argument("--out", type=Path, default=Path("/data/exp226_gt/protenix_inputs"))
    args = ap.parse_args()

    by_stem = {r["stem"]: r for r in csv.DictReader(args.targets.open())}
    rows = list(csv.DictReader(args.manifest.open()))
    (args.out / "jobs").mkdir(parents=True, exist_ok=True)
    (args.out / "gt").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for row in rows:
        target = MonomerTarget(pdb_id=by_stem[row["stem"]]["pdb_id"], assembly=1,
                               chain_id=by_stem[row["stem"]]["chain_id"])
        if target.stem != row["stem"]:
            raise SystemExit(f"stem mismatch: {target.stem} vs {row['stem']}")
        job = build_protenix_job(target, row["input_seq"])
        (args.out / "jobs" / f"{target.stem}.json").write_text(json.dumps([job], indent=2))
        cif = args.out / "gt" / f"{target.stem}.cif"
        shutil.copyfile(args.cif_cache / row["gt_cif"], cif)
        manifest_rows.append({
            "pdb_id": target.pdb_id, "chain_id": target.chain_id, "assembly": 1,
            "stem": target.stem, "n_residues": int(row["n_residues"]),
            "gt_cif": f"gt/{target.stem}.cif", "job_json": f"jobs/{target.stem}.json",
        })

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[protenix] {len(manifest_rows)} jobs + GT -> {args.out}", flush=True)
    print(f"[protenix] manifest -> {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

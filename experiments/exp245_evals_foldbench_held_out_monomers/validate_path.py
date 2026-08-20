# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 7b -- prove this evaluation reproduces PR #244's, protein by protein.

The usual gate for a new scoring path is the #75 E8 checkpoint on the legacy
554. That gate is not available here -- this eval set is not the 554 -- but a
stronger one is: two of the three checkpoints scored here are exactly the ones
PR #244 scored, and 97 of these proteins (all of eval-val, the historical
FoldBench-100 minus its three designs) are in #244's universe under the
``foldbench100`` label.

So the comparison is per protein, on the same weights, over the same proteins,
between two independent runs of the same recipe. It exercises everything this
experiment changed -- rebuilt ground truth, a new targets file, a new dataset
label, a re-adapted harness -- and holds all of it against a published
reference. Rollout scoring is stochastic, so the gate is a tolerance:
[#204](https://github.com/Open-Athena/MarinFold/issues/204) measured four
evaluations of one unchanged checkpoint spanning 0.0023 in mean R-precision, and
#244 reproduced its own E8 reference to 0.0015.

    uv run python validate_path.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import upstream as U

DATA = U.DATA
OUT = DATA / "path_validation.json"

PR244_ROWS = (U.EXP232_ROLLOUT_DIR / "data" / "all_r_rows.csv.gz")
#: PR #244's key for each checkpoint, and the model id this run emits.
PAIRS = {
    "exp232-m2-p06-decontam": "marinfold-exp232-decontam-m2-p06-step145199",
    "exp232-m1-p02-decontam": "marinfold-exp232-decontam-m1-p02-step145199",
}
#: Mean R-precision must agree within this. #204's four-evaluation spread of one
#: unchanged checkpoint is 0.0023; #244's own E8 gate uses 0.005.
MEAN_TOLERANCE = 0.005


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--marinfold", type=Path,
                        default=DATA / "coreweave_results" / "marinfold_precision.csv")
    args = parser.parse_args()

    published = pd.read_csv(PR244_ROWS)
    published = published[published.dataset == "foldbench100"]
    mine = pd.read_csv(args.marinfold)
    mine = mine[(mine["range"] == "all") & (mine["cut"] == "R")]
    sets = pd.read_csv(DATA / "eval_sets.csv")
    val_stems = set(sets.loc[sets.eval_set == "eval-val", "stem"])

    comparisons = {}
    for key, model in PAIRS.items():
        reference = published[published.key == key].set_index("stem").precision
        current = mine[mine.model == model].set_index("stem").precision
        shared = sorted(val_stems & set(reference.index) & set(current.index))
        if not shared:
            raise SystemExit(f"no shared proteins for {key}")
        left = current.loc[shared].to_numpy()
        right = reference.loc[shared].to_numpy()
        difference = left - right
        comparisons[key] = {
            "model": model,
            "n_proteins": len(shared),
            "mean_here": float(left.mean()),
            "mean_pr244": float(right.mean()),
            "mean_difference": float(difference.mean()),
            "max_abs_per_protein": float(np.abs(difference).max()),
            "n_identical": int((difference == 0).sum()),
            "pearson_r": float(np.corrcoef(left, right)[0, 1]),
            "within_tolerance": bool(abs(difference.mean()) <= MEAN_TOLERANCE),
        }

    report = {
        "reference": str(PR244_ROWS.relative_to(U.REPO)),
        "tolerance_on_mean": MEAN_TOLERANCE,
        "comparisons": comparisons,
        "passed": all(c["within_tolerance"] for c in comparisons.values()),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

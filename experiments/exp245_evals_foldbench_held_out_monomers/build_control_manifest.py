# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 6a -- pick the published proteins the baseline path is validated on.

Twelve of the units whose baseline scores this experiment *reuses*, spread over
the length range, re-scored from #78's stored predictions through exp245's
scoring path. Length is the axis chosen because the candidate-pair universe and
the ``L``-dependent precision cuts are where a scoring path most plausibly
diverges.

    uv run python build_control_manifest.py
"""
import argparse
from pathlib import Path

import pandas as pd

import upstream as U

DATA = U.DATA
OUT = DATA / "control_manifest.csv"
N_CONTROL = 12


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=N_CONTROL)
    args = parser.parse_args()

    reuse = pd.read_csv(DATA / "baseline_reuse.csv")
    gt = pd.read_csv(DATA / "gt_manifest.csv")
    published = gt[gt.stem.isin(reuse.loc[reuse.source == "published", "stem"])]
    ordered = published.sort_values("n_residues")
    step = max(1, len(ordered) // args.n)
    chosen = ordered.iloc[::step].head(args.n)
    chosen[["dataset", "stem", "gt_cif", "gt_chain", "input_seq", "n_residues"]].to_csv(
        OUT, index=False)
    print(f"[control] {len(chosen)} proteins, L={chosen.n_residues.min()}-"
          f"{chosen.n_residues.max()} -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

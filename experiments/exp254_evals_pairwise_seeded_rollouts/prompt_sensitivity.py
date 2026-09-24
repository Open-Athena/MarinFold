# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Describe prompt-arm map changes relative to within-iid sampling variation.

Remove injected seed rows before constructing contact-vote maps. Full consensus
maps use 100 samples per arm. The size-matched comparison uses disjoint random
halves (50 samples per map) both across arms and within iid; average splits
within each protein before averaging proteins. Splits are resamples of saved
outputs, not independent inference repeats or a formal equivalence test.

The same-index rollout Jaccard is descriptive: shared sampler seeds do not
ensure the same random path after different prompts. Neither this statistic nor
consensus overlap measures causal attention to particular contact statements.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_metrics import ARMS, load_detail, resolved_pairs, true_matrix
from common import EXPECTED_UNITS, load_ground_truth, load_targets


def top(votes: np.ndarray, cut: int) -> set[int]:
    """Select a fixed-size map using the original stable vote tie order."""
    if cut <= 0 or cut > len(votes):
        raise ValueError("map cut must be positive and fit the candidate universe")
    return set(np.argsort(-votes, kind="stable")[:cut])


def change(a: set[int], b: set[int]) -> float:
    """Return the replaced fraction of two equally sized contact maps."""
    if not a or len(a) != len(b):
        raise ValueError("map comparison requires nonempty, equal-size maps")
    return 1 - len(a & b) / len(a)


def main() -> None:
    """Compare existing prompt arms without model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--splits", type=int, default=40)
    args = parser.parse_args()
    if args.splits <= 0:
        parser.error("--splits must be positive")
    run = args.run
    gt = load_ground_truth()
    details = {arm.directory: load_detail(run / arm.directory) for arm in ARMS}
    groups = {
        arm: {key: g for key, g in f.groupby(["dataset", "stem"])}
        for arm, f in details.items()
    }
    expected = {(target.dataset, target.stem) for target in load_targets()}
    if len(expected) != EXPECTED_UNITS or any(set(group) != expected for group in groups.values()):
        raise ValueError("each arm must contain exactly the eval-val target universe")
    rows = []
    for key in groups["iid"]:
        record = gt[key]
        pi, pj, sep = resolved_pairs(np.asarray(record["resolved"]))
        pi, pj = pi[sep >= 6], pj[sep >= 6]
        truth = true_matrix(record["L"], record["contacts"])
        R = int(truth[pi, pj].sum())
        lookup = {
            int(i) * record["L"] + int(j): k for k, (i, j) in enumerate(zip(pi, pj))
        }
        matrices = {}
        for arm in groups:
            frame = groups[arm][key]
            if not np.array_equal(np.sort(frame.rollout.unique()), np.arange(100)):
                raise ValueError(f"{arm}/{key}: expected exactly rollouts 0..99")
            matrix = np.zeros((100, len(pi)), dtype=np.uint8)
            for row in frame[~frame.is_seed].itertuples():
                pair = lookup.get(row.i * record["L"] + row.j)
                if pair is not None:
                    matrix[row.rollout, pair] = 1
            matrices[arm] = matrix
        control = matrices["iid"]
        for arm in list(matrices)[1:]:
            sample = matrices[arm]
            paired = []
            for r in range(100):
                a = set(np.flatnonzero(control[r]))
                b = set(np.flatnonzero(sample[r]))
                paired.append(len(a & b) / len(a | b) if a | b else 1.0)
            half_cross = []
            half_iid = []
            generator = np.random.default_rng(254)
            for draw in range(args.splits):
                ids = generator.permutation(100)
                a, b = ids[:50], ids[50:]
                # Cross arms use disjoint realization/sampler IDs, as does control.
                half_cross.append(
                    change(
                        top(control[a].sum(0, dtype=np.int32), R),
                        top(sample[b].sum(0, dtype=np.int32), R),
                    )
                )
                half_iid.append(
                    change(
                        top(control[a].sum(0, dtype=np.int32), R),
                        top(control[b].sum(0, dtype=np.int32), R),
                    )
                )
            rows.append(
                dict(
                    stem=key[1],
                    arm=arm,
                    R=R,
                    top_R_replaced_100=change(
                        top(control.sum(0, dtype=np.int32), R),
                        top(sample.sum(0, dtype=np.int32), R),
                    ),
                    paired_rollout_jaccard=np.mean(paired),
                    top_R_replaced_cross_50=np.mean(half_cross),
                    top_R_replaced_iid_50=np.mean(half_iid),
                )
            )
    f = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    f.to_csv(args.out / "exp254_prompt_sensitivity_per_protein.csv", index=False)
    f.groupby("arm").mean(numeric_only=True).to_csv(
        args.out / "exp254_prompt_sensitivity_summary.csv"
    )
    print(f.groupby("arm").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()

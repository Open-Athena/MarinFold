# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-check reported scores independently and summarize saved run telemetry.

Run verify_conditioning.py first: this script checks the scoring arithmetic
without importing the primary analyzer. It does not replace raw-text validation.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(plan_path: Path, run: Path, data: Path) -> dict:
    """Recompute every score using explicit candidate pairs and Python sorting."""
    plan = json.loads(plan_path.read_text())
    saved = pd.read_csv(data / "conditioning_per_repeat.csv").set_index(
        ["stem", "repeat", "scope", "arm"]
    )
    checked = 0
    timing_frames = []
    with np.load(plan_path.parent / plan["source_votes_file"]) as source:
        for target in plan["targets"]:
            stem = target["stem"]
            resolved = sorted(target["resolved"])
            candidates = [(i, j) for i in resolved for j in resolved if j - i >= 6]
            truth = {tuple(pair) for pair in target["truth"]}
            timing = pd.read_csv(run / "units" / f"{stem}.timings.csv")
            with gzip.open(run / "units" / f"{stem}.raw.json.gz", "rt") as handle:
                raw = json.load(handle)
            timing["empty_rollouts"] = [
                sum(
                    not item["contacts"]
                    for item in raw[f"r{row.replicate}__{row.mode}"]
                )
                for row in timing.itertuples()
            ]
            timing_frames.append(timing)
            with np.load(run / "units" / f"{stem}.npz") as matrices:
                for repeat, contexts in enumerate(target["contexts"]):
                    excluded = {
                        tuple(pair) for pairs in contexts.values() for pair in pairs
                    }
                    scores = {
                        arm: matrices[f"r{repeat}__{arm}__votes"].copy()
                        for arm in plan["arms"]
                    }
                    scores["iid200"] = scores["iid"] + scores["iid_repeat"]
                    scores["source_plus_iid200"] = source[stem] + scores["iid"]
                    for scope in ("full_pipeline", "withheld_continuation"):
                        universe = [
                            pair
                            for pair in candidates
                            if scope == "full_pipeline" or pair not in excluded
                        ]
                        denominator = len(truth.intersection(universe))
                        if denominator <= 0:
                            raise ValueError(f"Empty positive universe: {stem}")
                        ii, jj = np.asarray(universe).T
                        for arm, matrix in scores.items():
                            values = matrix[ii, jj].copy()
                            given = {tuple(pair) for pair in contexts.get(arm, [])}
                            if scope == "full_pipeline":
                                for index, pair in enumerate(universe):
                                    if pair in given:
                                        values[index] = plan["n_rollouts"]
                            order = sorted(
                                range(len(universe)), key=lambda k: -int(values[k])
                            )
                            precision = (
                                sum(universe[k] in truth for k in order[:denominator])
                                / denominator
                            )
                            row = saved.loc[(stem, repeat, scope, arm)]
                            if (
                                row.n_true != denominator
                                or abs(row.precision - precision) > 1e-12
                            ):
                                raise ValueError(
                                    f"Independent score differs: {stem}/{repeat}/{scope}/{arm}"
                                )
                            checked += 1
    if checked != len(saved):
        raise ValueError("Score table contains missing or extra rows")
    timings = pd.concat(timing_frames, ignore_index=True)
    timings.to_csv(data / "conditioning_timings.csv", index=False)
    runtime = timings.groupby("mode", as_index=False).agg(
        groups=("stem", "size"),
        n_rollouts=("n_rollouts", "sum"),
        generated_tokens=("generated_tokens", "sum"),
        empty_rollouts=("empty_rollouts", "sum"),
        budget_terminated_rollouts=("unfinished_rollouts", "sum"),
        novel_generated_pairs=("novel_generated_pairs", "sum"),
        copied_context_pairs=("copied_context_pairs", "sum"),
        generation_seconds=("elapsed_seconds", "sum"),
        probability_probe_seconds=("probability_probe_seconds", "sum"),
    )
    runtime["budget_termination_rate"] = (
        runtime.budget_terminated_rollouts / runtime.n_rollouts
    )
    runtime["empty_rate"] = runtime.empty_rollouts / runtime.n_rollouts
    runtime["novel_pairs_per_rollout"] = (
        runtime.novel_generated_pairs / runtime.n_rollouts
    )
    runtime.to_csv(data / "conditioning_runtime.csv", index=False)
    changes = pd.read_csv(data / "conditioning_changes.csv")
    changes.groupby(["scope", "arm"], as_index=False)[
        [
            "top_R_turnover",
            "vote_frequency_MAD",
            "vote_frequency_L1",
            "probability_MAD",
            "probability_L1",
        ]
    ].mean().to_csv(data / "conditioning_response_summary.csv", index=False)
    report = {
        "independently_recomputed_scores": checked,
        "n_proteins": len(plan["targets"]),
        "method": "Explicit resolved residue pairs, truth-set intersection, Python stable sort; no analyzer imports",
        "max_allowed_absolute_difference": 1e-12,
        "runtime_scope": "Accepted final units only; excludes aborted attempts. Generation time excludes prompt construction and probing. Repeated model-load fields must not be summed.",
    }
    (data / "conditioning_independent_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    """Audit a complete analyzed run without a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.plan, args.run, args.data), indent=2))


if __name__ == "__main__":
    main()

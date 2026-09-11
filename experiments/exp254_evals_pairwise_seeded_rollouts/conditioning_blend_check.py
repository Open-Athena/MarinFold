# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc equal-weight blends of archived and conditioned contact votes.

This exploratory check retains first-pass score information without inference,
weight tuning, or candidate selection. Both readouts use the full standard
resolved-pair universe. The continuation readout includes any copied pairs that
the model actually emitted; the complete-document readout instead sets each
supplied pair to N votes before adding the archived votes. The predeclared
primary remains unchanged. Intervals are pointwise and conditional on saved runs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_conditioning import (
    PRIMARY_REFERENCE,
    candidate_universes,
    complete_votes,
    margin_status,
    paired_interval,
    precision_at_r,
    read_matrix,
)


def checked_unit(plan_sha: str, run: Path, stem: str) -> Path:
    """Require a complete unit and verify its vote-file digest before scoring."""
    marker = json.loads((run / "units" / f"{stem}.complete.json").read_text())
    if marker["stem"] != stem or marker["plan_sha256"] != plan_sha:
        raise ValueError(f"{stem}: completion manifest differs from frozen plan")
    path = run / "units" / f"{stem}.npz"
    payload = path.read_bytes()
    expected = marker["files"]["npz"]
    if (
        len(payload) != expected["bytes"]
        or hashlib.sha256(payload).hexdigest() != expected["sha256"]
    ):
        raise ValueError(f"{stem}: vote file checksum mismatch")
    return path


def analyze_blends(plan_path: Path, run: Path, out: Path) -> pd.DataFrame:
    """Score four fixed blends and bootstrap effects after replicate averaging."""
    plan_bytes = plan_path.read_bytes()
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    targets = plan["targets"]
    n_rollouts, n_repeats = plan["n_rollouts"], plan["n_repeats"]
    if not targets or len({target["stem"] for target in targets}) != len(targets):
        raise ValueError("plan must contain unique nonempty targets")
    if n_rollouts <= 0 or n_repeats <= 0:
        raise ValueError("plan requires positive rollout and replicate counts")
    if plan["source_n_rollouts"] != n_rollouts:
        raise ValueError("this fixed blend requires equal first/second-pass counts")
    source_path = plan_path.parent / plan["source_votes_file"]
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha != plan["source_votes_sha256"]:
        raise ValueError("archived first-pass vote checksum mismatch")
    rows = []
    with np.load(source_path, allow_pickle=False) as source_archive:
        if set(source_archive.files) != {target["stem"] for target in targets}:
            raise ValueError("source archive does not match the planned targets")
        for target in targets:
            stem, length = target["stem"], target["L"]
            source = read_matrix(source_archive, stem, length, n_rollouts, True)
            path = checked_unit(plan_sha, run, stem)
            if len(target["contexts"]) != n_repeats:
                raise ValueError(f"{stem}: incomplete planned context replicates")
            with np.load(path, allow_pickle=False) as archive:
                for repeat, contexts in enumerate(target["contexts"]):
                    pi, pj, truth, _ = candidate_universes(target, contexts)
                    iid = read_matrix(
                        archive, f"r{repeat}__iid__votes", length, n_rollouts, True
                    )
                    reference, _, n_true = precision_at_r(source + iid, truth, pi, pj)
                    for arm in ("pred_small", "pred_large"):
                        votes = read_matrix(
                            archive,
                            f"r{repeat}__{arm}__votes",
                            length,
                            n_rollouts,
                            True,
                        )
                        for readout, conditioned in (
                            ("source_plus_continuation", votes),
                            (
                                "source_plus_complete_document",
                                complete_votes(votes, contexts[arm], n_rollouts),
                            ),
                        ):
                            precision, _, _ = precision_at_r(
                                source + conditioned, truth, pi, pj
                            )
                            rows.append(
                                {
                                    "dataset": target["dataset"],
                                    "stem": stem,
                                    "arm": arm,
                                    "readout": readout,
                                    "repeat": repeat,
                                    "L": length,
                                    "n_true": n_true,
                                    "precision": precision,
                                    "reference_precision": reference,
                                }
                            )
    per_protein = (
        pd.DataFrame(rows)
        .groupby(["dataset", "stem", "arm", "readout"], as_index=False)
        .agg(
            precision=("precision", "mean"),
            reference_precision=("reference_precision", "mean"),
            n_repeats=("repeat", "nunique"),
            L=("L", "first"),
            n_true=("n_true", "first"),
        )
    )
    if not (per_protein.n_repeats == n_repeats).all():
        raise ValueError("blend scores have incomplete replicate coverage")
    per_protein["delta"] = per_protein.precision - per_protein.reference_precision
    summary_rows = []
    for (arm, readout), group in per_protein.groupby(["arm", "readout"]):
        interval = paired_interval(group.delta.to_numpy())
        summary_rows.append(
            dict(
                role="post_hoc_exploratory_fixed_equal_weight_blend",
                scope="full_pipeline",
                arm=arm,
                readout=readout,
                reference=PRIMARY_REFERENCE,
                arm_mean=float(group.precision.mean()),
                reference_mean=float(group.reference_precision.mean()),
                **interval,
                practical_margin=plan["practical_margin"],
                margin_status=margin_status(
                    interval["ci_low"], interval["ci_high"], plan["practical_margin"]
                ),
                n_repeats=n_repeats,
                n_rollouts_per_pass=n_rollouts,
                plan_sha256=plan_sha,
                source_votes_sha256=source_sha,
            )
        )
    summary = pd.DataFrame(summary_rows)
    out.mkdir(parents=True, exist_ok=True)
    per_protein.to_csv(out / "conditioning_blend_per_protein.csv", index=False)
    summary.to_csv(out / "conditioning_blend_summary.csv", index=False)
    return summary


def main() -> int:
    """Run the exploratory saved-vote check without modifying canonical results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(analyze_blends(args.plan, args.run, args.out).to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

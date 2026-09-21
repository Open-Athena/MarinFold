# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect matched ordinary validation CE against the exp232 m2/p06 *continuation*.

`collect_exp232_comparison.py` joins exp279 against the original exp232 sweep
run, which is only valid through step 116160. After that the sweep cools down
while exp279 holds peak LR, so matching the two beyond that point compares a
decaying learning rate against a constant one and understates the soft arm.

The one-hot lineage that actually carries the recipe to step 363000, and that
produced the published `contacts-v1-exp232-m2-p06-train-1.5B` model, is three
runs spliced at their resume points. Every continuation segment is asserted to
sit at the constant peak LR that exp279 also holds through update 333960, so
the comparison stays learning-rate matched end to end.
"""

import csv
import math
from pathlib import Path
from typing import Any

import wandb

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = "open-athena/MarinFold"
SOFT_RUN = "exp279-soft-s0-cw-h100x32-b02"
TOKENS_PER_STEP = 128 * 8192
PEAK_LR = 0.001

# (run, first step this run is authoritative for, last step, LR expectation).
# The sweep owns the warmup and the pre-fork stretch; the continuation forks
# from it at peak LR; the recover run supersedes the crashed train run from its
# own resume point onward.
LINEAGE = [
    ("prot-exp232-cw-cv1-decontam-s02-m2-p06-aug", 0, 116160, None),
    ("prot-exp232-cw-cv1-decontam-train-s01-m2-p06-srcpeak-augcont", 116161, 219855, PEAK_LR),
    ("prot-exp232-cw-cv1-decontam-recover-a03-skipstep-m2-p06-srcpeak-augcont", 219856, None, PEAK_LR),
]


def history(run: Any) -> dict[int, tuple[float, float | None]]:
    """Return finite (eval loss, LR) keyed by global step."""
    out: dict[int, tuple[float, float | None]] = {}
    frame = run.history(
        keys=["global_step", "eval/loss", "optim/learning_rate"], samples=1000000
    )
    if "global_step" not in frame.columns:
        raise ValueError(f"No global_step history in {run.name}")
    for row in frame.dropna(subset=["global_step", "eval/loss"]).itertuples():
        step = int(row.global_step)
        loss = float(getattr(row, "_3"))
        if not math.isfinite(loss):
            raise ValueError(f"Nonfinite eval/loss at step {step} in {run.name}")
        if step in out:
            raise ValueError(f"Duplicate eval/loss at step {step} in {run.name}")
        lr = getattr(row, "_4", None)
        out[step] = (loss, None if lr is None or math.isnan(lr) else float(lr))
    return out


def one_hot_series(api: Any) -> tuple[dict[int, float], dict[int, str]]:
    """Splice the lineage, asserting the peak LR on every continuation segment."""
    losses: dict[int, float] = {}
    provenance: dict[int, str] = {}
    for name, lo, hi, expected_lr in LINEAGE:
        for step, (loss, lr) in history(api.run(f"{PROJECT}/{name}")).items():
            if step < lo or (hi is not None and step > hi):
                continue
            if expected_lr is not None and lr is not None:
                if abs(lr - expected_lr) > 1e-9:
                    raise ValueError(
                        f"{name} step {step} is at LR {lr}, not the peak {expected_lr}; "
                        "the comparison would not be learning-rate matched"
                    )
            if step in losses:
                raise ValueError(f"Lineage segments overlap at step {step}")
            losses[step] = loss
            provenance[step] = name
    return losses, provenance


def main() -> None:
    api = wandb.Api(timeout=90)
    soft = {s: v[0] for s, v in history(api.run(f"{PROJECT}/{SOFT_RUN}")).items()}
    one_hot, provenance = one_hot_series(api)
    matched = sorted(soft.keys() & one_hot.keys())
    if len(matched) < 2:
        raise ValueError("The lineages do not share enough exact evaluations")

    rows = [
        {
            "global_step": step,
            "nominal_tokens": step * TOKENS_PER_STEP,
            "soft_eval_loss": soft[step],
            "one_hot_eval_loss": one_hot[step],
            "soft_minus_one_hot": soft[step] - one_hot[step],
            "soft_perplexity": math.exp(soft[step]),
            "one_hot_perplexity": math.exp(one_hot[step]),
            "one_hot_run": provenance[step],
        }
        for step in matched
    ]
    output = EXPERIMENT / "data/exp232_m2_p06_continuation_validation.csv"
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} matched evaluations through step {matched[-1]} to {output}")
    for name, *_ in LINEAGE:
        n = sum(1 for s in matched if provenance[s] == name)
        print(f"  {n:>3} matched points from {name}")


if __name__ == "__main__":
    main()

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare our reproduction's val-loss curve to Eric's exp117 run, step for step.

This is the experiment's verdict, and it does not require waiting for the run to
finish: both runs log the same metric on the same schedule (2 evals/epoch, every
2,230 steps), so the curves can be compared at matched steps while ours is still
training. A reproduction that tracks his all the way is far stronger evidence
than a single matching endpoint would be.

Metric names differ cosmetically -- his component key carries a storage prefix
(``tokenized/contacts-v1-val``), ours does not -- but it is the same series:
full held-out contacts-v1 val, unmasked, ``max_eval_batches=None``.

    ours   open-athena/MarinFold  exp150-cv1repro-1_5b-e16-lr3p162e-3-wd0p2-bs128
           eval/contacts-v1-val/loss
    Eric   eric-czech/marin       prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs128-us-east5
           eval/tokenized/contacts-v1-val/loss   -> final 2.71122 @ step 71,359

Usage (needs W&B credentials, e.g. ~/.netrc)::

    uv run --with wandb python compare_to_eric.py

Exit status is 0 when the runs agree at the final step (|delta| <= TOLERANCE),
1 on disagreement, and 2 while ours is still training (no verdict yet).
"""
from __future__ import annotations

import sys

OURS_PROJECT = "open-athena/MarinFold"
OURS_RUN = "exp150-cv1repro-1_5b-e16-lr3p162e-3-wd0p2-bs128"
OURS_METRIC = "eval/contacts-v1-val/loss"

ERIC_PROJECT = "eric-czech/marin"
ERIC_RUN = "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs128-us-east5"
ERIC_METRIC = "eval/tokenized/contacts-v1-val/loss"

# Issue #150 success criteria.
TARGET_LOSS = 2.7112       # Eric's final
TOLERANCE = 0.01           # primary criterion: |delta| <= 0.01
INVESTIGATE = 0.02         # > 0.02 => treat as a finding and bisect
FINAL_STEP = 71_359        # last step index of the 71,360-step run


def _curve(api, project: str, run_name: str, metric: str) -> tuple[list[tuple[int, float]], str]:
    """Return ``([(step, loss), ...], state)`` for one run, sorted by step."""
    matches = [r for r in api.runs(project) if r.name == run_name]
    if not matches:
        raise SystemExit(f"run not found: {project}/{run_name}")
    run = matches[0]
    history = run.history(keys=["_step", metric], pandas=False)
    points = sorted(
        (int(row["_step"]), float(row[metric]))
        for row in history
        if row.get(metric) is not None
    )
    return points, run.state


def main() -> int:
    import wandb

    api = wandb.Api()
    ours, ours_state = _curve(api, OURS_PROJECT, OURS_RUN, OURS_METRIC)
    eric, _ = _curve(api, ERIC_PROJECT, ERIC_RUN, ERIC_METRIC)
    eric_by_step = dict(eric)

    print(f"ours: {OURS_RUN}  [{ours_state}]  {len(ours)} evals")
    print(f"eric: {ERIC_RUN}  {len(eric)} evals\n")
    print(f"{'step':>8}  {'eric':>8}  {'ours':>8}  {'delta':>9}")
    print("-" * 40)

    deltas = []
    for step, ours_loss in ours:
        eric_loss = eric_by_step.get(step)
        if eric_loss is None:
            print(f"{step:>8}  {'--':>8}  {ours_loss:>8.4f}  {'(no match)':>9}")
            continue
        delta = ours_loss - eric_loss
        deltas.append(delta)
        print(f"{step:>8}  {eric_loss:>8.4f}  {ours_loss:>8.4f}  {delta:>+9.4f}")

    if deltas:
        mean_abs = sum(abs(d) for d in deltas) / len(deltas)
        print(f"\nmean |delta| over {len(deltas)} matched evals: {mean_abs:.4f}")

    final = dict(ours).get(FINAL_STEP)
    print()
    if final is None:
        last_step = ours[-1][0] if ours else 0
        pct = 100.0 * last_step / FINAL_STEP
        print(f"IN PROGRESS -- at step {last_step:,}/{FINAL_STEP:,} ({pct:.1f}%). No final verdict yet.")
        if deltas:
            print(f"Tracking so far: mean |delta| {mean_abs:.4f} (curves {'agree' if mean_abs <= INVESTIGATE else 'diverge'}).")
        return 2

    delta = final - TARGET_LOSS
    print(f"FINAL  ours {final:.4f}  vs  Eric {TARGET_LOSS:.4f}   delta {delta:+.4f}")
    if abs(delta) <= TOLERANCE:
        print(f"REPRODUCED -- within +-{TOLERANCE}. MarinFold's default_train matches Eric's harness.")
        return 0
    if abs(delta) <= INVESTIGATE:
        print(f"MARGINAL -- outside +-{TOLERANCE} but within {INVESTIGATE}. Report the delta; note it.")
        return 1
    print(f"DIVERGED -- |delta| > {INVESTIGATE}. Treat as a finding and bisect:")
    print("  shuffle policy -> token cache -> optimizer defaults -> packing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

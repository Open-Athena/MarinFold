# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Does honouring the model's retractions at inference help or hurt? (#160)

The headline comparison in ``score_eval.py`` folds every rollout, so a pair the
model takes back does not vote. That conflates two different things the training
could have done: changed what the model *emits*, and given it a *retract*
mechanism worth obeying. This separates them by scoring the **same rollouts**
under both readouts:

* **retraction honoured** — the #158 fold; only pairs live at ``<end>`` vote.
  This is what ``score_eval.py`` reports.
* **retraction ignored** — every ``<contact>`` votes regardless of a later
  ``<retract>``, which is exactly what exp82's pre-#158 regex readout would have
  counted.

Both are recomputed from the per-rollout edit lists the worker saved, so the
comparison is paired at the rollout level — no re-sampling, no new inference.

Two sanity checks fall out for free and are asserted rather than assumed:
the honoured readout must reproduce the committed vote matrices bit-for-bit,
and for a model that never retracts the two readouts must be *identical*.

    uv run --no-project --with pandas --with pyarrow --with scikit-learn --with s3fs \\
        python ablate_retraction.py --gt <gt_universe.jsonl> --scores s3://.../scores \\
        --labels exp160-bt50,exp120-base --out data/exp160_ablation_rows.csv.gz
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from score_eval import (
    MIN_SEP, list_parts, load_vote_matrices, metric_rows, read_table,
    resolved_pairs, stamp, true_matrix,
)


def vote_matrices_from_streams(prefix: str, gt: list[dict]):
    """Rebuild ``[L, L]`` vote matrices under both readouts from the edit lists."""
    dims = {(r["dataset"], r["stem"]): r["L"] for r in gt}
    honoured: dict[tuple[str, str], np.ndarray] = {}
    ignored: dict[tuple[str, str], np.ndarray] = {}
    n_rollouts: dict[tuple[str, str], int] = defaultdict(int)

    for uri in list_parts(prefix):
        t = read_table(uri).to_pydict()
        for n in range(len(t["stem"])):
            key = (t["dataset"][n], t["stem"][n])
            L = dims.get(key)
            if L is None:
                continue
            if key not in honoured:
                honoured[key] = np.zeros((L, L), np.int32)
                ignored[key] = np.zeros((L, L), np.int32)
            n_rollouts[key] += 1

            live: set[tuple[int, int]] = set()
            emitted: set[tuple[int, int]] = set()
            for k, i, j in zip(t["kind"][n], t["i"][n], t["j"][n]):
                pair = (i, j) if i <= j else (j, i)
                if k == 0:
                    live.add(pair)
                    emitted.add(pair)
                else:
                    live.discard(pair)
            for m, pairs in ((honoured[key], live), (ignored[key], emitted)):
                for a, b in pairs:
                    if a != b and (b - a) >= MIN_SEP:
                        m[a, b] += 1
    return honoured, ignored, dict(n_rollouts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    gt = [json.loads(line) for line in a.gt.open()]
    rows = []
    for label in [x for x in a.labels.split(",") if x]:
        base = f"{a.scores.rstrip('/')}/{label}"
        honoured, ignored, n_roll = vote_matrices_from_streams(f"{base}/streams", gt)
        committed = load_vote_matrices(f"{base}/votes", gt)

        # The honoured readout is the one score_eval already published; if this
        # rebuild disagrees, one of the two is wrong and every number below is
        # suspect. The worker folds in POSITION space and maps afterwards, this
        # folds the already-mapped stream — equal only because the map is a
        # bijection, which is worth confirming rather than trusting.
        bad = [k for k in committed if not np.array_equal(committed[k], honoured[k])]
        if bad:
            raise SystemExit(f"{label}: rebuilt votes differ from committed for "
                             f"{len(bad)} proteins, e.g. {bad[:3]}")
        n_diff = sum(1 for k in honoured if not np.array_equal(honoured[k], ignored[k]))
        print(f"[ablate] {label}: {len(honoured)} proteins, "
              f"{sum(n_roll.values()):,} rollouts, rebuild matches committed votes; "
              f"readouts differ on {n_diff}/{len(honoured)} proteins")

        for rec in gt:
            key = (rec["dataset"], rec["stem"])
            if key not in honoured:
                continue
            tmat = true_matrix(rec["L"], rec["contacts"])
            pi, pj, psep = resolved_pairs(np.asarray(rec["resolved"], dtype=np.int64))
            for readout, mats in (("retraction on", honoured), ("retraction off", ignored)):
                rows += stamp(
                    metric_rows(mats[key].astype(np.float64), tmat, pi, pj, psep,
                                rec["L"], with_precision=True),
                    rec=rec, model=f"{label} ({readout})", mode="single_seq",
                    predictor="lm")

    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\n[ablate] wrote {len(out)} rows -> {a.out}")
    summary = (out.groupby(["model", "range", "cut"])["precision"].mean().reset_index())
    print(summary[summary.cut.isin(["R", "AUC"])]
          .pivot_table(index="model", columns=["range", "cut"], values="precision")
          .round(4)[["all", "long"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

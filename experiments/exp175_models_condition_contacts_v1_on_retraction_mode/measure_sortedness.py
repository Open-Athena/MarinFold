# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Did the model inherit the corpus's sorted-flush artifact? (#159 bug check)

``backtrack_engine.py`` used to append every still-missing ground-truth contact
in ``sorted()`` order at the end of a document. That block is ~80% of a
backtracking document, so a model trained on it learns to finish with a sorted
sweep — and a sorted sweep is nearly deterministic, which collapses the
100-rollout vote the inference recipe depends on.

This measures the artifact *in the model's own rollouts*, which is the only
place it matters. For each rollout we take the ``<contact>`` statements in
emission order and ask what fraction of adjacent pairs are in increasing
canonical order. **0.5 is the null** (a random order); the published corpus's
backtracking half scored 0.869 and the model trained on it 0.851.

Reads the ``streams`` parquet the eval worker writes — the ordered edit list,
not the folded set, because order is precisely what is being tested.

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'
    uv run --no-project --with 'fsspec[s3]' --with pyarrow --with numpy \\
        python measure_sortedness.py --scores s3://.../scores_v2 \\
        --labels exp175-backtracking,exp175-clean,exp120-base
"""

from __future__ import annotations

import argparse

import numpy as np
import pyarrow.parquet as pq


def sortedness(prefix: str, max_parts: int) -> tuple[float, float, int]:
    """``(mean sortedness, fraction of fully-sorted rollouts, n rollouts)``.

    Rollouts with fewer than two contacts have no adjacent pair and are skipped
    rather than counted as sorted — with ~200 contacts per rollout they are
    rare, but scoring them 1.0 would bias the statistic toward the artifact.
    """
    import fsspec

    fs, _ = fsspec.core.url_to_fs(prefix)
    parts = sorted(fs.unstrip_protocol(p) for p in fs.glob(f"{prefix.rstrip('/')}/*.parquet"))
    if not parts:
        raise SystemExit(f"no parquet parts under {prefix}")

    fracs, full = [], 0
    for uri in parts[:max_parts]:
        with fsspec.open(uri, "rb") as fh:
            t = pq.read_table(fh, columns=["kind", "i", "j"]).to_pydict()
        for kinds, iis, jjs in zip(t["kind"], t["i"], t["j"]):
            # kind 0 is <contact>; the flush only ever emitted contacts, and a
            # retraction interleaved in the stream says nothing about whether
            # the assertions swept in order.
            seq = [(min(i, j), max(i, j)) for k, i, j in zip(kinds, iis, jjs) if k == 0]
            if len(seq) < 2:
                continue
            up = sum(a < b for a, b in zip(seq, seq[1:]))
            fracs.append(up / (len(seq) - 1))
            full += up == len(seq) - 1
    return float(np.mean(fracs)), full / len(fracs), len(fracs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="prefix holding <label>/streams")
    ap.add_argument("--labels", required=True, help="comma-separated arm labels")
    ap.add_argument("--max-parts", type=int, default=8,
                    help="parts per arm; each holds many rollouts, 8 is plenty")
    a = ap.parse_args()

    print(f"{'arm':24s} {'sortedness':>11s} {'fully sorted':>13s} {'n':>8s}")
    for label in a.labels.split(","):
        try:
            mean, full, n = sortedness(f"{a.scores.rstrip('/')}/{label}/streams", a.max_parts)
        except SystemExit as e:
            print(f"{label:24s} {e}")
            continue
        print(f"{label:24s} {mean:11.3f} {full:12.1%} {n:8d}")
    print("\n0.500 = random order (the null); published corpus 0.869, "
          "model trained on it 0.851")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

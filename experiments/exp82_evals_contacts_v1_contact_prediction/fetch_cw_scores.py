# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild local ``[L, L]`` npz score matrices from the CoreWeave shard parquets.

``score_rollout_worker_cw.py`` emits sparse upper-triangle triplets
(``dataset``/``stem``/``L``/``i``/``j``/``votes``) — one parquet part per chunk —
because per-protein npz on S3 would be thousands of tiny objects. This
reconstitutes the exact dense float16 matrices ``score_rollout_vllm.py`` writes
locally, so ``build_rollout_rows.py`` scores CoreWeave and local runs through one
unchanged path.

Verifies completeness: every ``(dataset, stem)`` must appear exactly once across
all parts, and 554 units must be present.

    python fetch_cw_scores.py --parts 's3://…/scores/exp117_nok' --out _scratch/scores_exp117_nok
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import fsspec
import numpy as np
import pyarrow.parquet as pq

EXPECTED_UNITS = 554


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="s3:// prefix holding shard-*-part-*.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect", type=int, default=EXPECTED_UNITS)
    a = ap.parse_args()

    fs, root = fsspec.core.url_to_fs(a.parts)
    parts = sorted(fs.glob(f"{root.rstrip('/')}/shard-*-part-*.parquet"))
    if not parts:
        raise SystemExit(f"no parquet parts under {a.parts}")
    print(f"{len(parts)} part files under {a.parts}")

    tri: dict[tuple[str, str], list] = defaultdict(list)
    lengths: dict[tuple[str, str], int] = {}
    seen_in_part: dict[tuple[str, str], set[str]] = defaultdict(set)
    for p in parts:
        with fs.open(p, "rb") as fh:
            t = pq.read_table(fh)
        d = t.to_pydict()
        for ds, stem, L, i, j, v in zip(d["dataset"], d["stem"], d["L"], d["i"], d["j"], d["votes"]):
            key = (ds, stem)
            tri[key].append((i, j, v))
            lengths[key] = L
            seen_in_part[key].add(p)

    dupes = {k: sorted(v) for k, v in seen_in_part.items() if len(v) > 1}
    if dupes:
        # Two parts covering one protein means a retried shard double-counted it;
        # the votes would be summed and the matrix wrong. Fail rather than average.
        raise SystemExit(f"!! {len(dupes)} protein(s) appear in more than one part, e.g. "
                         f"{list(dupes.items())[:2]}")

    a.out.mkdir(parents=True, exist_ok=True)
    for (ds, stem), triplets in tri.items():
        L = lengths[(ds, stem)]
        M = np.zeros((L, L), np.float32)
        for i, j, v in triplets:
            M[i, j] = v
            M[j, i] = v
        np.savez_compressed(a.out / f"{ds}__{stem}.npz", score=M.astype(np.float16))

    print(f"wrote {len(tri)} score matrices -> {a.out}")
    if len(tri) != a.expect:
        raise SystemExit(f"!! INCOMPLETE: {len(tri)} units, expected {a.expect}")
    print(f"completeness OK: {len(tri)}/{a.expect} (dataset, stem) units, "
          f"{len({s for _, s in tri})} unique stems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

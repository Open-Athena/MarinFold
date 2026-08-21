# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-protein timing records for this experiment's two rollout arms.

The root `AGENTS.md` asks every predictor run to leave a per-input timing CSV
behind rather than something reconstructable from logs. Two honest caveats
apply to this one and are recorded in the rows themselves:

* **The timing unit is a chunk, not a protein.** ``score_rollouts.py`` submits
  eight proteins x 100 rollouts to vLLM in one ``generate`` call and vLLM
  schedules them together, so there is no separable per-protein inference time
  to record -- only the chunk's wall time, which every protein in the chunk
  shares. ``timing_unit`` and ``chunk_size`` say so; ``elapsed_seconds`` is the
  chunk's, repeated across its members. Do not sum this column.
* **These rows were parsed from the run log after the fact** (``source`` =
  ``run_log``), because the timing requirement was noticed after the run had
  completed. ``score_rollouts.py`` now writes the same CSV at eval time.

    uv run python collect_timings.py --run /data/exp_contactseed/run \\
        --log /data/exp_contactseed/run_all.log --out data/timings.csv
"""

import argparse
import platform
import re
import socket
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import load_targets

CHUNK_RE = re.compile(
    r"^\[(?P<arm>[a-z0-9-]+)\] \[(?P<done>\d+)/(?P<total>\d+)\] "
    r"L=(?P<lo>\d+)-(?P<hi>\d+)\s+(?P<seconds>[\d.]+)s"
)
MODEL_NICKNAME = "contacts-v1-exp232-m2-p06-1.5B"


def gpu_metadata() -> dict:
    import torch

    properties = torch.cuda.get_device_properties(0)
    return dict(
        gpu_name=properties.name,
        gpu_total_memory_gb=round(properties.total_memory / 2**30, 2),
        gpu_compute_capability=f"{properties.major}.{properties.minor}",
        torch_version=torch.__version__,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--log", type=Path, nargs="+", required=True,
                    help="one or more score_rollouts.py run logs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()

    # `score_rollouts.py` walks the targets in ascending-length order, so
    # replaying that order against the chunk log assigns each protein to the
    # chunk it actually ran in.
    targets = load_targets()
    chunks: dict[str, list[float]] = {}
    for log in args.log:
        for line in log.read_text(errors="replace").splitlines():
            match = CHUNK_RE.match(line)
            if match:
                chunks.setdefault(match["arm"], []).append(float(match["seconds"]))
    assert chunks, f"no chunk lines matched in {args.log}"

    meta = gpu_metadata()
    stamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for arm, seconds in chunks.items():
        expected = -(-len(targets) // args.chunk)
        assert len(seconds) == expected, (
            f"{arm}: parsed {len(seconds)} chunk lines, expected {expected}"
        )
        for index, target in enumerate(targets):
            rows.append(dict(
                stem=target.stem, n_residues=target.L,
                n_pairs=target.L * (target.L - 1) // 2, mode=arm,
                elapsed_seconds=seconds[index // args.chunk],
                timing_unit="chunk", chunk_size=args.chunk,
                n_rollouts=args.n_rollouts,
                model_nickname=MODEL_NICKNAME, runner_tag="local",
                hostname=socket.gethostname(), platform=platform.platform(),
                source="run_log", timestamp_utc=stamp, **meta,
            ))

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    per_arm = frame.groupby("mode")["elapsed_seconds"].agg(["sum", "size"])
    print(f"[timings] wrote {len(frame)} rows -> {args.out}")
    print("[timings] chunk seconds are shared within a chunk; per-arm totals "
          "below divide by chunk_size to avoid the obvious double count:")
    print((per_arm["sum"] / args.chunk / 60).round(1).rename("arm_minutes").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

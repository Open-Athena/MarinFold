# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull a fan-out's prediction shards back from CoreWeave object storage.

Each shard wrote ``<dataset>/<stem>.pdb`` plus its own ``stats_shard*.jsonl``
and ``timings_shard*.jsonl`` under one prefix, so collecting a run is a mirror
plus a concatenation of the per-shard sidecars.

Reports how many of the 554 records arrived. A shard that was preempted and
never resumed shows up here as missing records rather than as a silently
smaller mean — the scorer counts a missing prediction as a total miss, so an
incomplete collection would understate the model rather than flatter it, but
you still want to know before reading the numbers.

Usage::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    uv run python collect_cw.py --tag f-cc1mix5-step50000 --out-dir _scratch/pred
"""

import argparse
import json
import os
from pathlib import Path

import fsspec

S3_PREFIX = os.environ.get("EXP174_S3", "s3://marin-us-east-02a/MarinFold/exp174")


def storage_options() -> dict:
    return {
        "key": os.environ.get("AWS_ACCESS_KEY_ID"),
        "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "endpoint_url": os.environ.get("AWS_ENDPOINT_URL", "https://cwobject.com"),
        "config_kwargs": {"s3": {"addressing_style": "virtual"}},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="prediction prefix, e.g. f-cc1mix5-step50000")
    ap.add_argument("--out-dir", type=Path, required=True, help="local root; <tag> is appended")
    ap.add_argument("--gt-index", type=Path, default=Path("_scratch/gt/gt_index.jsonl"))
    args = ap.parse_args(argv)

    fs = fsspec.filesystem("s3", **storage_options())
    remote_root = f"{S3_PREFIX}/pred/{args.tag}".replace("s3://", "")
    local_root = args.out_dir / args.tag
    local_root.mkdir(parents=True, exist_ok=True)

    n_pdb = 0
    stats: list[dict] = []
    timings: list[dict] = []
    for remote in fs.find(remote_root):
        name = remote[len(remote_root) + 1 :]
        local = local_root / name
        local.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(remote, str(local))
        if name.endswith(".pdb"):
            n_pdb += 1
        elif name.startswith("stats_shard"):
            stats += [json.loads(line) for line in local.open()]
        elif name.startswith("timings_shard"):
            timings += [json.loads(line) for line in local.open()]

    for rows, filename in ((stats, "stats.jsonl"), (timings, "timings.jsonl")):
        if rows:
            with (local_root / filename).open("w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")

    expected = sum(1 for _ in args.gt_index.open()) if args.gt_index.exists() else None
    print(f"[collect] {args.tag}: {n_pdb} structures, {len(stats)} stats rows -> {local_root}")
    if expected is not None:
        missing = expected - n_pdb
        marker = "" if missing == 0 else f"  ** {missing} MISSING **"
        print(f"[collect] {n_pdb}/{expected} records present{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

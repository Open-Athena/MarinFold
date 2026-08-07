# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload a prepared bf16 checkpoint dir to CoreWeave object storage.

The eval workers stage their model from S3 (`score_rollout_worker_cw.py`), and
in-cluster reads run at ~500 MB/s, so this one-time upload from the workstation
is the critical path for the whole experiment — hence the per-file rate
reporting, and hence `prepare_hf_export.py` recasting to bf16 first.

CoreWeave rejects path-style S3 addressing, so the endpoint is driven with
virtual-hosted addressing. Credentials come from `~/.config/marin/cw-rno2a.env`.

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    uv run python stage_model_s3.py --src <dir> --dst s3://marin-us-east-02a/MarinFold/...
"""

import argparse
import os
import time
from pathlib import Path

import s3fs

ENDPOINT = "https://cwobject.com"


def filesystem() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        key=os.environ["CW_KEY_ID"],
        secret=os.environ["CW_KEY_SECRET"],
        # `client_kwargs`, not a bare `endpoint_url=`: s3fs forwards unknown
        # kwargs to `botocore.session.Session`, which rejects it.
        client_kwargs={"endpoint_url": ENDPOINT},
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", required=True, help="s3://bucket/prefix (no trailing slash)")
    a = ap.parse_args()

    fs = filesystem()
    files = sorted(p for p in a.src.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"no files under {a.src}")
    dst = a.dst.rstrip("/")

    t_all = time.time()
    total = 0
    for path in files:
        size = path.stat().st_size
        t0 = time.time()
        fs.put_file(str(path), f"{dst}/{path.name}")
        dt = max(time.time() - t0, 1e-6)
        total += size
        print(f"[stage] {path.name}: {size / 2**20:8.1f} MiB in {dt:6.1f}s "
              f"({size / dt / 2**20:5.1f} MiB/s)", flush=True)

    elapsed = time.time() - t_all
    print(f"[stage] {len(files)} files, {total / 2**30:.2f} GiB in {elapsed / 60:.1f} min "
          f"({total / elapsed / 2**20:.1f} MiB/s) -> {dst}")

    remote = {os.path.basename(f) for f in fs.ls(dst, detail=False)}
    missing = {p.name for p in files} - remote
    if missing:
        raise SystemExit(f"!! upload incomplete, missing on S3: {sorted(missing)}")
    print(f"[stage] verified {len(remote)} objects at {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

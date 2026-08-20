# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage exp120's Levanter checkpoint GCS -> HF bucket, cloud-side (#160).

Levanter continue-training needs a full **training-state** checkpoint dir, not
the HF export. exp120's lives on GCS (16.4 GiB), and CoreWeave pods cannot read
GCS — so the checkpoint has to move somewhere both sides can see. The HF bucket
is exactly that: GCP can write it, CoreWeave already reads it (#159 pulled the
model and corpus from there on 48 workers).

Run this **on a marin CPU iris job**, not the workstation: the pod is GCS-local,
so the 16.4 GiB read is fast and only the upload crosses the internet — versus
~2 h each way over the workstation's ~2.5 MB/s uplink.

Destination follows the checkpoint-naming convention
(`checkpoints/<wandb-run-name>/...`), alongside exp120's existing `hf/` export:

    checkpoints/exp120-cv1-1_5b-orig-lr3e-4-e1-cos/levanter/step-1005/

    # on a marin CPU job (needs disk >= 24GB and HF_TOKEN):
    uv run --with gcsfs --with 'huggingface_hub[cli]>=1.5' \\
        python stage_checkpoint.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

SRC = (
    "gs://marin-us-east5/protein-structure/MarinFold/"
    "exp120_regen_vs_reepoch_contacts_v1/checkpoints/"
    "exp120-cv1-1_5b-orig-lr3e-4-e1-cos-fb79f7/checkpoints/step-1005"
)
# CoreWeave S3, written directly from the GCS-local pod. Originally routed via
# the HF bucket, but its uploader panics on multi-GB files ("File #10 ... is not
# fully completed: 2257162240/2261618688 bytes") — this checkpoint has a 2.26 GB
# shard. boto3 multipart handles it, it is one hop instead of two, and the
# training job then reads from CoreWeave-local storage.
DST = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp160_backtracking_training/init/exp120-step-1005"
)


def download(src: str, local: Path) -> int:
    """Mirror the GCS checkpoint dir to local disk; return bytes copied."""
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    prefix = src[len("gs://"):]
    files = [f for f in fs.find(prefix) if not f.endswith("/")]
    print(f"{len(files)} objects to copy", flush=True)
    total = 0
    for i, remote in enumerate(files, 1):
        rel = remote[len(prefix):].lstrip("/")
        dest = local / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        fs.get(remote, str(dest))
        total += dest.stat().st_size
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  ({total / 1e9:.1f} GB)", flush=True)
    return total


def upload_s3(local: Path, dst: str) -> None:
    """Upload the checkpoint dir to CoreWeave S3 (multipart, virtual-hosted).

    Needs CW_KEY_ID / CW_KEY_SECRET in the environment — CoreWeave credentials
    are not auto-injected into a *marin* pod the way they are on cw-* clusters.
    """
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    bucket, prefix = dst[len("s3://"):].split("/", 1)
    s3 = boto3.client(
        "s3",
        endpoint_url="https://cwobject.com",
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"}),
        region_name="auto",
    )
    # 64 MB parts: the largest shard here is ~2.3 GB, far past the 5 GB
    # single-PUT ceiling only in aggregate, but multipart also survives a
    # mid-transfer hiccup.
    transfer = TransferConfig(multipart_threshold=64 * 1024**2,
                              multipart_chunksize=64 * 1024**2,
                              max_concurrency=8)
    files = [p for p in local.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    done = 0
    print(f"uploading {len(files)} files ({total / 1e9:.1f} GB) -> {dst}", flush=True)
    for i, path in enumerate(sorted(files), 1):
        key = f"{prefix.rstrip('/')}/{path.relative_to(local).as_posix()}"
        s3.upload_file(str(path), bucket, key, Config=transfer)
        done += path.stat().st_size
        print(f"  {i}/{len(files)}  {done / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--local", type=Path, default=Path("/tmp/exp120_ckpt"))
    ap.add_argument("--hf", default="hf")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    if not args.skip_download:
        args.local.mkdir(parents=True, exist_ok=True)
        size = download(args.src, args.local)
        print(f"downloaded {size / 1e9:.2f} GB -> {args.local}", flush=True)

    if args.dst.startswith("s3://"):
        upload_s3(args.local, args.dst)
    else:
        cmd = [
            "uv", "run", "--no-project",
            "--with", "huggingface_hub[cli]>=1.6,<2",
            "hf", "buckets", "sync", str(args.local), args.dst,
        ]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env={**os.environ})
    print(f"staged -> {args.dst}", flush=True)


if __name__ == "__main__":
    main()

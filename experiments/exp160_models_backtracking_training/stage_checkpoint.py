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
DST = (
    "hf://buckets/open-athena/MarinFold/checkpoints/"
    "exp120-cv1-1_5b-orig-lr3e-4-e1-cos/levanter/step-1005"
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

    # The project env pins huggingface_hub<1.0 (marinfold -> transformers), and
    # the `buckets` subcommand needs >=1.5. Run the upload in an ISOLATED env
    # rather than fighting the pin — exp139's documented workaround.
    cmd = [
        "uv", "run", "--no-project",
        "--with", "huggingface_hub[cli]>=1.6,<2",
        "hf", "buckets", "sync", str(args.local), args.dst,
    ] if args.hf == "hf" else [args.hf, "buckets", "sync", str(args.local), args.dst]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ})
    print(f"staged -> {args.dst}", flush=True)


if __name__ == "__main__":
    main()

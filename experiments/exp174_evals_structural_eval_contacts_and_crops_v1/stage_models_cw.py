# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the HF exports to bf16 and stage them to CoreWeave object storage.

The workstation uplink is ~2.2 MB/s, so staging is the critical path for a
CoreWeave run and halving the bytes halves the wait. The exports are fp32
(5.5 GB each); inference loads them in bf16 anyway, so casting first is lossless
with respect to what actually runs and takes each model to ~2.8 GB. It also
keeps the whole transfer under the 10 GB threshold the root ``AGENTS.md`` puts
a sign-off gate on.

Reads the GCS copies already downloaded to ``--models-dir`` (no re-download) and
writes to ``s3://marin-us-east-02a/MarinFold/exp174/models/<label>/``.

Run with the CoreWeave credentials sourced::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    uv run python stage_models_cw.py --models-dir _scratch/models
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import fsspec
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

S3_PREFIX = "s3://marin-us-east-02a/MarinFold/exp174/models"

# CoreWeave object storage rejects path-style addressing.
S3_STORAGE_OPTIONS = {
    "key": os.environ.get("AWS_ACCESS_KEY_ID"),
    "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    "endpoint_url": os.environ.get("AWS_ENDPOINT_URL", "https://cwobject.com"),
    "config_kwargs": {"s3": {"addressing_style": "virtual"}},
}


def to_bf16(src: Path, dst: Path) -> Path:
    """Rewrite an fp32 HF export as bf16, tokenizer co-located."""
    if dst.exists():
        shutil.rmtree(dst)
    model = AutoModelForCausalLM.from_pretrained(str(src), dtype=torch.bfloat16)
    model.save_pretrained(str(dst))
    # Root AGENTS.md hard rule: a model is unloadable without its tokenizer.
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(dst))
    return dst


def upload(local: Path, s3_dir: str) -> None:
    """Copy every file of a local model directory to ``s3_dir``."""
    fs = fsspec.filesystem("s3", **S3_STORAGE_OPTIONS)
    for path in sorted(local.iterdir()):
        if not path.is_file():
            continue
        target = f"{s3_dir.rstrip('/')}/{path.name}"
        size_mb = path.stat().st_size / 1e6
        started = time.time()
        fs.put_file(str(path), target)
        elapsed = time.time() - started
        print(
            f"  {path.name}: {size_mb:.0f} MB in {elapsed:.0f}s "
            f"({size_mb / max(elapsed, 1e-6):.1f} MB/s)",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument(
        "--label",
        action="append",
        default=None,
        help="model subdirectory to stage; repeatable (default: both)",
    )
    ap.add_argument("--s3-prefix", default=S3_PREFIX)
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args(argv)

    if not S3_STORAGE_OPTIONS["key"]:
        raise SystemExit(
            "AWS_ACCESS_KEY_ID unset — run:\n"
            "  set -a; source ~/.config/marin/cw-rno2a.env; set +a"
        )

    labels = args.label or ["cc1mix5-step50000", "3way-step20000"]
    for label in labels:
        src = args.models_dir / label
        dst = args.models_dir / f"{label}-bf16"
        print(f"[stage] {label}: casting to bf16", flush=True)
        to_bf16(src, dst)
        total = sum(p.stat().st_size for p in dst.iterdir() if p.is_file())
        print(f"[stage] {label}: {total / 1e9:.2f} GB bf16", flush=True)
        if args.skip_upload:
            continue
        print(f"[stage] {label}: uploading -> {args.s3_prefix}/{label}", flush=True)
        upload(dst, f"{args.s3_prefix}/{label}")
    print("[stage] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

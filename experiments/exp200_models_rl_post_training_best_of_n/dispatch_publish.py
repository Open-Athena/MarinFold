# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish exp163's arm-F checkpoint to an HF model repo — issue #200.

Cloud-side because the workstation uplink is ~2.5 MB/s and this moves ~2.9 GB.
The HF token travels as a job env var; it needs `repo.write` on the target
namespace, which the workstation's ACTIVE token does not have — the stored
`write2` token does (the active `oa-marinfold` one can write buckets but cannot
create repos).

    uv run python dispatch_publish.py
    uv run python dispatch_publish.py --dry-run
"""

import argparse
import configparser
from pathlib import Path

from _submit import check_clean, submit

SRC = ("hf://buckets/open-athena/MarinFold/checkpoints/"
       "plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF/hf/step-404")
REPO = "timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404"
STORED_TOKENS = Path.home() / ".cache/huggingface/stored_tokens"


def read_token(name: str) -> str:
    parser = configparser.ConfigParser()
    parser.read(STORED_TOKENS)
    if name not in parser:
        raise SystemExit(f"no token named {name!r} in {STORED_TOKENS} (have: {parser.sections()})")
    return list(dict(parser[name]).values())[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--token-name", default="write2",
                    help="section in ~/.cache/huggingface/stored_tokens")
    ap.add_argument("--region", default="us-east5")
    ap.add_argument("--job-name", default="exp200-publish-ckpt")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.dry_run:
        check_clean()

    submit(
        job_name=a.job_name,
        extras=("cpu",),
        cpu=8, memory="32GB",
        # ~2.9 GB staged locally before upload, plus the venv.
        disk="64GB",
        region=a.region,
        env={"HF_TOKEN": read_token(a.token_name)} if not a.dry_run else {},
        command=["python", "publish_checkpoint_hf.py", "--src", a.src, "--repo", a.repo],
        dry_run=a.dry_run,
    )
    print(f"  repo: https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

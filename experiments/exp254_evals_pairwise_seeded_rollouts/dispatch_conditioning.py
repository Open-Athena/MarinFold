# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a small immutable workspace and submit independent CoreWeave shards."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BOOTSTRAP = r"""#!/bin/bash
set -euo pipefail
VLLM_PY=$(command -v python3)
export PATH="$HOME/.local/bin:$PATH"
uv pip install --python "$VLLM_PY" --no-deps \
  fsspec==2026.6.0 s3fs==2026.6.0 aiobotocore==3.9.1 botocore==1.43.75 \
  aioitertools==0.13.0 wrapt==1.17.0 aiohttp==3.14.3 frozenlist==1.7.0 \
  multidict==6.6.3 yarl==1.20.1 aiosignal==1.4.0 attrs==23.2.0 \
  propcache==0.3.2 jmespath==1.0.1 python-dateutil==2.9.0.post0 six==1.17.0 \
  pandas==2.3.0 pyarrow==19.0.1 gemmi==0.6.5 aiohappyeyeballs==2.6.1 \
  transformers==5.15.0
export PYTHONPATH="$PWD/library${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PORT=$(uv run --no-project --no-sync "$VLLM_PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
exec uv run --no-project --no-sync "$VLLM_PY" conditioning_worker.py --plan plan.json "$@"
"""


def prepare(inputs: Path, workspace: Path) -> None:
    """Copy just the executable protocol, inputs, and document builder."""
    workspace.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    for name in (
        "conditioning_worker.py",
        "common.py",
        "rank_pairwise.py",
        "analyze_conditioning.py",
        "CONDITIONING_PROTOCOL.md",
    ):
        shutil.copy2(here / name, workspace / name)
    for name in ("plan.json", "source_votes.npz"):
        shutil.copy2(inputs / name, workspace / name)
    shutil.copytree(
        here.parents[1] / "marinfold" / "marinfold",
        workspace / "library" / "marinfold",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (workspace / "bootstrap.sh").write_text(BOOTSTRAP)
    manifest = {
        str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(workspace.rglob("*"))
        if path.is_file()
    }
    (workspace / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    """Submit a smoke target or a complete interleaved batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--iris", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--stem")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("At least one shard is required")
    if not args.workspace.exists():
        prepare(args.inputs, args.workspace)
    else:
        manifest = json.loads((args.workspace / "bundle_manifest.json").read_text())
        for name, expected in manifest.items():
            if (
                hashlib.sha256((args.workspace / name).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError(f"Workspace changed after preparation: {name}")
    if args.prepare_only:
        return
    for shard in range(1 if args.stem else args.shards):
        command = [
            args.iris,
            "--cluster=marin",
            "job",
            "run",
            "--target-cluster",
            "cw-us-east-02a",
            "--priority",
            "batch",
            "--gpu",
            "H100x1",
            "--enable-extra-resources",
            "--cpu",
            "8",
            "--memory",
            "64GB",
            "--disk",
            "128GB",
            "--task-image",
            "vllm/vllm-openai:v0.19.1",
            "--no-sync",
            "--max-retries",
            "2",
            "--timeout",
            "14400",
            "--no-wait",
            "--job-name",
            f"{args.run_name}-{'smoke' if args.stem else shard}",
            "--",
            "bash",
            "bootstrap.sh",
            "--out",
            args.out,
            "--shard",
            "0/1" if args.stem else f"{shard}/{args.shards}",
        ]
        if args.stem:
            command.extend(["--stem", args.stem])
        subprocess.run(command, cwd=args.workspace, check=True)


if __name__ == "__main__":
    main()

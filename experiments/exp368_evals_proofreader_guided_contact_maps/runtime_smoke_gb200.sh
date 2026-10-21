#!/usr/bin/env bash
set -euo pipefail
unset FSSPEC_S3_CONFIG_KWARGS
export UV_LINK_MODE=copy TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export EXP356_S3_ENDPOINT=https://cwobject.com
# PyPI's ARM build is CPU-only. Keep the experiment's other locked packages,
# then install the hash-pinned official CUDA build of the same torch version.
uv sync --locked --no-install-package torch
uv pip install --python .venv/bin/python 'https://download-r2.pytorch.org/whl/cu130/torch-2.10.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=4fc8f67637f4c92b989a07d80ffe755e79a3510ca02ebf23ce66396fb277c88d'
exec uv run --no-sync python runtime_smoke_gb200.py "$@"

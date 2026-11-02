#!/usr/bin/env bash
set -euo pipefail
unset FSSPEC_S3_CONFIG_KWARGS
export UV_LINK_MODE=copy TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export EXP356_S3_ENDPOINT=https://cwobject.com
# PyPI's ARM build is CPU-only. Keep the experiment's other locked packages,
# then install the hash-pinned official CUDA build of the same torch version.
uv sync --locked --no-install-package torch
uv pip install --python .venv/bin/python 'https://download-r2.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=6f09cdf2415516be028ae82e6b985bcfc3eac37bc52ab401142689f6224516ca'
if [ "${1:-}" = "--recover" ]; then
  shift
  exec uv run --no-sync python recover_task.py "$@"
fi
if [ "${1:-}" = "--runtime-smoke" ]; then
  shift
  exec uv run --no-sync python runtime_smoke_gb200.py "$@"
fi
if [ "${1:-}" = "--backend-check" ]; then
  shift
  exec uv run --no-sync python backend_check.py "$@"
fi
echo 'Expected --recover, --runtime-smoke, or --backend-check'
exit 2

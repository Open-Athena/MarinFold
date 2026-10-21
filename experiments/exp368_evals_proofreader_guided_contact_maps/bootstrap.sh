#!/usr/bin/env bash
set -euo pipefail
unset FSSPEC_S3_CONFIG_KWARGS
export UV_LINK_MODE=copy TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export EXP356_S3_ENDPOINT=https://cwobject.com
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv sync --locked
if [ "${1:-}" = "--recover" ]; then
  shift
  exec uv run --no-sync python recover_task.py "$@"
fi
if [ "${1:-}" = "--serve" ]; then
  shift
  exec uv run --no-sync python serve.py "$@"
fi
if [ "${1:-}" = "--score" ]; then
  shift
  exec uv run --no-sync python score_pool.py "$@"
fi
exec uv run --no-sync python worker.py "$@"

#!/usr/bin/env bash
set -euo pipefail
unset FSSPEC_S3_CONFIG_KWARGS
export UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=MarinFold
export WANDB_ENTITY=open-athena
uv sync --locked --extra train
entrypoint=train.py
if [[ "${1:-}" == "--evaluate" ]]; then entrypoint=evaluate.py; shift; fi
if [[ "${1:-}" == "--collect" ]]; then
    shift
    exec uv run --no-sync python collect.py "$@"
fi
exec uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node="${PROOFREADER_GPUS:-1}" "$entrypoint" "$@"

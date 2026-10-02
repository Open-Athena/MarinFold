#!/usr/bin/env bash
set -euo pipefail
export UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MAX_JOBS=8
export WANDB_PROJECT=MarinFold
export WANDB_ENTITY=open-athena
uv sync --locked --extra train
uv pip install --python .venv/bin/python --no-build-isolation 'causal-conv1d==1.6.0'
exec uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=8 train.py "$@"

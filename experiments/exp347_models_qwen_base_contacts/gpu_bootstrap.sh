#!/usr/bin/env bash
set -euo pipefail
export UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MAX_JOBS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=MarinFold
export WANDB_ENTITY=open-athena
uv sync --locked --extra train
uv pip install --python .venv/bin/python --no-deps 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7'
entrypoint=train.py
if [[ "${1:-}" == "--rollout" ]]; then
  entrypoint=rollout_validation.py
  shift
fi
exec uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=8 "$entrypoint" "$@"

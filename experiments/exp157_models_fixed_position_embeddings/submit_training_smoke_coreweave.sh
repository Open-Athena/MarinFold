#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MARIN_WORKTREE="${MARIN_WORKTREE:-${ROOT}/../repos/marin-gamma}"
IRIS="${IRIS:-${MARIN_WORKTREE}/.venv/bin/iris}"
MAIN_CONFIG="${MAIN_CONFIG:-${MARIN_WORKTREE}/lib/iris/config/marin.yaml}"
TARGET_CLUSTER="${TARGET_CLUSTER:-cw-rno2a}"
JOB_NAME="${JOB_NAME:-exp157-fixed-position-training-smoke-r1-driver}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY must be set for the training smoke" >&2
  exit 2
fi

"${IRIS}" --config "${MAIN_CONFIG}" job run \
    --target-cluster "${TARGET_CLUSTER}" \
    --no-wait \
    --priority batch \
    --enable-extra-resources \
    --cpu=4 \
    --memory=16GB \
    --disk=32GB \
    --extra cpu \
    --job-name "${JOB_NAME}" \
    -e WANDB_API_KEY "${WANDB_API_KEY}" \
    -e EXP157_TRAIN_BATCH "${EXP157_TRAIN_BATCH:-16}" \
    -e EXP157_MODEL_SIZE "${EXP157_MODEL_SIZE:-1_5b}" \
    -e EXP157_GPU_COUNT "${EXP157_GPU_COUNT:-8}" \
    -e EXP157_MAX_STEPS "${EXP157_MAX_STEPS:-20}" \
    -e EXP157_STEPS_PER_EVAL "${EXP157_STEPS_PER_EVAL:-20}" \
    -e EXP157_MAX_EVAL_BATCHES "${EXP157_MAX_EVAL_BATCHES:-1}" \
    -e EXP157_RUN_SUFFIX "${EXP157_RUN_SUFFIX:-smoke20-r1}" \
    -e EXP157_ATTN "${EXP157_ATTN:-jax_flash}" \
    -e TF_GPU_ALLOCATOR "${TF_GPU_ALLOCATOR:-cuda_malloc_async}" \
    -e XLA_PYTHON_CLIENT_PREALLOCATE "${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION "${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" \
    -- python train_fixed_position_smoke.py

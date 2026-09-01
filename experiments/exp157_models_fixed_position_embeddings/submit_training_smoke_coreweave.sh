#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MARIN_WORKTREE="${MARIN_WORKTREE:-${ROOT}/../repos/marin-alpha}"
IRIS="${IRIS:-uv run --project /tmp/marin-iris-origin-main-fresh/lib/iris iris}"
IRIS_CONTROLLER_URL="${IRIS_CONTROLLER_URL:-http://10.128.0.3:10000}"
TARGET_CLUSTER="${TARGET_CLUSTER:-cw-rno2a}"
JOB_NAME="${JOB_NAME:-exp157-fixed-position-training-smoke-r1-driver}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY must be set for the training smoke" >&2
  exit 2
fi

${IRIS} --controller-url "${IRIS_CONTROLLER_URL}" job run \
    --target-cluster "${TARGET_CLUSTER}" \
    --no-wait \
    --priority "${IRIS_PRIORITY:-batch}" \
    --enable-extra-resources \
    --cpu=4 \
    --memory=16GB \
    --disk=32GB \
    --extra cpu \
    --job-name "${JOB_NAME}" \
    -e WANDB_API_KEY "${WANDB_API_KEY}" \
    -e EXP157_TRAIN_BATCH "${EXP157_TRAIN_BATCH:-16}" \
    -e EXP157_MODEL_SIZE "${EXP157_MODEL_SIZE:-1_5b}" \
    -e EXP157_MODEL_FAMILY "${EXP157_MODEL_FAMILY:-llama}" \
    -e EXP157_POSITION_MODE "${EXP157_POSITION_MODE:-fixed}" \
    -e EXP157_POSITION_DELTA_L2_WEIGHT "${EXP157_POSITION_DELTA_L2_WEIGHT:-0.0}" \
    -e EXP157_GPU_VARIANT "${EXP157_GPU_VARIANT:-H100}" \
    -e EXP157_GPU_COUNT "${EXP157_GPU_COUNT:-8}" \
    -e EXP157_GPU_REPLICAS "${EXP157_GPU_REPLICAS:-1}" \
    -e EXP157_TARGET_CLUSTER "${EXP157_TARGET_CLUSTER:-${TARGET_CLUSTER}}" \
    -e EXP157_CPU "${EXP157_CPU:-}" \
    -e EXP157_RAM "${EXP157_RAM:-}" \
    -e EXP157_DISK "${EXP157_DISK:-}" \
    -e EXP157_MAX_STEPS "${EXP157_MAX_STEPS:-20}" \
    -e EXP157_STEPS_PER_EVAL "${EXP157_STEPS_PER_EVAL:-20}" \
    -e EXP157_MAX_EVAL_BATCHES "${EXP157_MAX_EVAL_BATCHES:-1}" \
    -e EXP157_INITIALIZE_FROM_CHECKPOINT_PATH "${EXP157_INITIALIZE_FROM_CHECKPOINT_PATH:-}" \
    -e EXP157_IRIS_PRIORITY "${EXP157_IRIS_PRIORITY:-batch}" \
    -e EXP157_RUN_SUFFIX "${EXP157_RUN_SUFFIX:-smoke20-r1}" \
    -e EXP157_LR "${EXP157_LR:-3.162e-3}" \
    -e EXP157_WEIGHT_DECAY "${EXP157_WEIGHT_DECAY:-0.2}" \
    -e EXP157_WARMUP "${EXP157_WARMUP:-0.1}" \
    -e EXP157_ATTN "${EXP157_ATTN:-jax_flash}" \
    -e TF_GPU_ALLOCATOR "${TF_GPU_ALLOCATOR:-cuda_malloc_async}" \
    -e XLA_PYTHON_CLIENT_PREALLOCATE "${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION "${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" \
    -- python train_fixed_position_smoke.py

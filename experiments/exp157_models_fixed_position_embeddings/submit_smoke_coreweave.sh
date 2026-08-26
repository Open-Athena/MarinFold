#!/usr/bin/env bash
set -euo pipefail

# Submit the fixed-position embedding smoke test to CoreWeave via Iris.
#
# Submit through the main Iris controller and federate to CoreWeave. Direct
# CoreWeave kubeconfig submission is not the normal path for these jobs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MARIN_WORKTREE="${MARIN_WORKTREE:-${ROOT}/../repos/marin-gamma}"
IRIS="${IRIS:-${MARIN_WORKTREE}/.venv/bin/iris}"
MAIN_CONFIG="${MAIN_CONFIG:-${MARIN_WORKTREE}/lib/iris/config/marin.yaml}"
TARGET_CLUSTER="${TARGET_CLUSTER:-cw-rno2a}"
JOB_NAME="${JOB_NAME:-exp157-fixed-position-smoke-r3}"

"${IRIS}" --config "${MAIN_CONFIG}" job run \
    --target-cluster "${TARGET_CLUSTER}" \
    --no-wait \
    --priority batch \
    --enable-extra-resources \
    --cpu=2 \
    --memory=8GB \
    --disk=20GB \
    --extra cpu \
    --job-name "${JOB_NAME}" \
    -- python -m pytest -q

#!/bin/bash
# Fan the rollout+resample contact eval out over CoreWeave for the three
# checkpoints issue #169 selected. Three models x 12 single-H100 batch-band
# shards; the 554-protein set takes ~5 min of wall clock per checkpoint.
#
# The dispatcher, worker and metric code are exp82's, unchanged — that is the
# point. exp169 supplies only the checkpoints, the S3 prefix and the job name,
# so these numbers sit on exactly the scale as the published #75 / #117 rows.
#
# The #117 final checkpoint is re-scored here rather than reusing exp167's
# published matrices: the headline comparison is a 0.0076-nat loss difference
# between two #117 checkpoints, which deserves to be measured in one submission
# — and the reproduction of the published 0.535 is itself the harness check.
set -euo pipefail
set -a; source ~/.config/marin/cw-rno2a.env; set +a

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISPATCH_DIR="$REPO_ROOT/experiments/exp82_evals_contacts_v1_contact_prediction"
# The fresh marin checkout, not this experiment's venv: iris rejects submissions
# from a client older than 14 days, and only marin ships fray/iris at all.
PY=/home/bizon/git/marin-freshiris/.venv/bin/python

S3=s3://marin-us-east-02a/MarinFold/exp169_eval

export EVAL_CW_JOB_PREFIX=exp169-rolleval
export EVAL_CW_S3_PREFIX="$S3"
# Byte-identical to the targets exp167 scored (verified by sha256 at copy time),
# so the candidate-pair universe is the published one.
export EVAL_CW_TARGETS="$S3/eval_targets.parquet"
export EVAL_CW_OUT="$S3/scores"
export EVAL_CW_NUM_SHARDS="${EVAL_CW_NUM_SHARDS:-12}"
# The settled recipe (exp82, 2026-07-27): 100 resampled rollouts, top-k OFF.
export EVAL_CW_N_ROLLOUTS=100
export EVAL_CW_TOP_K=-1
export EVAL_CW_TOP_P=0.95
export EVAL_CW_TEMPERATURE=1.0
export EVAL_CW_NO_WAIT=1          # shards are root jobs; they outlive this script

cd "$DISPATCH_DIR"
KUBECONFIG=~/.kube/coreweave-iris-rno2a "$PY" dispatch_rollout_eval_cw.py \
    --model "exp117_e16_final_step35679=s3://marin-us-east-02a/MarinFold/exp167_eval/model_exp117_bs256_step35679" \
    --model "exp117_e16_early_step33450=$S3/model_exp117_step33450" \
    --model "exp146_3b_e8_step17839=$S3/model_exp146_3b_step17839" \
    "$@"

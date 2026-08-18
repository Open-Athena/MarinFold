#!/usr/bin/env bash
# Arm M-K — reward the rollout for the number the metric reports. Issue #237.
#
# The last gap in the design space, and the simplest thing in it: one scalar per
# rollout, its OWN consensus R-precision, with a GRPO group baseline. No
# per-section machinery, no shaping term, no lambda.
#
# It has appeared only as a lambda-weighted INGREDIENT in M-BC and M-FC, where a
# second objective confounds it. Standalone it has never been run — which is an
# oversight, because it is the only reward measured here to be SCALE-CORRECT in
# the section count:
#
#     group-centred advantage    1 section    22 sections
#     M-C's per-section marginal    +4.79          -0.22   <- pathological
#     causal prefix marginal        +2.03          -0.22   <- also pathological
#     C_i(all), this arm            -1.37          +0.79   <- correct
#
# Dropping sections lowers your own consensus (0.543 at 22, 0.341 at one), so the
# direction that destroyed M-C is penalised rather than paid.
#
# The preregistered risk: this is the structural analogue of #208's arm D
# (document F1 + GRPO), which lost 61 % of its vote coverage. The difference is
# that a CONSENSUS over a rollout's own sections rewards complementary drafts,
# where document F1 had no reason to keep them different. Whether that is enough
# is exactly the untested question, and `union/R` plus `mean_jaccard` are the
# columns that answer it.
set -u

LR=${LR:-1e-5}
STEPS=${STEPS:-48}
CKPT_EVERY=${CKPT_EVERY:-6}
EVAL_STEPS=${EVAL_STEPS:-"12 18 24 36"}
ROOT=${ROOT:-$HOME/exp237_data_mk}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl

drain() {
  for i in $(seq 1 240); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
    [ "$busy" -eq 0 ] && break
    sleep 15
  done
  for i in $(seq 1 60); do pgrep -f "VLLM::EngineCore" >/dev/null || return 0; sleep 5; done
  return 1
}

mkdir -p "$ROOT"
cd "$HERE" || exit 1
echo "[mk] train M-K lr=$LR steps=$STEPS ckpt=$CKPT_EVERY at $(date -u)"
drain || { echo "[mk] cards never drained"; exit 1; }
ARM=M-K LR=$LR STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY ROOT=$ROOT \
  bash run_arm.sh >> "$LOGS/mk.log" 2>&1
echo "[mk] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  [ -d "$ROOT/ckpts_m_k/global_step_$st" ] || { echo "[mk] no ckpt $st"; continue; }
  echo "[mk] eval step $st at $(date -u)"
  drain && ARM=M-K STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/mk.log" 2>&1
  echo "[mk]   rc=$?"
done
echo "[mk] DONE at $(date -u)"

#!/usr/bin/env bash
# Arm M-KS2 — M-KS's shaping with the POSITIONAL correction. Issue #237.
#
#   A_i,k = GRPO_group(C_i(all))_i + beta * ( (m_k - b_k) - mean_k (m - b) )
#
# where m_k is the causal prefix marginal and b_k is the GROUP's mean marginal at
# the SAME section position k.
#
# **What went wrong in M-KS, measured.** The plain centred term
# `(m_k - mean_k m)` is not neutral in k. `C(s_1..s_k)` saturates, so the prefix
# marginal decays by construction, and the FIRST section -- scored against an
# empty prefix -- captures nearly the whole telescoped total. On 566 real
# rollouts:
#
#     k        0       1       2       4       8      12      20
#     term  +0.357  +0.013  -0.002  -0.018  -0.022  -0.021  -0.016
#
# with a negative slope in **100 %** of rollouts. At beta 3 that is +1.07 on the
# first section's tokens against a base of ~1 unit, and a small penalty on every
# section after the second.
#
# Zero-sum within the rollout bounds the *level* -- the rollout's total advantage
# cannot move with the section count -- but it leaves the *shape* untouched. And
# a section owns the `<begin_statements>` token that OPENS it, so a term
# decreasing in k is a direct penalty on the decision to write another candidate.
# **Arm M-KS collapsed to 10.66 sections by step 21, the fastest count collapse
# in #237, entirely inside its own zero-sum guarantee.** The guarantee was real
# and it was guarding the wrong thing.
#
# **The fix and its evidence.** Subtracting the group's marginal at the same
# position removes the deterministic trend and leaves only "did this section beat
# a typical k-th section?". Measured: residual slope in k goes from negative in
# 100 % of rollouts to a mean of -0.000002, negative in 50 % -- i.e. gone.
#
# **beta = 15, recalibrated and NOT carried over.** Removing the trend shrinks
# the term's spread 5.9x (sd 0.0987 -> 0.0167), so the weight that put the old
# term at a quarter of the GRPO base's unit spread puts the corrected one at
# 1/25th of it. Reusing beta 3 here would have produced a null that meant
# nothing -- #208's "it was not a weak signal, it was no signal", exactly.
# 15.0 restores the quarter-weight (calib2.py).
set -u

LR=${LR:-1e-5}
STEPS=${STEPS:-48}
CKPT_EVERY=${CKPT_EVERY:-12}
BETA=${BETA:-15.0}
ROOT=${ROOT:-$HOME/exp237_data_mks2}
EVAL_STEPS=${EVAL_STEPS:-"36 24 12 48"}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl
# Capture WAIT_PID with `ps -eo pid,args | grep <script> | grep -v "grep\|bash -c"`,
# never `pgrep -f` -- it matches its own wrapper and returns a pid that exits at once.
WAIT_PID=${WAIT_PID:-}
if [ -n "$WAIT_PID" ]; then
  echo "[mks2] waiting for pid $WAIT_PID at $(date -u)"
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 60; done
fi

drain() {
  for i in $(seq 1 240); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
    [ "$busy" -eq 0 ] && break
    sleep 15
  done
  for i in $(seq 1 60); do pgrep -f "VLLM::EngineCore" >/dev/null || { sleep 30; return 0; }; sleep 5; done
  return 1
}

mkdir -p "$ROOT"
cd "$HERE" || exit 1
echo "[mks2] train M-KS2 beta_shape=$BETA positional=true lr=$LR steps=$STEPS at $(date -u)"
drain || { echo "[mks2] cards never drained"; exit 1; }
ARM=M-KS LR=$LR STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY ROOT=$ROOT \
  BETA_SHAPE=$BETA POSITIONAL_SHAPE=true RUN_SUFFIX=_pos \
  bash run_arm.sh >> "$LOGS/mks2.log" 2>&1
echo "[mks2] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  if [ ! -d "$ROOT/ckpts_m_ks/global_step_$st" ]; then
    echo "[mks2] no checkpoint at step $st, skipping"; continue
  fi
  echo "[mks2] eval step $st at $(date -u)"
  drain && ARM=M-KS STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/mks2.log" 2>&1
  echo "[mks2]   rc=$? step=$st done at $(date -u)"
done
echo "[mks2] DONE at $(date -u)"

#!/usr/bin/env bash
# Arm M-KS3 — novelty scored DIRECTLY, not through a proxy. Issue #237.
#
#   A_i,k = GRPO_group(C_i(all))_i + beta * ( (m_k - b_k) - mean_k (m - b) )
#
# where m_k is the causal prefix marginal and b_k is the GROUP's mean marginal at
# the SAME section position k.
#
#   A_i,k = GRPO_group(C_i(all))_i + beta * ( (n_k - b_k) - mean_k (n - b) )
#
# with n_k = ( |new true| - |new false| ) / R against the prefix union, and b_k
# the group's mean at the same section POSITION.
#
# **Why.** Arm M-KS2 shaped on the causal consensus marginal, which was only a
# PROXY for "what did this section add": measured on 11,516 real sections it
# correlates **+0.194** with actual novelty. It nonetheless produced the best
# ORACLE candidates in #237 (best 0.5677 at step 24, +0.0074 over arm M-K with
# the CI excluding zero) at no cost to consensus (-0.0007, tie). A weak proxy
# already bought the experiment's best candidates; this asks what the signal it
# was approximating is worth measured directly. The two are genuinely different
# quantities -- they correlate only **+0.283** with each other.
#
# **Precision-aware by construction, and that is not optional.** Plain recall
# gain (|new true| / R) pays for VOLUME: a section that dumps a hundred junk
# pairs catches new true ones by chance and is paid for it. #237 watched that
# exact failure once, when arm M-F ran to 259 sections carrying 1.4 contacts
# each. Subtracting the new false pairs makes the term "did the union get better
# or worse", so padding is priced. Pinned by test.
#
# **Positional correction retained, for the reason M-KS died of.** The first
# section is scored against an empty prefix and so books its whole content; a
# term decaying in k penalises the decision to open another candidate, because a
# section owns the `<begin_statements>` token that opens it. Subtracting the
# group's marginal at the same position removes it -- residual slope negative in
# 52 % of rollouts here, against 100 % without.
#
# **beta = 7.0, recalibrated and NOT carried over from M-KS2.** The corrected
# novelty term has sd **0.0375** against M-KS2's 0.0167, so M-KS2's beta 15
# would put this at half again the GRPO base's whole spread. 6.7 restores the
# quarter-weight this line has used throughout (calib3.py); 7.0 rounds it.
set -u

LR=${LR:-1e-5}
STEPS=${STEPS:-48}
CKPT_EVERY=${CKPT_EVERY:-12}
BETA=${BETA:-7.0}
ROOT=${ROOT:-$HOME/exp237_data_mks3}
EVAL_STEPS=${EVAL_STEPS:-"36 24 12 48"}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl
# Capture WAIT_PID with `ps -eo pid,args | grep <script> | grep -v "grep\|bash -c"`,
# never `pgrep -f` -- it matches its own wrapper and returns a pid that exits at once.
WAIT_PID=${WAIT_PID:-}
if [ -n "$WAIT_PID" ]; then
  echo "[mks3] waiting for pid $WAIT_PID at $(date -u)"
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
echo "[mks3] train M-KS3 beta_shape=$BETA positional=true lr=$LR steps=$STEPS at $(date -u)"
drain || { echo "[mks3] cards never drained"; exit 1; }
ARM=M-KS LR=$LR STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY ROOT=$ROOT \
  BETA_SHAPE=$BETA POSITIONAL_SHAPE=true SHAPE_SIGNAL=novelty RUN_SUFFIX=_nov \
  bash run_arm.sh >> "$LOGS/mks3.log" 2>&1
echo "[mks3] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  if [ ! -d "$ROOT/ckpts_m_ks/global_step_$st" ]; then
    echo "[mks3] no checkpoint at step $st, skipping"; continue
  fi
  echo "[mks3] eval step $st at $(date -u)"
  drain && ARM=M-KS STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/mks3.log" 2>&1
  echo "[mks3]   rc=$? step=$st done at $(date -u)"
done
echo "[mks3] DONE at $(date -u)"

#!/usr/bin/env bash
# Arm M-B at lr 3e-6 — a slower walk through the window where it peaked. #237.
#
# **Why.** Both arms that improved anything peaked in KL 0.005-0.020, and M-B's
# best result (oracle-best 0.5574, +0.0232) came at KL 0.0163 -- near the TOP of
# that window, sampled once, by a run that checkpointed every 18 steps. At lr
# 1e-5 the policy crosses the whole useful window in about a dozen steps, so it
# was never really sampled. A third of the step size covers the same KL in ~3x
# the steps, which is the resolution the question needs.
#
# **Step budget.** M-B at 1e-5 read KL 0.0163 at step 36 and 0.0196 by step 37.
# KL grows roughly as (lr * steps)^2 on this run, so matching 0.0163 at 3e-6
# needs ~120 steps. Checkpoints every 15 give 8 points across the window.
#
# **Fresh ROOT, deliberately.** `resume_mode=latest` reads `trainer.ckpt_path`,
# and $HOME/exp237_data/ckpts_m_b now holds global_step_80 -- the destroyed
# policy (consensus 0.3969). Reusing that root would silently resume from it.
set -u

LR=${LR:-3e-6}
STEPS=${STEPS:-120}
CKPT_EVERY=${CKPT_EVERY:-15}
EVAL_STEPS=${EVAL_STEPS:-"120 90 60 30"}
ROOT=${ROOT:-$HOME/exp237_data_mb_lowlr}
WAIT_LOG=${WAIT_LOG:-$HOME/exp237_logs/mb_curve_driver.log}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl

if [ -n "${WAIT_LOG}" ] && [ "${WAIT_FOR:-1}" = "1" ]; then
  echo "[mblow] waiting for $WAIT_LOG to report DONE"
  until grep -q "^\[mb\] DONE" "$WAIT_LOG" 2>/dev/null; do sleep 60; done
fi

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
echo "[mblow] train M-B lr=$LR steps=$STEPS ckpt=$CKPT_EVERY root=$ROOT at $(date -u)"
drain || { echo "[mblow] cards never drained"; exit 1; }
ARM=M-B LR=$LR STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY ROOT=$ROOT \
  bash run_arm.sh >> "$LOGS/mb_lowlr.log" 2>&1
echo "[mblow] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  if [ ! -d "$ROOT/ckpts_m_b/global_step_$st" ]; then
    echo "[mblow] no checkpoint at step $st, skipping"; continue
  fi
  echo "[mblow] eval step $st at $(date -u)"
  drain && ARM=M-B STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/mb_lowlr.log" 2>&1
  echo "[mblow]   rc=$?"
done
echo "[mblow] DONE at $(date -u)"

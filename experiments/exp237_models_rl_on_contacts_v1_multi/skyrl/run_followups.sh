#!/usr/bin/env bash
# Everything the main pipeline could not finish, in descending value per GPU-hour.
#
# Runs ON the node, after `run_pipeline.sh` writes PIPELINE COMPLETE.
#
# 1. **M-B, re-run.** Its first attempt died 80 s after the previous stage's
#    eight vLLM engines exited, with "Engine core initialization failed" and no
#    mention of why. `run_arm.sh` now waits out the teardown; this is the retry.
# 2. **eval M-C step-18** -- the eval it never got, because its first attempt
#    died on the ninja PATH bug.
# 3. **eval M-F step-18** -- a second point on the trade-off curve. Step-36
#    bought +0.051 last-section R-precision for -0.014 consensus; step-18 is the
#    same reward at half the distance, and is the cheapest way to ask whether the
#    two always move together or whether some checkpoint gets one without paying
#    the other.
# 4. **M-C at lr 3e-6.** #208 fitted, and then refuted, a model in which
#    diversity loss depends only on how far the policy moves -- its arm C v3
#    moved furthest and lost the LEAST coverage. Same reward, a third of the step
#    size. If coverage still collapses at a third the distance, the collapse
#    belongs to the reward rather than to the distance.
set -u

HERE=$HOME/exp237/skyrl
LOGS=$HOME/exp237_logs
WAIT_FOR_PIPELINE=${WAIT_FOR_PIPELINE:-1}

if [ "$WAIT_FOR_PIPELINE" = "1" ]; then
  until grep -q "PIPELINE COMPLETE" "$LOGS/pipeline.status" 2>/dev/null; do sleep 60; done
fi
echo "[followup] starting at $(date -u)"

# Wait for the cards AND for vLLM's engine cores, which outlive the memory being
# freed. See run_arm.sh for the failure this exists to prevent.
drain() {
  for i in $(seq 1 240); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
    [ "$busy" -eq 0 ] && break
    sleep 10
  done
  for i in $(seq 1 60); do
    pgrep -f "VLLM::EngineCore" >/dev/null || return 0
    sleep 5
  done
  return 1
}

step() {
  local name=$1; shift
  echo "[followup] START $name at $(date -u)"
  local t0=$SECONDS
  drain || { echo "[followup] $name SKIPPED: cards never drained"; return 1; }
  ( "$@" ) >> "$LOGS/followups.log" 2>&1
  echo "[followup] END   $name rc=$? after $(( (SECONDS - t0) / 60 ))m"
}

cd "$HERE" || exit 1
step train-M-B   env ARM=M-B LR=1e-5 STEPS=72 CKPT_EVERY=18 bash run_arm.sh
step eval-M-B    env ARM=M-B bash run_eval.sh
step eval-M-C-18 env ARM=M-C STEP=18 bash run_eval.sh
step eval-M-F-18 env ARM=M-F STEP=18 bash run_eval.sh
step train-M-C-lowlr env ARM=M-C LR=3e-6 STEPS=48 CKPT_EVERY=16 ROOT=$HOME/exp237_data_lowlr bash run_arm.sh
step eval-M-C-lowlr  env ARM=M-C ROOT=$HOME/exp237_data_lowlr bash run_eval.sh
echo "[followup] ALL DONE at $(date -u)"

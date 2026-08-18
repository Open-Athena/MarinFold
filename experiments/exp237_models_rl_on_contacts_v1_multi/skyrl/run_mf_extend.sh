#!/usr/bin/env bash
# Continue arm M-F — how far does the final section actually go? — issue #237.
#
# **Why.** M-F is the only arm whose own reward was still climbing when it
# stopped: last-section F1 ran 0.33 -> 0.48 over 42 batches with no sign of a
# plateau, and it was stopped by the coverage criterion that was later shown to
# be in the wrong units (union/R was 2.80, nowhere near the 1.25 floor where
# #208's mechanism actually binds). So the arm was never trained to exhaustion.
#
# **Why it matters, and it is not the consensus number.** A single rollout's
# final section is a *deployable* prediction: one generation, one self-consistent
# contact set, no vote and no cutoff. Consensus is a ranking over pairs that must
# be cut at R to become a set -- and R comes from ground truth. Measured cost per
# protein: plain 22-rollout consensus 11,005 generated tokens for 0.5896; M-F's
# final section 3,697 tokens for 0.5075. This run asks how far that 0.5075 goes.
#
# Resumes from ckpts_m_f/global_step_36 under the CORRECTED gates, so the run
# stops on a real collapse rather than on the retired criterion.
set -u

STEPS=${STEPS:-120}
CKPT_EVERY=${CKPT_EVERY:-12}
EVAL_STEPS=${EVAL_STEPS:-"120 84 60"}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl

# Watch EVERY driver log, not one named file. A chained stage that hands off with
# `exec` keeps the *caller's* stdout, so M-BC's output landed in
# insert75_driver.log rather than mbc_driver.log -- and a waiter watching the
# named file would have blocked forever on a log nothing will write again.
# Grepping the whole directory cannot be wrong-footed by where a stage was
# launched from.
if [ "${WAIT_FOR:-1}" = "1" ]; then
  echo "[mfx] waiting for '[mbc] DONE' in any driver log"
  until grep -qs "^\[mbc\] DONE" "$LOGS"/*_driver.log; do sleep 60; done
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

cd "$HERE" || exit 1
echo "[mfx] resuming M-F from step 36 to $STEPS at $(date -u)"
drain || { echo "[mfx] cards never drained"; exit 1; }
ARM=M-F LR=1e-5 STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY \
  bash run_arm.sh >> "$LOGS/mf_extend.log" 2>&1
echo "[mfx] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  [ -d "$HOME/exp237_data/ckpts_m_f/global_step_$st" ] || { echo "[mfx] no ckpt $st"; continue; }
  echo "[mfx] eval step $st at $(date -u)"
  drain && ARM=M-F STEP=$st bash run_eval.sh >> "$LOGS/mf_extend.log" 2>&1
  echo "[mfx]   rc=$?"
done
echo "[mfx] DONE at $(date -u)"

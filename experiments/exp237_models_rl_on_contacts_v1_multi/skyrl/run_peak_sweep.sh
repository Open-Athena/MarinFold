#!/usr/bin/env bash
# Where is the peak? — issue #237 follow-up.
#
# The dose-response has exactly ONE arm-M-C point: step-18 at KL 0.0072, which
# reads consensus 0.5750 against the warm start's 0.5673 and the zero-LR
# control's 0.5678. The run died at step 26 on a coverage criterion that was
# later shown to be in the wrong units, and it checkpointed every 18 steps, so
# nothing else survives. That single point cannot distinguish
#
#   (a) 0.0072 is the peak and the curve turns over after it, from
#   (b) the curve is still climbing and 0.5750 understates what this reward does.
#
# Both are consistent with everything measured so far, and they imply opposite
# next experiments. So: **resume M-C from its own step-18 checkpoint** -- same
# reward, same learning rate, same data order, so this extends the existing curve
# rather than starting a new one -- with the corrected gate and a checkpoint
# every 4 steps instead of every 18.
#
# Resuming rather than retraining is deliberate. The interval that matters is
# ABOVE 0.0072: the anchor below it is already measured twice (M-0 at KL 0 reads
# +0.0006 with a CI including zero, and the warm start is the same number), so
# the curve from 0 to 0.0072 is pinned at both ends and only the far side is
# unknown. Retraining steps 1-18 would spend an hour reproducing a checkpoint
# that already exists.
#
#   STEPS=48 EVAL_STEPS="24 32 40 48" ./run_peak_sweep.sh
set -u

STEPS=${STEPS:-48}
CKPT_EVERY=${CKPT_EVERY:-4}
EVAL_STEPS=${EVAL_STEPS:-"24 32 40 48"}
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

cd "$HERE" || exit 1
echo "[peak] resuming M-C from step 18 to $STEPS, ckpt every $CKPT_EVERY, at $(date -u)"
drain || { echo "[peak] cards never drained"; exit 1; }
# lr UNCHANGED at 1e-5. A different rate would sample a different curve; the
# whole point is to add points to the one that produced 0.5750.
ARM=M-C LR=1e-5 STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY \
  bash run_arm.sh >> "$LOGS/peak_sweep.log" 2>&1
echo "[peak] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  if [ ! -d "$HOME/exp237_data/ckpts_m_c/global_step_$st" ]; then
    echo "[peak] no checkpoint at step $st, skipping"; continue
  fi
  echo "[peak] eval step $st at $(date -u)"
  drain && ARM=M-C STEP=$st bash run_eval.sh >> "$LOGS/peak_sweep.log" 2>&1
  echo "[peak]   rc=$?"
done
echo "[peak] DONE at $(date -u)"

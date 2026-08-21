#!/usr/bin/env bash
# Arm M-KB — arm M-K's reward on a 4x larger batch. Issue #237.
#
# **The observation this is built on.** Every arm in this experiment peaks after
# seeing a startlingly small amount of data:
#
#     M-B lr1e-5  peak step 18 ->  144 proteins       of a 10,000-prompt pool
#     M-C         peak step 18 ->  144 proteins
#     M-K         peak step 36 ->  288 proteins
#     M-B lr3e-6  peak step 90 ->  720 proteins
#
# and the gradient at each step is an average over **8 prompt groups**. The lr-0
# control (arm M-0) measured the per-batch diagnostics swinging 3.6x under a
# policy that was not changing at all, so the per-step signal is dominated by
# which eight proteins were drawn.
#
# **Why this is a different axis from everything already tried.** Three
# interventions have been run against the peak -- a 3x smaller learning rate, a
# 50x KL penalty, and a candidate-count floor -- and all three changed the SIZE
# or DIRECTION of the step while leaving the gradient's *quality* alone. None
# beat the peak they started from. If the peak is set by gradient noise rather
# than by distance, then averaging over 4x the prompts is the one change that
# moves it, and none of the previous three could have.
#
# **The change is exactly one variable.** `train_batch_size` 8 -> 32 prompts
# (256 rollouts/step against 64), same reward, same lr 1e-5, same group size,
# same pool and order. `policy_mini_batch_size` tracks it, so there is still one
# inner epoch and one minibatch and the PPO clip stays inert -- the update is the
# same REINFORCE-with-a-group-baseline it has been all along, computed from four
# times as many prompts.
#
# **It should also cost less than 4x.** The compute analysis measured generation
# as UNDER-BATCHED at ~11 concurrent sequences per engine, far below what
# saturates an A100's decode path, so 256 rollouts over six engines (~43 each)
# uses capacity that was idle rather than adding proportional wall clock.
#
# **What each outcome means.** If the peak moves later AND higher, gradient noise
# was the binding constraint and the whole experiment has been under-batched. If
# it moves later at the same height, the batch buys stability but not accuracy.
# If nothing changes, the peak is a property of the reward or of the model and
# the next arm should attack the reward.
set -u

GROUP=${GROUP:-8}
PROMPTS=${PROMPTS:-32}
LR=${LR:-1e-5}
STEPS=${STEPS:-48}
CKPT_EVERY=${CKPT_EVERY:-12}
ROOT=${ROOT:-$HOME/exp237_data_mk_bb}
EVAL_STEPS=${EVAL_STEPS:-"12 24 36 48"}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl

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
echo "[mkb] train M-K batch=$PROMPTS prompts x $GROUP = $((PROMPTS*GROUP)) rollouts/step, $STEPS steps at $(date -u)"
drain || { echo "[mkb] cards never drained"; exit 1; }
ARM=M-K LR=$LR STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY ROOT=$ROOT \
  GROUP=$GROUP PROMPTS=$PROMPTS RUN_SUFFIX=_bb \
  bash run_arm.sh >> "$LOGS/mkb.log" 2>&1
echo "[mkb] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  if [ ! -d "$ROOT/ckpts_m_k/global_step_$st" ]; then
    echo "[mkb] no checkpoint at step $st, skipping"; continue
  fi
  echo "[mkb] eval step $st at $(date -u)"
  drain && ARM=M-K STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/mkb.log" 2>&1
  echo "[mkb]   rc=$? step=$st done at $(date -u)"
done
echo "[mkb] DONE at $(date -u)"

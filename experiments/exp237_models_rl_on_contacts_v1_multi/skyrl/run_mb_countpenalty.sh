#!/usr/bin/env bash
# Arm M-BP — M-B resumed from its best checkpoint with a candidate-count floor. #237.
#
# **The proposal.** Add `beta * min(0, K - floor)` to arm M-B's reward, where K is
# the rollout's candidate count. Arm M-B's reward is `max_k F1(section k)`, which
# does not depend on K at all, so the arm has no first-order opinion about how
# many candidates it emits -- and both M-B runs declined into a section collapse:
#
#   lr 3e-6, sections/rollout at eval:  20.2 (step 90) -> 19.3 (120) -> 15.9 (150)
#                                       -> killed at 180 with a training median of 11.0
#
# **Why the floor is 18 and not 10.** The gate that stops these runs fires at 11
# sections. A floor of 10 is BELOW that, so `min(0, K - 10)` is identically zero
# through the entire decline above and first becomes non-zero after the run has
# already been stopped -- it would be a no-op dressed as a fix. 18 sits just
# under the healthy band (19-26 on every good M-B checkpoint), so it is silent
# while the arm is healthy and engages as soon as the decline starts.
#
# **Why beta = 0.03.** The penalty is added to the RAW scalar, before GRPO. Group
# spread of `max_k F1` is ~0.1 within a prompt group, so beta 0.03 puts a rollout
# at K = 11 about two within-group standard deviations down, and at K = 1 it is
# -0.51 against a reward in [0, 1]. Bounded, unlike arm M-C's marginal, which hit
# +4.79 at one section and 366x its value at 22 -- no fixed weight could balance
# that, which is why it ran away rather than drifting.
#
# **This is a controlled A/B.** Same checkpoint, same learning rate, same prompt
# pool and same data order as MBLONG; one term added. Its steps 120/150/180 are
# directly comparable to MBLONG's 0.5739 / 0.5575 / killed.
#
# **Staged into a FRESH root on purpose.** `resume_mode=latest` reads
# `trainer.ckpt_path` and takes the newest global_step_N under it. The lr-3e-6
# root now holds global_step_150 -- a checkpoint 0.0200 past its own peak -- so
# pointing at that root would silently resume from the wrong policy.
set -u

SRC=${SRC:-$HOME/exp237_data_mb_lowlr/ckpts_m_b/global_step_90}
ROOT=${ROOT:-$HOME/exp237_data_mb_pen}
FROM=${FROM:-90}
STEPS=${STEPS:-210}
CKPT_EVERY=${CKPT_EVERY:-30}
BETA=${BETA:-0.03}
FLOOR=${FLOOR:-18}
EVAL_STEPS=${EVAL_STEPS:-"120 150 180 210"}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl

WAIT_PID=${WAIT_PID:-}
if [ -n "$WAIT_PID" ]; then
  echo "[mbp] waiting for pid $WAIT_PID at $(date -u)"
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

if [ ! -d "$ROOT/ckpts_m_b/global_step_$FROM" ]; then
  echo "[mbp] staging $SRC -> $ROOT/ckpts_m_b/global_step_$FROM"
  mkdir -p "$ROOT/ckpts_m_b"
  cp -r "$SRC" "$ROOT/ckpts_m_b/global_step_$FROM" || exit 1
  echo "$FROM" > "$ROOT/ckpts_m_b/latest_ckpt_global_step.txt"
fi
staged=$(ls "$ROOT/ckpts_m_b" | grep -c global_step)
[ "$staged" -eq 1 ] || { echo "[mbp] FATAL: $staged checkpoints staged, expected exactly 1"; exit 1; }

cd "$HERE" || exit 1
echo "[mbp] train M-B+penalty beta=$BETA floor=$FLOOR from step $FROM to $STEPS at $(date -u)"
drain || { echo "[mbp] cards never drained"; exit 1; }
ARM=M-B LR=3e-6 STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY ROOT=$ROOT \
  EXTRA_OVERRIDES="min_union_ratio=0.0 min_union_over_r=1.25 count_penalty_beta=$BETA count_penalty_floor=$FLOOR" \
  bash run_arm.sh >> "$LOGS/mbp.log" 2>&1
echo "[mbp] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  if [ ! -d "$ROOT/ckpts_m_b/global_step_$st" ]; then
    echo "[mbp] no checkpoint at step $st, skipping"; continue
  fi
  echo "[mbp] eval step $st at $(date -u)"
  drain && ARM=M-B STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/mbp.log" 2>&1
  echo "[mbp]   rc=$? step=$st done at $(date -u)"
done
echo "[mbp] DONE at $(date -u)"

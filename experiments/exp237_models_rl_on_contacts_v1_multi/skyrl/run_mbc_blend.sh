#!/usr/bin/env bash
# Arm M-BC: GRPO(max_k F1) + lam * GRPO(C_i(all)) — issue #237.
#
# Runs after the lr-3e-6 M-B run. Both terms are ROLLOUT-level scalars,
# standardised separately over the prompt group, so `lam` weights standardised
# quantities and neither term can be gamed by section count -- which is the
# whole reason this is the blend rather than M-B plus M-C's per-section
# marginal (+4.79 at one section, -0.22 at 22).
#
# lam = 1.0 is the natural first point: equal weight in units of within-group
# spread. Not swept here, deliberately -- one arm first, to see whether the two
# terms cohere at all before spending a night on lam. M-B alone (lam = 0) is
# already measured at oracle-best 0.5574 / consensus 0.5741, which is the
# comparison this run is against.
set -u

LR=${LR:-1e-5}
STEPS=${STEPS:-48}
CKPT_EVERY=${CKPT_EVERY:-12}
LAM=${LAM_CONSENSUS:-1.0}
EVAL_STEPS=${EVAL_STEPS:-"36 48 24 12"}
ROOT=${ROOT:-$HOME/exp237_data_mbc}
WAIT_LOG=${WAIT_LOG:-$HOME/exp237_logs/mb_lowlr_driver.log}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl

if [ "${WAIT_FOR:-1}" = "1" ]; then
  echo "[mbc] waiting for $WAIT_LOG to report DONE"
  until grep -q "^\[mblow\] DONE" "$WAIT_LOG" 2>/dev/null; do sleep 60; done
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
echo "[mbc] train M-BC lr=$LR lam=$LAM steps=$STEPS at $(date -u)"
drain || { echo "[mbc] cards never drained"; exit 1; }
ARM=M-BC LR=$LR STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY ROOT=$ROOT LAM_CONSENSUS=$LAM \
  bash run_arm.sh >> "$LOGS/mbc.log" 2>&1
echo "[mbc] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  [ -d "$ROOT/ckpts_m_bc/global_step_$st" ] || { echo "[mbc] no ckpt $st"; continue; }
  echo "[mbc] eval step $st at $(date -u)"
  drain && ARM=M-BC STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/mbc.log" 2>&1
  echo "[mbc]   rc=$?"
done
echo "[mbc] DONE at $(date -u)"

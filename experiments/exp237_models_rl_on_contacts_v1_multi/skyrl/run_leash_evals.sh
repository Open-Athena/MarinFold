#!/usr/bin/env bash
# Score MKLEASH in the order the question needs — issue #237.
#
# The stock order in run_long_trajectory.sh is newest-first (300, 240, 180 ...),
# which is exactly backwards here. The leash HELD to about step 130 (policy_kl
# 0.013) and then broke: 0.032 at 140, 0.19 at 155, 0.37 at 160, with sections
# per rollout running 25 -> 54 (arm M-F's runaway, which tripped the max_sections
# gate 2/3 twice without ever hitting 3 in a row).
#
# So the checkpoints that answer "does more optimisation at a FIXED distance
# help?" are the in-window ones -- 120, 90, 60, 30, all under KL 0.016 -- and
# those get scored first. Step 180 is scored last, to document the runaway with
# a number rather than a training curve. 240 and 300 are skipped: at KL 0.56
# they measure a destroyed policy, which arm M-B step-80 already did.
#
# Reference points, both at the same lr and reward:
#   unleashed M-K reached KL 0.032 by step  24 and peaked at step 36 (0.5806)
#   leashed   M-K reached KL 0.013 by step 120
# -- so if the leashed step-120 beats 0.5806 the answer is yes, and if it lands
# near the unleashed step-12/18 numbers (0.5739 / 0.5764) then distance is the
# ceiling and the extra 100 steps bought nothing.
#
# **Getting WAIT_PID right.** Capture the pid with `ps -eo pid,args | grep <script>
# | grep -v "grep\|bash -c"`, NOT with `pgrep -f`. `pgrep -f` matches its own
# `bash -c ...` wrapper whenever the wrapper's command line contains the pattern,
# so it returns the wrapper's pid -- which exits immediately, leaving this script
# to fall straight through the wait into `drain()` and time out an hour later
# against cards that were never going to free. That happened once here.
set -u

ROOT=${ROOT:-$HOME/exp237_data_mk_leash}
STEPS=${STEPS:-"120 90 60 30 180"}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl

# Wait out the training process by PID -- `pgrep -f` matches its own bash -c
# wrapper here and would return immediately.
TRAIN_PID=${TRAIN_PID:-}
if [ -n "$TRAIN_PID" ]; then
  echo "[leash] waiting for training pid $TRAIN_PID at $(date -u)"
  while ps -p "$TRAIN_PID" >/dev/null 2>&1; do sleep 60; done
  echo "[leash] training exited at $(date -u)"
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

cd "$HERE" || exit 1
for st in $STEPS; do
  if [ ! -d "$ROOT/ckpts_m_k/global_step_$st" ]; then
    echo "[leash] no checkpoint at step $st, skipping"; continue
  fi
  echo "[leash] eval step $st at $(date -u)"
  drain && ARM=M-K STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/leash_eval.log" 2>&1
  echo "[leash]   rc=$? step=$st done at $(date -u)"
done
echo "[leash] DONE at $(date -u)"

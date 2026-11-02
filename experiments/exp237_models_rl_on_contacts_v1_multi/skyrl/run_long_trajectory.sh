#!/usr/bin/env bash
# Do LONG RL trajectories beat the very short ones? — issue #237.
#
# Every result in this experiment so far comes from a run of 26-120 steps, and
# five of the six arms peaked by step ~20. The two facts that make "just train
# longer" a real question rather than a hopeful one:
#
#   * arm M-K ran its full 48-step budget with **no gate strike and no
#     divergence** -- it stopped for the schedule, not for a reason;
#   * arm M-B at lr 3e-6 held 0.5754 / 0.5760 / 0.5775 / 0.5739 across steps
#     60 / 75 / 90 / 120 -- a plateau, not a decay, with a healthy reward curve
#     the whole way.
#
# But the experiment ALSO measured that outcome tracks **distance moved (KL)**
# and not schedule: two learning rates 3.3x apart agree to 0.002 at matched KL.
# That makes "more steps" and "more distance" the same thing at a fixed lr, and
# the dose-response says more distance is bad past KL ~0.02. So the two runs
# below ask different questions on purpose.
#
# ---------------------------------------------------------------------------
# RUN 1 -- MBLONG: does M-B's lr-3e-6 plateau go anywhere if you keep walking?
#
# The direct version of the question. Resume the lr 3e-6 run from its own step
# 120 (full FSDP state, optimiser included) and take it to 360, checkpointing
# every 30. Prediction from the dose-response: KL grows past 0.02 and the score
# decays like every other arm. If instead the plateau HOLDS out to KL 0.05+,
# "distance decides" is wrong in an important way and the small-lr regime is
# qualitatively different.
#
# Gates stay ARMED and fatal. At lr 1e-5 arm M-B's Jaccard rose 1.48x before it
# was killed; if the slow run does the same thing over 240 steps, tripping is
# the answer, not a failure.
#
# ---------------------------------------------------------------------------
# RUN 2 -- MKLEASH: many steps at a FIXED distance, on the best arm.
#
# The version of the question that "distance decides" cannot pre-answer. If the
# outcome depends only on KL, then the way to profit from a long run is to keep
# optimising while NOT travelling -- which needs the KL penalty to actually
# bind. It currently does not: `kl_loss_coef=0.001` is inert (terminal KLs of
# 0.09, 0.49 and 3.26 were reached with the penalty in place), and with
# `update_epochs_per_batch=1` the PPO clip never fires either, so nothing in
# the optimiser limits the step.
#
# So: arm M-K, lr 1e-5, 300 steps, `kl_loss_coef=0.05` -- 50x, chosen to make
# the penalty comparable to a unit-spread advantage rather than 0.1 % of it.
# Read the run by its `policy_kl` column first: if KL settles near M-K's own
# optimum (~0.016-0.03) and the score keeps climbing, long trajectories win and
# the leash is how you buy them. If KL settles and the score does NOT move, the
# ceiling is the distance and no amount of optimisation at that distance helps
# -- which is a clean, publishable negative.
#
# The coefficient is a first guess and is stated as one. If KL runs away
# anyway, that is a measurement about the k3 penalty's strength, not a wasted
# run, and it costs one eval to find out (the step-30 checkpoint).
set -u

LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl
mkdir -p "$LOGS"

drain() {
  for i in $(seq 1 240); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
    [ "$busy" -eq 0 ] && break
    sleep 15
  done
  # vLLM's engine teardown outlives the memory it frees: six new engines race
  # the old IPC sockets and lose with "Engine core initialization failed".
  for i in $(seq 1 60); do pgrep -f "VLLM::EngineCore" >/dev/null || { sleep 30; return 0; }; sleep 5; done
  return 1
}

evaluate() {  # arm-tag root step
  local tag=$1 root=$2 st=$3
  if [ ! -d "$root/ckpts_$tag/global_step_$st" ]; then
    echo "[long] no checkpoint at $tag step $st, skipping"; return 0
  fi
  echo "[long] eval $tag step $st at $(date -u)"
  drain && ARM=$4 STEP=$st ROOT=$root bash run_eval.sh >> "$LOGS/long.log" 2>&1
  echo "[long]   rc=$?"
}

cd "$HERE" || exit 1

# ---------------- RUN 1: M-B lr 3e-6, 120 -> 360 ----------------
if [ "${DO_MBLONG:-1}" = "1" ]; then
  ROOT=$HOME/exp237_data_mb_lowlr
  echo "[long] MBLONG: resuming M-B lr3e-6 from step 120 to ${MBLONG_STEPS:-360} at $(date -u)"
  drain || { echo "[long] cards never drained"; exit 1; }
  ARM=M-B LR=3e-6 STEPS=${MBLONG_STEPS:-360} CKPT_EVERY=30 ROOT=$ROOT \
    EXTRA_OVERRIDES="min_union_ratio=0.0 min_union_over_r=1.25" \
    bash run_arm.sh >> "$LOGS/long.log" 2>&1
  echo "[long] MBLONG train rc=$? at $(date -u)"
  for st in 360 300 240 180 150; do evaluate m_b "$ROOT" "$st" M-B; done
  echo "[long] MBLONG DONE at $(date -u)"
fi

# ---------------- RUN 2: M-K on a KL leash, 300 steps ----------------
if [ "${DO_MKLEASH:-1}" = "1" ]; then
  ROOT=$HOME/exp237_data_mk_leash
  mkdir -p "$ROOT"
  echo "[long] MKLEASH: M-K lr1e-5 kl_loss_coef=${LEASH:-0.05} for ${MKLEASH_STEPS:-300} steps at $(date -u)"
  drain || { echo "[long] cards never drained"; exit 1; }
  ARM=M-K LR=1e-5 STEPS=${MKLEASH_STEPS:-300} CKPT_EVERY=30 ROOT=$ROOT \
    EXTRA_OVERRIDES="trainer.algorithm.kl_loss_coef=${LEASH:-0.05}" \
    bash run_arm.sh >> "$LOGS/long.log" 2>&1
  echo "[long] MKLEASH train rc=$? at $(date -u)"
  for st in 300 240 180 120 60 30; do evaluate m_k "$ROOT" "$st" M-K; done
  echo "[long] MKLEASH DONE at $(date -u)"
fi

echo "[long] ALL DONE at $(date -u)"

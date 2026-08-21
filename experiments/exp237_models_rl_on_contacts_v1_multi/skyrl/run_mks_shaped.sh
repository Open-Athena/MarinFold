#!/usr/bin/env bash
# Arm M-KS — arm M-K's base plus zero-sum within-rollout shaping. Issue #237.
#
#     A_i,k  =  GRPO_group( C_i(all) )_i  +  beta * ( m_k - mean_k m )
#
# where m_k = C(s_1..s_k) - C(s_1..s_{k-1}) is the CAUSAL prefix marginal: what
# section k added, given exactly what was in context when it was written.
#
# **This is the last untested piece of the arm derived in RESULTS.md.** That
# design had three terms; arm M-K is the `beta = lam = 0` corner of it and
# produced the best consensus in the experiment (0.5806). `beta` has never been
# run.
#
# **Why it is the right next thing.** Arm M-K reinforces every one of a good
# rollout's ~22 sections identically -- including the ones that merely repeat
# their siblings. Its rolling Jaccard climbs 0.23 -> 0.39 exactly as its score
# turns over, and *every* arm in this experiment that improved anything did so
# while making its candidates more alike. But the metric feeds on the opposite:
# 22 sections of one rollout cover 658 distinct pairs against 1,065 for 22
# INDEPENDENT rollouts, and that 62 % coverage gap is the whole distance from
# 0.5673 to the 0.5896 bar. This term is the one that pays a section for
# covering something its predecessors missed and pays it nothing for repeating
# them.
#
# **Why the prefix form is safe here and was refuted as a standalone reward.**
# Measured on its own it reads +2.03 at one section against -0.22 at 22 -- the
# same count-adverse pathology that destroyed arm M-C, milder. It telescopes,
# which bounds the SUM, but `loss_reduction=token_mean` reads the MEAN. Here it
# is centred within the rollout, so `sum_k (m_k - mean_k m) = 0` identically: it
# cannot move the rollout's total advantage at any section count, only
# redistribute the base among the sections that earned it. Every count pathology
# in #237 was a statement about a reward's LEVEL as a function of K; a zero-sum
# term has no level to move. Pinned by test.
#
# **beta = 3.0, and it was measured rather than guessed.** The base is
# GRPO-standardised (unit spread by construction). On 400 real rollouts from
# M-K's own step-36 checkpoint the centred prefix marginal has sd **0.078**
# (calib_beta.py), so beta 3.2 puts the shaping at a quarter of the base's
# spread and beta 12.8 would match it. 3.0 is deliberately at the low end: the
# base is the best arm here and this should modulate it, not replace it.
# #208 set the analogous weight by intuition twice and was wrong in both
# directions.
set -u

LR=${LR:-1e-5}
STEPS=${STEPS:-48}
CKPT_EVERY=${CKPT_EVERY:-12}
BETA=${BETA:-3.0}
ROOT=${ROOT:-$HOME/exp237_data_mks}
EVAL_STEPS=${EVAL_STEPS:-"36 24 48 12"}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl
#
# **Getting WAIT_PID right.** Capture it with `ps -eo pid,args | grep <script>
# | grep -v "grep\|bash -c"`, NOT with `pgrep -f` -- pgrep matches its own
# `bash -c ...` wrapper and returns a pid that exits immediately.
WAIT_PID=${WAIT_PID:-}
if [ -n "$WAIT_PID" ]; then
  echo "[mks] waiting for pid $WAIT_PID at $(date -u)"
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

# **Swap in the new runner only after the previous run has released it.** Bash
# reads a script incrementally, by byte offset, so overwriting `run_arm.sh` while
# an earlier arm's `bash run_arm.sh` sits blocked on its python call makes that
# process resume at the wrong offset in a different file. Arm M-KS needs a
# `run_arm.sh` that knows the M-KS case and passes `beta_shape`, so the new copy
# is staged as `.next` at deploy time and moved into place HERE — after WAIT_PID,
# by which point the previous runner has exited.
if [ -f "$HERE/run_arm.sh.next" ]; then
  echo "[mks] installing the staged run_arm.sh"
  mv "$HERE/run_arm.sh.next" "$HERE/run_arm.sh" || exit 1
fi
grep -q "M-KS)" "$HERE/run_arm.sh" || {
  echo "[mks] FATAL: run_arm.sh does not know arm M-KS; the staged copy never landed"; exit 1; }

mkdir -p "$ROOT"
cd "$HERE" || exit 1
echo "[mks] train M-KS beta_shape=$BETA lr=$LR steps=$STEPS at $(date -u)"
drain || { echo "[mks] cards never drained"; exit 1; }
ARM=M-KS LR=$LR STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY ROOT=$ROOT BETA_SHAPE=$BETA \
  bash run_arm.sh >> "$LOGS/mks.log" 2>&1
echo "[mks] train rc=$? at $(date -u)"

# Scored best-first: M-K peaked at 36 and every other arm by ~20, so if the GPUs
# are needed back the informative checkpoints are already done.
for st in $EVAL_STEPS; do
  if [ ! -d "$ROOT/ckpts_m_ks/global_step_$st" ]; then
    echo "[mks] no checkpoint at step $st, skipping"; continue
  fi
  echo "[mks] eval step $st at $(date -u)"
  drain && ARM=M-KS STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/mks.log" 2>&1
  echo "[mks]   rc=$? step=$st done at $(date -u)"
done
echo "[mks] DONE at $(date -u)"

#!/usr/bin/env bash
# Arm M-KP — per-PAIR shaping, dropping the section as the credit unit. Issue #237.
#
#   A_token = GRPO_group(C_i(all))_i + beta * v_token
#
# where v_token scores each emitted `<contact> <pI> <pJ>` triple on its own three
# tokens:  +1/R for a first-time TRUE pair, -lam_false/R for a first-time FALSE
# one, and exactly 0 for a repeat -- true or false -- anywhere later in the
# rollout.
#
# **Why the section had to go.** Both section-based shaping arms failed the same
# way: the value of a section depended on how the predictions were SLICED, not
# on what they contained.
#
#   M-KS   value fell with the section INDEX (the prefix marginal decays by
#          construction) -> "stop early" -> collapse to 10.7 sections by step 21
#   M-KS3  value rose as sections got SMALLER (a smaller section adds less junk,
#          and token_mean reads the mean) -> "fragment" -> runaway to 102
#          sections by step 15, consensus 0.4627
#
# M-KS2 survives between them, and it survives by accident: the consensus
# marginal it shapes on is a weak (r = +0.194) and partition-insensitive proxy
# for novelty. Sharpening the proxy destroyed the property that made it safe.
# Here the partition never enters the arithmetic -- cutting a section in two
# leaves every pair's value and every pair's tokens exactly as they were.
#
# **Structural tokens carry no shaping, and that is load-bearing.**
# `<begin_statements>`, `<end>` and everything outside a triple stay at exactly
# 0, and the zero-sum centring runs over the triple tokens ONLY. Centring over
# all tokens would hand every structural token `-mean`, and since most emitted
# pairs are false the mean is negative -- so each `<begin_statements>` would be
# paid for existing. That is M-KS3's runaway, reachable without emitting a
# single new pair. Under this construction the decision to open a section
# receives no shaping signal in either direction. Pinned by test.
#
# **On #237's per-contact exclusion.** The issue rules out per-contact-ONLY
# rewards, because #208 established they are sharpening operators. This is not
# that, in two respects, and the distinction is the experiment: it is *shaping*
# on top of arm M-K's scale-correct rollout-level base rather than the whole
# objective, and **a repeated pair scores exactly zero whether it is true or
# false**. A policy that sharpens by repeating its confident set earns nothing
# after the first section; the only way to score is to add correct content that
# is not already there. If #208's result transfers anyway, that is worth knowing.
#
# **beta = 80, measured (calib4.py).** Values are +/-1/R with R ~ 165, so the
# per-token spread after centring is sd 0.00306 against a GRPO base normalised
# to unit spread. 81.8 puts the shaping at a quarter of the base -- the same
# quarter-weight convention used for M-KS2 (beta 15) and M-KS3 (beta 7). The
# large number is the units, not an aggressive setting.
set -u

LR=${LR:-1e-5}
STEPS=${STEPS:-48}
CKPT_EVERY=${CKPT_EVERY:-12}
BETA=${BETA:-80.0}
ROOT=${ROOT:-$HOME/exp237_data_mkp}
EVAL_STEPS=${EVAL_STEPS:-"36 24 12 48"}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl
# Capture WAIT_PID with `ps -eo pid,args | grep <script> | grep -v "grep\|bash -c"`,
# never `pgrep -f` -- it matches its own wrapper and returns a pid that exits at once.
WAIT_PID=${WAIT_PID:-}
if [ -n "$WAIT_PID" ]; then
  echo "[mkp] waiting for pid $WAIT_PID at $(date -u)"
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

mkdir -p "$ROOT"
cd "$HERE" || exit 1
echo "[mkp] train M-KP beta_shape=$BETA positional=true lr=$LR steps=$STEPS at $(date -u)"
drain || { echo "[mkp] cards never drained"; exit 1; }
ARM=M-KS LR=$LR STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY ROOT=$ROOT \
  BETA_SHAPE=$BETA POSITIONAL_SHAPE=true SHAPE_SIGNAL=pair RUN_SUFFIX=_pair \
  bash run_arm.sh >> "$LOGS/mkp.log" 2>&1
echo "[mkp] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  if [ ! -d "$ROOT/ckpts_m_ks/global_step_$st" ]; then
    echo "[mkp] no checkpoint at step $st, skipping"; continue
  fi
  echo "[mkp] eval step $st at $(date -u)"
  drain && ARM=M-KS STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/mkp.log" 2>&1
  echo "[mkp]   rc=$? step=$st done at $(date -u)"
done
echo "[mkp] DONE at $(date -u)"

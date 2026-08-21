#!/usr/bin/env bash
# Arm M-FC — teach the final section to SYNTHESISE the drafts above it. #237.
#
# The selector framing is wrong, and the measurement says so. On M-B's own
# generations, one rollout's readouts are:
#
#     last section, deployed today          0.5082
#     ORACLE best single section            0.5646
#     consensus of sections 1..K-1          0.5750   <- the real target
#     consensus of ALL sections             0.5775
#
# A perfect SELECTOR of one draft lands 0.010 BELOW simply voting the drafts. So
# the job of the final section is not to pick the best of what precedes it, it is
# to aggregate it -- which needs no ground truth and no extra sampling, because
# every draft is already in context when the final section is written. Headroom
# over today's behaviour: +0.067.
#
#     A_i = GRPO( F1(last section) ) + lam * GRPO( C_i(all) )
#
# The consensus term does two jobs. It keeps the drafts worth aggregating -- a
# synthesis is only as good as what it reads -- and it is the restoring force
# arm M-F lacked: C(all) collapses under a section-count runaway (0.33 at M-F's
# worst against ~0.50 healthy), so the direction M-F ran is now penalised.
#
# Unlike M-BC, the two terms are complementary rather than competing: one shapes
# the drafts, the other shapes the synthesis. M-BC failed because max_k F1 and
# C(all) both want to own the same sections.
#
# Dense checkpoints: every arm here peaked in KL 0.005-0.02, which at lr 1e-5 is
# about steps 10-20, so 6 is the spacing that actually samples it.
set -u

LR=${LR:-1e-5}
STEPS=${STEPS:-48}
CKPT_EVERY=${CKPT_EVERY:-6}
LAM=${LAM_CONSENSUS:-1.0}
EVAL_STEPS=${EVAL_STEPS:-"12 18 24 36"}
ROOT=${ROOT:-$HOME/exp237_data_mfc}
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

mkdir -p "$ROOT"
cd "$HERE" || exit 1
echo "[mfc] train M-FC lr=$LR lam=$LAM steps=$STEPS ckpt=$CKPT_EVERY at $(date -u)"
drain || { echo "[mfc] cards never drained"; exit 1; }
ARM=M-FC LR=$LR STEPS=$STEPS CKPT_EVERY=$CKPT_EVERY ROOT=$ROOT LAM_CONSENSUS=$LAM \
  bash run_arm.sh >> "$LOGS/mfc.log" 2>&1
echo "[mfc] train rc=$? at $(date -u)"

for st in $EVAL_STEPS; do
  [ -d "$ROOT/ckpts_m_fc/global_step_$st" ] || { echo "[mfc] no ckpt $st"; continue; }
  echo "[mfc] eval step $st at $(date -u)"
  drain && ARM=M-FC STEP=$st ROOT=$ROOT bash run_eval.sh >> "$LOGS/mfc.log" 2>&1
  echo "[mfc]   rc=$?"
done
echo "[mfc] DONE at $(date -u)"

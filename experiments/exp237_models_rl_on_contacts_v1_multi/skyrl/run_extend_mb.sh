#!/usr/bin/env bash
# Continue arm M-B past the gate that stopped it — issue #237.
#
# **Why.** M-B was stopped at step 36 by #237's preregistered coverage criterion
# (union pairs per rollout below 80 % of the run's own warmup). It was then
# scored, and it had improved *every* aggregation mode on *every* cut:
# consensus +0.0068, oracle-best +0.0232, last +0.0341, second-last +0.0652 on
# the legacy 554, all with paired bootstrap CIs excluding zero.
#
# The gate was measuring the wrong quantity. #208's coverage mechanism is that
# R-precision cuts a ranking at R = |gt|, so zero-vote pairs start padding the
# top-R **only once the union falls below R**. Measured at eval, union/R was
# 3.98 for the warm start and never left 2.8-4.0 for any arm here. The coverage
# that was lost was headroom nothing was using.
#
# So: same reward, resumed from its own step-36 checkpoint (SkyRL's
# `resume_mode=latest` reads `trainer.ckpt_path`), with the relative gate off and
# the mechanism's own gate -- union/R >= 1.25 -- in its place.
set -u

STEPS=${STEPS:-80}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl

for i in $(seq 1 240); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
  [ "$busy" -eq 0 ] && break
  sleep 15
done
for i in $(seq 1 60); do
  pgrep -f "VLLM::EngineCore" >/dev/null || break
  sleep 5
done

cd "$HERE" || exit 1
echo "[extend] resuming M-B to step $STEPS at $(date -u)"
ARM=M-B LR=1e-5 STEPS=$STEPS CKPT_EVERY=18 \
  EXTRA_OVERRIDES="min_union_ratio=0.0 min_union_over_r=1.25" \
  bash run_arm.sh >> "$LOGS/extend_mb.log" 2>&1
echo "[extend] train rc=$? at $(date -u)"
ARM=M-B bash run_eval.sh >> "$LOGS/extend_mb.log" 2>&1
echo "[extend] eval rc=$? at $(date -u)"
echo "[extend] DONE at $(date -u)"

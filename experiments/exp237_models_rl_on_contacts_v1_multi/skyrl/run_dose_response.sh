#!/usr/bin/env bash
# The remaining evaluation, in the order the results made valuable — issue #237.
#
# What the first four evaluations showed, ordered by how far the policy moved:
#
#   checkpoint       KL      consensus   best*    last
#   #230 warm start  0        0.5673    0.5342   0.4566
#   M-0 (lr 0)       0        0.5678    0.5364   0.4594
#   M-C step-18     ~0.007    0.5750    0.5578   0.5267
#   M-B step-36     ~0.016    0.5741    0.5574   0.4908
#   M-F step-36     ~0.031    0.5529    0.5189   0.5075
#
# Consensus rises at small KL and falls at large -- a dose-response, not a
# verdict, and #208's negative result is its far end. M-F step-18 is the missing
# point: the arm that moved furthest, read at the distance the others were read
# at. Without it the pattern could still be "M-F's reward is the bad one" rather
# than "distance is".
set -u
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl

until grep -q "^\[extend\] DONE" "$LOGS/extend_mb_driver.log" 2>/dev/null; do sleep 60; done
for i in $(seq 1 240); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
  [ "$busy" -eq 0 ] && break
  sleep 15
done
for i in $(seq 1 60); do pgrep -f "VLLM::EngineCore" >/dev/null || break; sleep 5; done

cd "$HERE" || exit 1
echo "[dose] eval M-F step-18 at $(date -u)"
ARM=M-F STEP=18 bash run_eval.sh >> "$LOGS/dose.log" 2>&1
echo "[dose] rc=$? DONE at $(date -u)"

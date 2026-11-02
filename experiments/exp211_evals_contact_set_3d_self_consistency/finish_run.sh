#!/usr/bin/env bash
# Chain the remaining issue #211 pipeline once rollout generation finishes.
# Waits for run_rollouts_local.py's DONE line, then: verify -> score -> analyze
# -> plot -> summary.pdf. One log, one completion.
set -uo pipefail
cd "$(dirname "$0")"

echo "[finish] waiting for rollout generation..."
until grep -qE "^\[local\] DONE|^Traceback" _scratch/rollouts.log 2>/dev/null; do sleep 60; done
if ! grep -q "^\[local\] DONE" _scratch/rollouts.log; then
  echo "[finish] ABORT: generation did not finish cleanly"; exit 1
fi
grep "^\[local\] DONE" _scratch/rollouts.log
echo "[finish] $(ls _scratch/rollouts/contacts | wc -l) protein parquets"

echo "[finish] === step 1c: reproduce exp82's published R-precision ==="
uv run python verify_against_exp82.py --rollouts _scratch/rollouts \
  --out data/verify_exp82.csv 2>&1 | tail -12 || exit 1

echo "[finish] === step 2-3: score all seven arms ==="
# 30 replicates x 3 rollout-derived arms + 4 reference arms = 94 sets/protein.
# max-pairs caps rows x O(L^2) pairs so long proteins chunk instead of OOMing.
uv run python score_arms.py --rollouts _scratch/rollouts --gt-dir _scratch/gt \
  --bounds data/bounds.json --out data/arm_scores.csv \
  --n-replicates 30 --n-restarts 4 --iters 3000 --max-pairs 15000000 2>&1 \
  | grep -vE "^\[score\] [0-9]+/" | tail -20
[ -s data/arm_scores.csv ] || { echo "[finish] ABORT: no arm scores"; exit 1; }

echo "[finish] === step 4: analysis ==="
uv run python analyze.py --scores data/arm_scores.csv 2>&1 | tail -60 || exit 1

echo "[finish] === step 4: figures ==="
uv run python plot_results.py --scores data/arm_scores.csv 2>&1 | tail -3 || exit 1

echo "[finish] === summary.pdf ==="
uv run python build_summary.py 2>&1 | tail -4 || exit 1

echo "[finish] PIPELINE_COMPLETE"

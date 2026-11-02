#!/bin/bash
# exp230 -- everything after Gate A, in order, unattended.
#
# Each stage waits for the GPUs to drain before starting, so this can be launched
# while Gate A is still running and will simply queue behind it.
set -u
REPO=$HOME/MarinFold/experiments/exp230_models_contacts_v1_multi_from_exp199
LOGS=$HOME/exp230_logs
cd "$REPO" || exit 1

echo "=== [1/2] aggregation-mode generation (577 units) ==="
./run_agg_modes.sh 2>&1 | tail -20

echo "=== [2/2] leak-vs-steps curve (8 checkpoints x 2 modes) ==="
./run_leak_curve.sh 2>&1 | tail -40

echo "=== ALL EVAL GENERATION DONE ==="

#!/usr/bin/env bash
# Launch the FoldBench rescoring on a multi-GPU box, detached. Arguments go to
# score_foldbench_rollouts.py; a run takes tens of minutes, so it runs under nohup
# and its progress is read from the per-shard logs.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/MarinFold"
git fetch -q origin exp250/evals-exploration-notebook
git reset -q --hard origin/exp250/evals-exploration-notebook
git log --oneline -1
cd experiments/exp250_evals_exploration_notebook
export FIGLIB_MACHINE_LABEL="8xA100 node"
nohup "$HOME/nbenv/bin/python" score_foldbench_rollouts.py "$@" > /tmp/score.log 2>&1 &
echo "started pid $! — tail /tmp/score.log and data/foldbench_rescore/*/shard*.log"

#!/usr/bin/env bash
# Poll a marin iris job until it reaches a terminal state, then print its tail.
# Exists as a file because the shell sandbox rejects multi-clause inline commands.
#   usage: _wait_job.sh <job-id> [poll_seconds] [max_polls]
set -u
JOB="$1"
SLEEP="${2:-150}"
MAX="${3:-20}"
PROJ=/home/bizon/git/marin-worktrees/exp163-tpu
s=""
for _ in $(seq 1 "$MAX"); do
  s=$(timeout 150 uv run --project "$PROJ" iris --cluster=marin job summary "$JOB" 2>/dev/null \
      | awk '/^ +[0-9]+ /{printf "%s", $2}')
  case "$s" in
    succeeded|failed|killed) echo "TERMINAL=$s"; break;;
  esac
  sleep "$SLEEP"
done
echo "LAST=$s"
timeout 200 uv run --project "$PROJ" iris --cluster=marin job logs "$JOB" 2>/dev/null \
  | grep -iE "\[rprec\]|R0_all=|Traceback|Error:" | tail -8

#!/usr/bin/env bash
# Poll several marin iris jobs until all reach a terminal state.
set -u
PROJ=/home/bizon/git/marin-worktrees/exp163-tpu
for _ in $(seq 1 24); do
  out=""; alldone=1
  for j in "$@"; do
    s=$(timeout 120 uv run --project "$PROJ" iris --cluster=marin job summary "$j" 2>/dev/null | awk "/^ +[0-9]+ /{printf \"%s\", \$2}")
    out="$out ${j##*-}=$s"
    case "$s" in succeeded|failed|killed) ;; *) alldone=0;; esac
  done
  if [ "$alldone" = 1 ]; then echo "ALL_TERMINAL:$out"; break; fi
  sleep 200
done
echo "LAST:$out"

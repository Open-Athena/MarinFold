#!/usr/bin/env bash
# Sync exp237's SkyRL port to a GPU host and run a command in SkyRL's venv — issue #237.
#
# The host is a REQUIRED argument with no default, deliberately: the machine is
# private infrastructure and is not named anywhere in this repository.
#
#   ./run_on_host.sh --host user@ip -- python main_exp237.py <hydra overrides>
#   ./run_on_host.sh --host user@ip --smoke      # import + register only, no GPUs
#   ./run_on_host.sh --host user@ip --sync       # push the tree and stop
set -euo pipefail

HOST=""
SMOKE=0
SYNC_ONLY=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --sync) SYNC_ONLY=1; shift ;;
    --) shift; ARGS=("$@"); break ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ -z "$HOST" ]]; then
  echo "usage: $0 --host <user@host> [--smoke|--sync] [-- <command>]" >&2
  echo "  --host is required and has no default (private infra)." >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="~/exp237/skyrl"
VENV="~/SkyRL/.venv/bin/python"

echo "[run] syncing $(basename "$HERE") -> $HOST:$REMOTE"
ssh "$HOST" "mkdir -p $REMOTE"
# --delete keeps the remote a faithful mirror; without it a renamed module lingers
# and shadows its replacement on the import path.
rsync -az --delete \
  --exclude '__pycache__' --exclude '.venv' --exclude 'data' \
  "$HERE/" "$HOST:$REMOTE/"
[[ "$SYNC_ONLY" == "1" ]] && exit 0

if [[ "$SMOKE" == "1" ]]; then
  echo "[run] smoke: import + register, no training"
  ssh "$HOST" "cd $REMOTE && PYTHONPATH=$REMOTE $VENV -c \"
import main_exp237, section_rewards, consensus
main_exp237.register_everything(vocab_size=2845)
print('[smoke] registration OK, modes', section_rewards.REWARD_MODES)
import torch; print('[smoke] gpus', torch.cuda.device_count())
\""
  exit 0
fi

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "[run] nothing to run; pass a command after --" >&2
  exit 2
fi
# The venv python is prepended, so a leading `python` would be passed to it as a
# filename ("can't open file '.../python'"). Drop it if present.
if [[ "${ARGS[0]}" == "python" || "${ARGS[0]}" == "python3" ]]; then
  ARGS=("${ARGS[@]:1}")
fi
echo "[run] ${ARGS[*]}"
ssh "$HOST" "cd $REMOTE && PYTHONPATH=$REMOTE $VENV ${ARGS[*]}"

#!/bin/bash
# exp230 -- the BUDGET-MATCHED plain baseline for the multi modes.
#
# Gate A's plain number (0.6058) votes 100 rollouts. The multi modes each use ONE
# rollout, whose ~22 sections are then combined. Comparing those directly answers
# "how good is the model" but not "is the multi format worth anything", because
# the two sit at very different sampling budgets.
#
# This runs plain <contacts-v1> at N_ROLLOUTS=22 -- matching the ~22 sections a
# multi rollout emits -- and the same three aggregation rules are then applied to
# the 22 ROLLOUTS that were applied to the 22 SECTIONS:
#
#   single      one rollout, averaged over the 22   <-> (no multi analogue)
#   best        oracle pick of 22                   <-> multi `best`
#   consensus   vote across 22                      <-> multi `consensus`
#
# Same model, same targets, same sampler, same metric. The only difference is
# whether the 22 candidate contact sets came from 22 independent rollouts or from
# one rollout's 22 sections -- which is exactly the question.
set -u

NGPU=${NGPU:-8}
ROLL=${ROLL:-22}
FT=${FT:-$HOME/exp230_data/checkpoints/hf/step-1988}
TARGETS=${TARGETS:-$HOME/exp230_data/eval577_targets.parquet}
OUT=${OUT:-$HOME/exp230_data/eval/plain_sections}
LOGS=$HOME/exp230_logs
REPO=$HOME/MarinFold/experiments/exp230_models_contacts_v1_multi_from_exp199
PY=$HOME/exp230_vllm/bin/python

for p in "$FT" "$TARGETS"; do [ -e "$p" ] || { echo "FATAL: missing $p"; exit 1; }; done
mkdir -p "$OUT" "$LOGS"
export PATH=$HOME/exp230_vllm/bin:$PATH
cd "$REPO" || exit 1

if pgrep -f "train_local.py" >/dev/null 2>&1; then
  echo "FATAL: train_local.py is running"; exit 1
fi
for i in $(seq 1 240); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
  [ "$busy" -eq 0 ] && break
  sleep 15
done

pids=()
for (( g=0; g<NGPU; g++ )); do
  export VLLM_CACHE_ROOT=$HOME/.vllm_cache_g$g
  CUDA_VISIBLE_DEVICES=$g nohup $PY eval_agg_worker.py \
    --model "$FT" --targets "$TARGETS" --out "$OUT" --mode plain --shard "$g/$NGPU" \
    --n-rollouts "$ROLL" --temperature 1.0 --top-p 0.95 --top-k -1 \
    --tensor-parallel-size 1 \
    > "$LOGS/plain_base_g$g.log" 2>&1 &
  pids+=($!)
  echo "  GPU $g -> shard $g/$NGPU"
  sleep 2
done
wait "${pids[@]}"
echo "ALL DONE -> $OUT"

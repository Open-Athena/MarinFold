#!/bin/bash
# exp230 -- generate the <contacts-v1.multi> rollouts the five aggregation modes
# are scored from, on the 577-unit universe.
#
# One generation pass feeds modes 2-5 (consensus / best / last / second-to-last).
# They differ only in how sections are COMBINED, so generating once and scoring
# offline makes the comparison between them paired rather than four independent
# samples -- and a new rule costs a re-score, not another 8-GPU pass.
#
# Mode 1 (plain <contacts-v1>) is NOT here: it is Gate A's exp82 rollout+resample
# scorer, which is a different and already-validated measurement.
set -u

NGPU=${NGPU:-8}
ROLL=${ROLL:-8}
MAX_SECTIONS=${MAX_SECTIONS:-16}
FT=${FT:-$HOME/exp230_data/checkpoints/hf/step-1988}
TARGETS=${TARGETS:-$HOME/exp230_data/eval577_targets.parquet}
OUT=${OUT:-$HOME/exp230_data/eval/agg_sections}
LOGS=$HOME/exp230_logs
REPO=$HOME/MarinFold/experiments/exp230_models_contacts_v1_multi_from_exp199
PY=$HOME/exp230_vllm/bin/python

for p in "$FT" "$TARGETS"; do [ -e "$p" ] || { echo "FATAL: missing $p"; exit 1; }; done
mkdir -p "$OUT" "$LOGS"
export PATH=$HOME/exp230_vllm/bin:$PATH
cd "$REPO" || exit 1

if pgrep -f "train_local.py" >/dev/null 2>&1; then
  echo "FATAL: train_local.py is running -- the GPU drain would kill it."; exit 1
fi
for i in $(seq 1 240); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
  [ "$busy" -eq 0 ] && break
  [ "$i" = "1" ] && echo "waiting for $busy GPU(s) from the previous stage..."
  sleep 15
done
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
if [ "$busy" -ne 0 ]; then echo "FATAL: GPUs still busy after 60 min"; exit 1; fi

pids=()
for (( g=0; g<NGPU; g++ )); do
  export VLLM_CACHE_ROOT=$HOME/.vllm_cache_g$g
  CUDA_VISIBLE_DEVICES=$g nohup $PY eval_agg_worker.py \
    --model "$FT" --targets "$TARGETS" --out "$OUT" --shard "$g/$NGPU" \
    --n-rollouts "$ROLL" --max-sections "$MAX_SECTIONS" \
    --temperature 1.0 --top-p 0.95 --top-k -1 --tensor-parallel-size 1 \
    > "$LOGS/agg_g$g.log" 2>&1 &
  pids+=($!)
  echo "  GPU $g -> shard $g/$NGPU"
  sleep 2
done
wait "${pids[@]}"
echo "ALL DONE -> $OUT"
echo "Next: python score_agg_modes.py --sections $OUT --targets $TARGETS --out data/"

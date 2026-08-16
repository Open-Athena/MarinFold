#!/bin/bash
# exp230 -- the leak-vs-steps curve: how many contact sets the model emits under
# <contacts-v1> versus <contacts-v1.multi>, at every checkpoint of the run.
#
# This is H2 made falsifiable rather than assumed. #163's arm F emitted ~2.94
# sections under the PLAIN sentinel after 405 steps; #175 got a completely clean
# switch after 2,070. The prediction is that plain-mode section count falls to
# ~1.0 somewhere in between, and the only way to know is to measure every
# checkpoint rather than just the last one.
#
# Both modes, one subset, every checkpoint:
#   plain  -> should collapse to 1.0 (the leak closing)
#   multi  -> should stay high (the format still working)
# A run where BOTH fall to 1.0 has not fixed the leak, it has lost the format.
#
# The subset is a SEEDED random 200 of the 577, identical across checkpoints, so
# the curve is paired. It is deliberately not `--limit 200`: the workers walk
# targets in ascending length, so --limit would hand every checkpoint the
# shortest proteins in the set.
#
# The base model is measured too, in plain mode only, as the step-0 reference.
# It cannot run multi mode -- its tokenizer has no <contacts-v1.multi>; id 7 is
# still <contacts-and-distances-v1>.
set -u

NGPU=${NGPU:-8}
ROLL=${ROLL:-4}
MAX_SECTIONS=${MAX_SECTIONS:-16}      # generous: we are measuring the count, so
                                      # the cap must sit ABOVE any plausible leak
BASE=${BASE:-$HOME/exp230_data/model/exp199}
CKPT_DIR=${CKPT_DIR:-$HOME/exp230_data/checkpoints/hf}
STEPS=${STEPS:-"250 500 750 1000 1250 1500 1750 1988"}
TARGETS=${TARGETS:-$HOME/exp230_data/eval_curve200_targets.parquet}
OUT=${OUT:-$HOME/exp230_data/eval/curve}
LOGS=$HOME/exp230_logs
REPO=$HOME/MarinFold/experiments/exp230_models_contacts_v1_multi_from_exp199
PY=$HOME/exp230_vllm/bin/python

[ -e "$TARGETS" ] || { echo "FATAL: missing $TARGETS"; exit 1; }
mkdir -p "$OUT" "$LOGS"
export PATH=$HOME/exp230_vllm/bin:$PATH
cd "$REPO" || exit 1

if pgrep -f "train_local.py" >/dev/null 2>&1; then
  echo "FATAL: train_local.py is running -- the GPU drain would kill it."; exit 1
fi

drain() {
  pkill -f eval_modes_worker 2>/dev/null || true
  pkill -f _exp82_score_rollout_worker 2>/dev/null || true
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
  done
  for i in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
    [ "$busy" -eq 0 ] && return 0
    sleep 5
  done
  echo "FATAL: GPUs still occupied after 5 min"; return 1
}

one_pass() {   # $1=label  $2=model  $3=mode
  local label=$1 model=$2 mode=$3 pids=()
  local dest="$OUT/$label-$mode"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "=== skip $label/$mode (already has output) ==="; return 0
  fi
  echo "=== curve: $label / $mode ==="
  drain || return 1
  for (( g=0; g<NGPU; g++ )); do
    export VLLM_CACHE_ROOT=$HOME/.vllm_cache_g$g
    CUDA_VISIBLE_DEVICES=$g nohup $PY eval_modes_worker.py \
      --model "$model" --targets "$TARGETS" --out "$dest" \
      --mode "$mode" --shard "$g/$NGPU" \
      --n-rollouts "$ROLL" --max-sections "$MAX_SECTIONS" \
      --temperature 1.0 --top-p 0.95 --top-k -1 \
      --tensor-parallel-size 1 \
      > "$LOGS/curve_${label}_${mode}_g$g.log" 2>&1 &
    pids+=($!)
    sleep 2
  done
  wait "${pids[@]}"
  echo "=== done $label/$mode ==="
}

one_pass base "$BASE" plain          # step-0 reference; base cannot run multi
for s in $STEPS; do
  m="$CKPT_DIR/step-$s"
  [ -d "$m" ] || { echo "skip step-$s (missing)"; continue; }
  one_pass "step-$s" "$m" plain
  one_pass "step-$s" "$m" multi
done
echo "ALL DONE -> $OUT"
echo "Next: python plot_leak_curve.py --curve $OUT --out plots/"

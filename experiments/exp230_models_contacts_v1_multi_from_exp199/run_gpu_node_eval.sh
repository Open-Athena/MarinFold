#!/bin/bash
# exp230 evaluation on 8x A100-80GB -- Gate A, Gate B and the multi-mode report.
#
# Gate A (rprec):  plain-mode R-precision on the 554-protein exp89 benchmark,
#   scored by exp82's vendored worker, which #209 established as THE reference
#   scorer (exp199 reads 0.6103 there against its published 0.5873 -- the gap is
#   the eval pipeline, not the accelerator). The BASE and the FINE-TUNE are
#   scored in the SAME run, on the same machine, with identical settings; the
#   published number is never the comparison point.
# Gate B (modes): plain-mode section count. Target <=1.05 mean and >=95% single
#   section. #163's arm F leaked at 2.94 sections, which is the failure this run
#   exists to fix.
#
# ONE engine per GPU, tensor-parallel-size 1: a 1.5B model fits one A100 with
# room to spare, so 8 independent engines beat one 8-way split.
#
# NB: the inner workers are plain `nohup ... &` background jobs, NOT setsid.
# `wait` must actually block on them -- setsid forks when the child is already a
# process-group leader, and then $! is the short-lived wrapper, so `wait` returns
# immediately and the NEXT phase's drain() kills workers that are still running.
# Detach this SCRIPT instead (setsid nohup ./run_gpu_node_eval.sh), not its
# children.
set -u

WHAT=${WHAT:-all}                 # all | rprec | modes
NGPU=${NGPU:-8}
ROLL=${ROLL:-100}                 # exp82's settled recipe
MODE_ROLL=${MODE_ROLL:-4}         # free-generation sample, as #163
MAX_SECTIONS=${MAX_SECTIONS:-8}
BASE=${BASE:-$HOME/exp230_data/model/exp199}
FT=${FT:-}                        # REQUIRED: the fine-tuned HF export
TARGETS=${TARGETS:-$HOME/exp230_data/eval554_targets.parquet}
OUT=${OUT:-$HOME/exp230_data/eval}
LOGS=$HOME/exp230_logs
REPO=$HOME/MarinFold/experiments/exp230_models_contacts_v1_multi_from_exp199
PY=$HOME/exp230_vllm/bin/python

if [ -z "$FT" ]; then
  echo "FATAL: set FT=<path to the fine-tuned HF export>, e.g."
  echo "  FT=\$HOME/exp230_data/checkpoints/hf/step-1989 $0"
  exit 1
fi
for p in "$BASE" "$FT" "$TARGETS"; do
  [ -e "$p" ] || { echo "FATAL: missing $p"; exit 1; }
done

# ---------------------------------------------------------------------------
# REFUSE to run while the fine-tune is still training. The drain step below
# kill -9's every process holding a GPU, which is correct for a stale vLLM
# engine and catastrophic for an 11-hour training run. Checking the GPUs are
# "busy" is not enough -- that is also what a leftover engine looks like.
# ---------------------------------------------------------------------------
if pgrep -f "train_local.py" >/dev/null 2>&1; then
  echo "FATAL: train_local.py is still running -- the GPU drain below would kill it."
  echo "       Wait for training to finish, or set FORCE_OVER_TRAINING=1 to override."
  [ "${FORCE_OVER_TRAINING:-0}" = "1" ] || exit 1
  echo "       FORCE_OVER_TRAINING=1 set; continuing anyway."
fi

mkdir -p "$OUT" "$LOGS"
export PATH=$HOME/exp230_vllm/bin:$PATH     # vLLM shells out to ninja
cd "$REPO" || exit 1

# Drain the GPUs. vLLM sizes its KV cache from FREE memory at startup and hard
# fails when short, so launching into a stale engine kills every new engine at
# once. vLLM also renames its child to "VLLM::EngineCore", so pkill on the
# worker name matches only the parent and leaves the engines holding ~74 GB.
drain() {
  # Re-checked on EVERY call, not just at startup: this function kill -9's
  # every GPU process, and the cost of getting that wrong is the whole run.
  if pgrep -f "train_local.py" >/dev/null 2>&1 && [ "${FORCE_OVER_TRAINING:-0}" != "1" ]; then
    echo "FATAL: train_local.py appeared -- refusing to drain the GPUs."; return 1
  fi
  pkill -f gen_rollouts_worker 2>/dev/null || true
  pkill -f _exp82_score_rollout_worker 2>/dev/null || true
  pkill -f eval_modes_worker 2>/dev/null || true
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
  done
  for i in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
    [ "$busy" -eq 0 ] && return 0
    echo "  waiting for $busy GPU(s) to release memory ($i)"; sleep 5
  done
  echo "FATAL: GPUs still occupied after 5 min:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  return 1
}

# ---------------------------------------------------------------------------
# Gate A: both models concurrently, 4 GPUs each. Concurrent rather than
# sequential so "same batch" is literally true -- same machine, same driver,
# same wall clock, same everything except the weights.
# ---------------------------------------------------------------------------
run_rprec() {
  echo "=== Gate A: R-precision, base + fine-tune concurrently, $ROLL rollouts/protein ==="
  drain || return 1
  local half=$((NGPU / 2)) pids=()
  for (( g=0; g<NGPU; g++ )); do
    if [ "$g" -lt "$half" ]; then model="$BASE"; label="base";    shard=$g;              n=$half
    else                          model="$FT";   label="finetune"; shard=$((g - half));   n=$half
    fi
    export VLLM_CACHE_ROOT=$HOME/.vllm_cache_g$g
    CUDA_VISIBLE_DEVICES=$g nohup $PY _exp82_score_rollout_worker.py \
      --model "$model" --targets "$TARGETS" --out "$OUT/rprec" --label "$label" \
      --shard "$shard/$n" --n-rollouts "$ROLL" \
      --temperature 1.0 --top-p 0.95 --top-k -1 \
      --no-per-request-seed --gpu-frac 0.90 \
      > "$LOGS/eval_rprec_${label}_g$g.log" 2>&1 &
    pids+=($!)
    echo "  GPU $g -> $label shard $shard/$n"
    sleep 3
  done
  wait "${pids[@]}"
  echo "=== Gate A generation done ==="
}

# ---------------------------------------------------------------------------
# Gate B + the multi-mode report. Three passes, each across all 8 GPUs. Cheap
# next to Gate A (4 rollouts x 554 vs 100 x 554), so sequential is fine.
#   finetune/plain -> Gate B, the leak gate
#   finetune/multi -> the RL hand-off report
#   base/plain     -> the control: what "no leak" looks like on this harness
# ---------------------------------------------------------------------------
run_modes() {
  for spec in "finetune:$FT:plain" "finetune:$FT:multi" "base:$BASE:plain"; do
    IFS=: read -r label model mode <<< "$spec"
    echo "=== modes: $label / $mode ==="
    drain || return 1
    local pids=()
    for (( g=0; g<NGPU; g++ )); do
      export VLLM_CACHE_ROOT=$HOME/.vllm_cache_g$g
      CUDA_VISIBLE_DEVICES=$g nohup $PY eval_modes_worker.py \
        --model "$model" --targets "$TARGETS" --out "$OUT/modes/$label-$mode" \
        --mode "$mode" --shard "$g/$NGPU" \
        --n-rollouts "$MODE_ROLL" --max-sections "$MAX_SECTIONS" \
        --temperature 1.0 --top-p 0.95 --top-k -1 \
        --tensor-parallel-size 1 \
        > "$LOGS/eval_modes_${label}_${mode}_g$g.log" 2>&1 &
      pids+=($!)
      sleep 2
    done
    wait "${pids[@]}"
    echo "=== modes $label/$mode done ==="
  done
}

case "$WHAT" in
  rprec) run_rprec ;;
  modes) run_modes ;;
  all)   run_rprec && run_modes ;;
  *)     echo "FATAL: WHAT must be all|rprec|modes"; exit 1 ;;
esac
echo "ALL DONE -> $OUT"
echo
echo "Next -- reduce the votes into the Gate A verdict:"
echo "  python score_gate_a.py --rprec $OUT/rprec --targets $TARGETS \\"
echo "      --base base --finetune finetune --out $OUT/gate_a"
echo "Gate B (section counts) comes from summarize_modes.py over $OUT/modes/*."

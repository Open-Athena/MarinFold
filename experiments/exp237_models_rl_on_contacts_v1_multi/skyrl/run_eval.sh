#!/usr/bin/env bash
# Score one exp237 arm the way #230 scored itself — issue #237.
#
# **The metric is not re-derived here.** #209 established exp82's
# `score_rollout_worker.py` + exp89's `compute_metrics` as the reference scorer
# (exp199 reads 0.6103 there against its published 0.5873), and #230 already
# wraps both in `eval_agg_worker.py` / `score_agg_modes.py`. This script exports
# the SkyRL checkpoint and then calls those, unchanged, on the same 577-unit
# target file (#89's 554 + #226's 23) with the same sampler and the same
# `--n-rollouts`. Anything else and the arm's number stops being comparable with
# #230's 0.5673 or with #180's frontier.
#
#   ARM=M-C STEP=60 ./run_eval.sh
set -u

ARM=${ARM:?set ARM}
STEP=${STEP:-}                       # blank = the newest global_step_N
ROLL=${ROLL:-8}                      # #230 used 8; keep it for comparability
NGPU=${NGPU:-8}
ROOT=${ROOT:-$HOME/exp237_data}
TARGETS=${TARGETS:-$HOME/exp230_data/eval577_targets.parquet}
EXP230=$HOME/MarinFold/experiments/exp230_models_contacts_v1_multi_from_exp199
HERE=$HOME/exp237/skyrl
PY230=$HOME/exp230_vllm/bin/python
PYSKY=$HOME/SkyRL/.venv/bin/python
LOGS=$HOME/exp237_logs

TAG=$(echo "$ARM" | tr 'A-Z-' 'a-z_')
CKPT_ROOT=$ROOT/ckpts_$TAG
if [ -z "$STEP" ]; then
  STEP=$(ls "$CKPT_ROOT" 2>/dev/null | sed -n 's/^global_step_//p' | sort -n | tail -1)
fi
[ -n "$STEP" ] || { echo "FATAL: no global_step_N under $CKPT_ROOT"; exit 1; }
CKPT=$CKPT_ROOT/global_step_$STEP
HF=$ROOT/hf_${TAG}_step${STEP}
OUT=$ROOT/eval/${TAG}_step${STEP}
mkdir -p "$LOGS" "$OUT"

# 1. FSDP shard -> HF directory. At world_size 1 (the only configuration this
#    experiment may use) rank 0 holds the whole model, so this is a load-and-save.
if [ ! -f "$HF/config.json" ]; then
  echo "=== exporting $CKPT -> $HF ==="
  PYTHONPATH=$HERE $PYSKY "$HERE/export_skyrl_checkpoint.py" \
      --ckpt "$CKPT" --out "$HF" > "$LOGS/export_${TAG}_${STEP}.log" 2>&1 || {
    echo "FATAL: export failed"; tail -20 "$LOGS/export_${TAG}_${STEP}.log"; exit 1; }
  # The export copies the tokenizer but writes config.json from the training
  # config, which carries rope in the transformers-5 form only. Repair it so the
  # vLLM the evaluator uses cannot silently fall back to a default rope -- that
  # bug costs 0.76 nats/token and already forced one retraction in #163.
  PYTHONPATH=$HERE $PYSKY - "$HF" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "exp237" / "skyrl"))
from prepare_model import repair_rope
p = Path(sys.argv[1]) / "config.json"
cfg = json.loads(p.read_text())
notes = repair_rope(cfg)
if cfg.get("rope_theta") is None:
    raise SystemExit("FATAL: no top-level rope_theta after repair")
p.write_text(json.dumps(cfg, indent=2) + "\n")
print("[eval] rope:", notes or "already 4.x-readable")
PYEOF
fi

# 2. Drain, then generate. `nohup ... &` and a real `wait`: setsid would fork and
#    $! would be the short-lived wrapper, so the next stage would kill workers
#    that are still running (#230's launcher note).
for i in $(seq 1 120); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
  [ "$busy" -eq 0 ] && break
  [ "$i" = "1" ] && echo "waiting for $busy GPU(s) to drain..."
  sleep 10
done
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
[ "$busy" -eq 0 ] || { echo "FATAL: GPUs still busy"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 1; }

echo "=== agg sections: $ARM step $STEP, $ROLL rollouts x 577 proteins ==="
# vLLM shells out to `ninja` when it compiles, and finds it only on PATH -- the
# venv's bin dir is not on PATH just because its python is invoked by absolute
# path. Without this every engine dies with `FileNotFoundError: 'ninja'` wrapped
# in "Engine core initialization failed", which names neither ninja nor PATH.
# #230's launchers export it for the same reason.
export PATH=$HOME/exp230_vllm/bin:$PATH
cd "$EXP230" || exit 1
pids=()
for (( g=0; g<NGPU; g++ )); do
  export VLLM_CACHE_ROOT=$HOME/.vllm_cache_g$g
  CUDA_VISIBLE_DEVICES=$g nohup $PY230 eval_agg_worker.py \
    --model "$HF" --targets "$TARGETS" --out "$OUT/agg_sections" --mode multi \
    --shard "$g/$NGPU" --n-rollouts "$ROLL" --max-sections 64 \
    --temperature 1.0 --top-p 0.95 --top-k -1 --tensor-parallel-size 1 \
    > "$LOGS/agg_${TAG}_${STEP}_g$g.log" 2>&1 &
  pids+=($!)
  sleep 2
done
wait "${pids[@]}"

# 3. Score. The five #230 aggregation modes, every eval cut, offline.
#    Refuse to "succeed" on an empty generation: `score_agg_modes.py` exits 0 with
#    a one-line complaint when it finds no parquets, and a pipeline that treats
#    that as a completed stage reports an arm as evaluated when it was not.
n_parts=$(ls "$OUT/agg_sections"/*.parquet 2>/dev/null | wc -l)
if [ "$n_parts" -eq 0 ]; then
  echo "FATAL: generation produced no section parquets for $ARM step $STEP."
  echo "       First worker log:"; tail -25 "$LOGS/agg_${TAG}_${STEP}_g0.log"
  exit 1
fi
echo "=== scoring $ARM step $STEP ($n_parts shards) ==="
$PY230 score_agg_modes.py --sections "$OUT/agg_sections" --targets "$TARGETS" \
    --out "$OUT" --label "${TAG}_step${STEP}" 2>&1 | tee "$LOGS/score_${TAG}_${STEP}.log"
echo "ALL DONE -> $OUT"

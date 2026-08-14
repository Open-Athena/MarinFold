#!/bin/bash
# exp230 rollout generation on a standalone 8-GPU node (8x A100-80GB).
#
# Not an iris job: this node is a plain SSH box, so the fan-out is 8 local
# processes rather than 8 pods. Used because the marin v5p-16 pool sat at zero
# ready / zero booting / zero registered demand while training queued on it, and
# because the v6e pool preempts hard.
#
# FOUR things here were each learned the expensive way:
#
#  1. ONE engine per GPU, walking that GPU's whole shard list. A process per
#     shard rebuilt the engine every time -- 128 builds across the fleet -- and
#     eight simultaneous builds raced on the inductor compile cache, killing 5
#     of 8 GPUs at init. Each then silently advanced to its next shard, so
#     shards were skipped at speed while every GPU read 0% utilisation.
#  2. --enforce-eager removes the compile path entirely, which removes that race
#     and minutes of startup, at a modest per-token cost.
#  3. vLLM RENAMES its child to "VLLM::EngineCore", so pkill on the worker name
#     kills parents and orphans engines holding ~74 GB each. vLLM sizes its KV
#     cache from FREE memory at startup, so the next launch then fails all 8
#     engines at once. Kill CUDA holders BY PID from nvidia-smi, and refuse to
#     start if the GPUs have not drained.
#  4. chunk x n_rollouts must reach max_num_seqs. One protein is only n_rollouts
#     prompts; a per-protein generate() call filled ~2% of the batch.
#
# NOT tensor-parallel: a 1.5B model fits one A100 with room to spare, so eight
# independent engines give 8x the batch concurrency and no cross-GPU traffic.
#
#     MAXSHARD=64 ./run_gpu_node_gen.sh     # leave shards 64+ to the TPU fleet
set -u
NSHARDS=${NSHARDS:-128}
MAXSHARD=${MAXSHARD:-128}    # exclusive; shards at or above this belong to the TPU fleet
NGPU=${NGPU:-8}
ROLL=${ROLL:-32}
CHUNK=${CHUNK:-16}
MAXSEQ=${MAXSEQ:-512}
OUT=$HOME/exp230_data/rollouts
mkdir -p $OUT $HOME/exp230_logs
export PATH=$HOME/exp230_vllm/bin:$PATH     # vLLM shells out to ninja

# Wait for the GPUs to actually drain before starting. vLLM sizes its KV cache
# from FREE memory at startup and hard-fails if it is short -- so relaunching
# while the previous run still holds memory kills all 8 engines at once, which
# is exactly what happened when this fleet was re-pointed at a smaller shard
# range. pkill returning is not the same as the memory being back.
# Kill EVERYTHING holding a GPU, then verify. vLLM renames its child process to
# "VLLM::EngineCore", so `pkill -f gen_rollouts_worker` matches the parent and
# leaves eight engines alive holding 74 GB each. vLLM sizes its KV cache from
# FREE memory at startup and hard-fails when short, so a relaunch into those
# orphans kills all 8 new engines at once -- which is exactly what happened.
pkill -f gen_rollouts_worker 2>/dev/null || true
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
drained=0
for i in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
  if [ "$busy" -eq 0 ]; then drained=1; break; fi
  echo "waiting for $busy GPU(s) to release memory ($i)"; sleep 5
done
if [ "$drained" -ne 1 ]; then
  # Launching anyway would fail all 8 engines. Refuse loudly instead.
  echo "FATAL: GPUs still occupied after 5 min:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi

cd $HOME/MarinFold/experiments/exp230_models_contacts_v1_multi_from_exp199
for (( g=0; g<NGPU; g++ )); do
  LIST=$(python3 -c "print(','.join(str(s) for s in range($g, $MAXSHARD, $NGPU)))")
  [ -z "$LIST" ] && continue
  CUDA_VISIBLE_DEVICES=$g setsid nohup $HOME/exp230_vllm/bin/python gen_rollouts_worker.py \
    --model $HOME/exp230_data/model/exp199 \
    --targets $HOME/exp230_data/targets_multi.parquet \
    --out $OUT --shard "$LIST/$NSHARDS" \
    --n-rollouts $ROLL --chunk $CHUNK --max-num-seqs $MAXSEQ \
    --gpu-memory-utilization 0.90 --tensor-parallel-size 1 --enforce-eager \
    > $HOME/exp230_logs/gpu$g.log 2>&1 &
  echo "GPU $g -> shards $LIST"
  sleep 3
done
echo "launched, one engine per GPU, $((CHUNK*ROLL)) prompts/call"

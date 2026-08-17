#!/usr/bin/env bash
# Train one exp237 arm on the 8x A100 node — issue #237.
#
# Runs ON the node (run_on_host.sh syncs this file here). One arm at a time on
# all 8 cards, because generation dominates a multi-mode step: a rollout is
# ~6,760 tokens against plain's ~500, so the engines are the throughput and
# splitting the node between two arms would starve both.
#
# THE PLACEMENT IS NOT NEGOTIABLE. `policy_num_gpus_per_node=1`, i.e. the policy
# is UNSHARDED. #208 established that SkyRL's FSDP policy sharding diverges from
# the inference engines at any shard count and the first weight sync pushes the
# divergent copy into them (trainer/engine logprob gap 1.33 nats sharded vs 0.017
# unsharded); a zero-LR control proved it was the sync and not the gradient. A
# sharded run looks superficially fine -- it trains, it logs, it reports a
# falling reward -- while generating from a destroyed policy. Throughput comes
# from `colocate_all=false` plus the spare cards given to engines instead.
#
#   ARM=M-C LR=1e-6 STEPS=80 ./run_arm.sh
set -u

ARM=${ARM:?set ARM to one of M-C, M-F, M-B, M-0}
LR=${LR:-1e-6}
STEPS=${STEPS:-80}
GROUP=${GROUP:-8}                 # generator.n_samples_per_prompt
PROMPTS=${PROMPTS:-8}             # trainer.train_batch_size, in PROMPTS
GEN_TOKENS=${GEN_TOKENS:-7000}
CKPT_EVERY=${CKPT_EVERY:-20}
DATA=${DATA:-$HOME/exp237_data/skyrl_multi_10k.parquet}
MODEL=${MODEL:-$HOME/exp237_data/model/exp230_step1988_bf16}
ROOT=${ROOT:-$HOME/exp237_data}
HERE=$HOME/exp237/skyrl
PY=$HOME/SkyRL/.venv/bin/python

case "$ARM" in
  # Arm M-C -- the arm the hypothesis predicts. Per-section leave-one-out
  # marginal on the rollout's OWN consensus, so the reward is computed on the
  # same kind of object the metric scores and its credit assignment is WITHIN
  # the sequence, where the policy gradient can reach it.
  M-C) MODE=section_consensus; EST=contacts_section ;;
  # Arm M-F -- can the model learn to commit to its own best work? #230 measured
  # last 0.4566 against best 0.5342: +0.078 of headroom in selection alone.
  M-F) MODE=final_f1;          EST=grpo ;;
  # Arm M-B -- ORACLE. Raises the ceiling rather than the selector. Not
  # deployable on its own; the upper rung of the ladder.
  M-B) MODE=best_f1;           EST=grpo ;;
  # Arm M-0 -- zero-LR control. #208 needed one to prove that FSDP sharding, not
  # the gradient, was destroying the policy. Cheap, and it makes every other arm
  # interpretable: whatever M-0 does is the harness, not the reward.
  M-0) MODE=section_consensus; EST=contacts_section; LR=0.0 ;;
  *) echo "FATAL: unknown ARM $ARM"; exit 2 ;;
esac

for p in "$DATA" "$MODEL"; do [ -e "$p" ] || { echo "FATAL: missing $p"; exit 1; }; done

TAG=$(echo "$ARM" | tr 'A-Z-' 'a-z_')
RUN=exp237_${TAG}_lr${LR}
CKPT=$ROOT/ckpts_$TAG
EXPORT=$ROOT/exports_$TAG
LOG=$HOME/exp237_logs/${RUN}.log
mkdir -p "$CKPT" "$EXPORT" "$HOME/exp237_logs"

# Refuse to start on busy cards. vLLM sizes its KV cache from FREE memory, so
# launching into a stale engine kills every new engine at once -- and vLLM
# renames its child to VLLM::EngineCore, so a pkill on the worker name matches
# only the parent and leaves ~74 GB held per card.
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
if [ "$busy" -ne 0 ]; then
  echo "FATAL: $busy GPU(s) still busy; drain before starting $ARM"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi

echo "=== exp237 $ARM: reward_mode=$MODE estimator=$EST lr=$LR ==="
echo "    group=$GROUP prompts/step=$PROMPTS -> $((GROUP * PROMPTS)) rollouts/step, $STEPS steps"
echo "    log -> $LOG"
cd "$HERE" || exit 1
export PYTHONPATH=$HERE
export VLLM_USE_V1=1
# Ray's raylet dies with "Too many open files" at the login shell's default 1024
# descriptors: six vLLM engines plus a policy and a ref worker open far more
# sockets than that between them, and the failure surfaces three minutes in as
# `LocalRayletDiedError` rather than as anything about descriptors. The hard
# limit here is 1,048,576, so raising the soft limit is free.
ulimit -n 65536 || echo "WARNING: could not raise the open-file limit; ray may die"

$PY main_exp237.py \
  trainer.policy.model.path="$MODEL" \
  data.train_data="['$DATA']" \
  data.val_data="[]" \
  trainer.eval_interval=-1 \
  trainer.eval_before_train=false \
  trainer.algorithm.advantage_estimator="$EST" \
  reward_mode="$MODE" \
  vocab_size=2845 \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=6 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  generator.inference_engine.max_num_batched_tokens=8192 \
  generator.n_samples_per_prompt="$GROUP" \
  generator.max_input_length=2048 \
  generator.sampling_params.max_generate_length="$GEN_TOKENS" \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.top_k=-1 \
  trainer.train_batch_size="$PROMPTS" \
  trainer.policy_mini_batch_size="$PROMPTS" \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_prompt_length=2048 \
  trainer.epochs=1 \
  trainer.max_training_steps="$STEPS" \
  trainer.policy.optimizer_config.lr="$LR" \
  trainer.ckpt_path="$CKPT" \
  trainer.ckpt_interval="$CKPT_EVERY" \
  trainer.export_path="$EXPORT" \
  trainer.logger=console \
  trainer.project_name=exp237 \
  trainer.run_name="$RUN" \
  > "$LOG" 2>&1
rc=$?
echo "=== $ARM exited rc=$rc ==="
# A tripped kill criterion is a RESULT, not a crash: the generator raises on
# purpose when a preregistered diversity gate is violated three batches running,
# and the last checkpoint under $CKPT is what gets evaluated.
grep -c "KILL CRITERION" "$LOG" 2>/dev/null | sed 's/^/kill-criterion hits: /'
tail -5 "$LOG"
exit $rc

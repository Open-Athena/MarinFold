#!/usr/bin/env bash
# The whole of exp237, unattended, on the 8x A100 node — issue #237.
#
# Stages, in the order they must happen:
#
#   0. prepare  -- #230's fp32 export -> bf16 + a 4.x-readable rope + a
#                  pass-through chat template baked into the tokenizer
#   1. dataset  -- #208's 10k prompt pool with the mode marker swapped
#   2. M-0      -- zero-LR control. Runs FIRST because it is the cheapest way to
#                  learn that the harness itself is not destroying the policy;
#                  #208 needed exactly this control to prove that FSDP sharding
#                  rather than the gradient was the culprit.
#   3. M-C      -- the arm the hypothesis predicts, then its eval
#   4. M-F      -- final-section reward, then its eval
#   5. M-B      -- best-section (ORACLE) reward, then its eval
#
# A failed stage does NOT stop the pipeline: a tripped kill criterion is a
# result, and a later arm is still worth running. Every stage's exit code lands
# in $HOME/exp237_logs/pipeline.status.
set -u

ROOT=${ROOT:-$HOME/exp237_data}
LOGS=$HOME/exp237_logs
HERE=$HOME/exp237/skyrl
SRC=${SRC:-$HOME/exp230_data/checkpoints/hf/step-1988}
POOL=${POOL:-$HOME/exp208_skyrl/data/skyrl_train_10k.parquet}
PYSKY=$HOME/SkyRL/.venv/bin/python

STEPS_M0=${STEPS_M0:-8}
STEPS_MC=${STEPS_MC:-80}
STEPS_MF=${STEPS_MF:-80}
STEPS_MB=${STEPS_MB:-80}
# 1e-5, not #208's 1e-6. Every arm here hands the optimiser an advantage
# NORMALISED to unit spread (M-C by construction, M-F/M-B by GRPO), which is the
# regime where #208's 1e-6 arms simply never moved: arm C v1 finished at KL
# 0.0004 and arm D v1 at 0.0014, both indistinguishable from their warm start.
# #208's own instruction is that a null result at a learning rate that does not
# move the policy is not a result. 1e-5 is the rate at which its normalised arm
# (D v2) reached KL 0.084 -- real movement, and an order of magnitude below the
# 4e-5 that diverged to 3.96.
LR=${LR:-1e-5}
SKIP_PREP=${SKIP_PREP:-0}
ARMS=${ARMS:-"M-0 M-C M-F M-B eval-M-0"}

mkdir -p "$ROOT" "$LOGS"
STATUS=$LOGS/pipeline.status
: > "$STATUS"

note() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }
stage() {
  local name=$1; shift
  note "START $name"
  local t0=$SECONDS
  ( "$@" ) >> "$LOGS/pipeline.log" 2>&1
  local rc=$?
  note "END   $name rc=$rc after $(( (SECONDS - t0) / 60 ))m"
  return $rc
}

if [ "$SKIP_PREP" != "1" ]; then
  stage prepare-model env PYTHONPATH=$HERE $PYSKY "$HERE/prepare_model.py" \
      --src "$SRC" --out "$ROOT/model/exp230_step1988_bf16" --verify || exit 1
  stage build-dataset env PYTHONPATH=$HERE $PYSKY "$HERE/build_multi_dataset.py" \
      --src "$POOL" --out "$ROOT/skyrl_multi_10k.parquet" || exit 1
fi

run_arm()  { ARM=$1 LR=$2 STEPS=$3 CKPT_EVERY=$4 ROOT=$ROOT bash "$HERE/run_arm.sh"; }
run_eval() { ARM=$1 ROOT=$ROOT bash "$HERE/run_eval.sh"; }

# ARMS selects which of them to run, so a stage that has already completed (or a
# node that has to be handed back) does not force the whole pipeline again.
case " $ARMS " in *" M-0 "*)
  stage train-M-0 run_arm M-0 0.0 "$STEPS_M0" "$((STEPS_M0 / 2))" ;; esac
case " $ARMS " in *" M-C "*)
  stage train-M-C run_arm M-C "$LR" "$STEPS_MC" "$((STEPS_MC / 4))"
  stage eval-M-C  run_eval M-C ;; esac
case " $ARMS " in *" M-F "*)
  stage train-M-F run_arm M-F "$LR" "$STEPS_MF" "$((STEPS_MF / 4))"
  stage eval-M-F  run_eval M-F ;; esac
case " $ARMS " in *" M-B "*)
  stage train-M-B run_arm M-B "$LR" "$STEPS_MB" "$((STEPS_MB / 4))"
  stage eval-M-B  run_eval M-B ;; esac
case " $ARMS " in *" eval-M-0 "*)
  stage eval-M-0  run_eval M-0 ;; esac

note "PIPELINE COMPLETE"
cat "$STATUS"

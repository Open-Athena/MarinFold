# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert one exp108 sweep run to HuggingFace format (issue #108).

Pick the sweep point to export with ``EXP108_EXPORT_LR`` (default ``1e-3``,
#75's winner). The contacts-v1 tokenizer is co-located with the exported weights
(hard rule). This runs as a CPU executor step — launch its driver with
``--priority batch`` too.

The training used **direct dispatch** with an explicit ``output_path`` per run
(``…/MarinFold/exp108_qwen_3b_contacts_v1/checkpoints/<run_name>``), so
checkpoints land under that dir. Confirm the exact ``step-{N}`` subdir (and any
run-id component ``run_levanter_train_lm`` may append) against the S3 listing
before running — set ``CHECKPOINT_PATH`` accordingly.

Usage (after the target checkpoint exists on S3)::

    uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \\
        --enable-extra-resources --memory=48GB --disk=32GB --cpu=4 --extra cpu \\
        -e EXP108_EXPORT_LR 1e-3 \\
        -- python -m export_qwen_3b_contacts_v1
"""

import os

from marin.execution import executor_main

from contacts_v1_train_common import CONTACTS_V1_S3_PREFIX, build_hf_export_step
from dispatch_train import build_on_pod_config
from train_qwen_3b_contacts_v1_sweep import (
    NUM_TRAIN_STEPS,
    SEQ_LEN,
    TRAIN_BATCH,
    WARMUP,
    WEIGHT_DECAY,
    protein_qwen3_3b,
    resources,
    run_name_for,
)

EXPORT_LR = float(os.environ.get("EXP108_EXPORT_LR", "1e-3"))
RUN_NAME = run_name_for(EXPORT_LR)
OUTPUT_PATH = f"{CONTACTS_V1_S3_PREFIX}/checkpoints/{RUN_NAME}"

# Final step (= num_train_steps - 1). Override to export an earlier checkpoint.
CHECKPOINT_STEP = NUM_TRAIN_STEPS - 1
# Concrete checkpoint dir. CONFIRM the exact layout against the S3 listing after
# the run — run_levanter_train_lm derives the checkpointer base from OUTPUT_PATH;
# if it appends a run-id subdir, insert it here.
CHECKPOINT_PATH = f"{OUTPUT_PATH}/checkpoints/step-{CHECKPOINT_STEP}"

# Rebuild the run's config to recover the levanter TrainerConfig the export needs.
_on_pod = build_on_pod_config(
    run_name=RUN_NAME,
    model_config=protein_qwen3_3b,
    learning_rate=EXPORT_LR,
    num_train_steps=NUM_TRAIN_STEPS,
    train_batch_size=TRAIN_BATCH,
    seq_len=SEQ_LEN,
    weight_decay=WEIGHT_DECAY,
    warmup=WARMUP,
    output_path=OUTPUT_PATH,
    resources=resources,
    wandb_name=RUN_NAME,
)

hf_export = build_hf_export_step(
    trainer=_on_pod.train_config.trainer,
    model_config=protein_qwen3_3b,
    checkpoint_path=CHECKPOINT_PATH,
    name_prefix=RUN_NAME,
    checkpoint_step=CHECKPOINT_STEP,
)


if __name__ == "__main__":
    executor_main(steps=[hf_export])

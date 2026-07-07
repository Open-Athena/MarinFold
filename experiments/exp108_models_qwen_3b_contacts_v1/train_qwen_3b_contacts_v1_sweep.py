# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""3B Qwen3 contacts-v1 sweep, 16 epochs, on CoreWeave — direct batch dispatch (issue #108).

Scales **Eric's #75 tuning sweep** up: keep #75's tuned recipe and architecture
family, change only (1) ~1.5B → ~3B params and (2) 8 → 16 epochs; goal is to beat
#75's best eval loss (~2.7).

Each LR is submitted as its OWN Fray job at iris **batch** priority via
``dispatch_train.dispatch_training_run`` (grug-style direct dispatch — NOT the
marin executor, which can't set a priority band; see ``dispatch_train.py`` and
the README "Batch priority" section).

Model — Qwen3, **#75's exact 1.5B width** (hidden 2048, ff 8192, 32 heads / 8 KV,
head_dim 64, ``Llama3RotaryEmbeddingsConfig``) with layers **doubled 24 → 48**
→ ~2.9B params. Depth-only scaling keeps #75's width so its tuned LR/wd transfer.
seq 8192; vocab (~2845) from the contacts-v1 tokenizer at model-init.

Sweep — peak **LR ∈ {5e-4, 1e-3, 2e-3}** at fixed **wd 0.2**, 10% warmup, cosine,
batch 128 × seq 8192, **16 epochs** (~71,312 steps). One single-node 8×H100 job
per LR.

Env knobs (so a smoke run needs no code edit):
    EXP108_LRS         comma list of peak LRs (default "5e-4,1e-3,2e-3");
                       set ONE value for a single smoke run
    EXP108_TRAIN_BATCH global batch in sequences (default 128, = #75)
    EXP108_REPLICAS    number of 8×H100 nodes per run (default 1 = 8 GPUs)
    EXP108_MAX_STEPS   cap steps for a smoke run (default: full 16-epoch count)

Launch (batch priority — required by #108):

    WANDB_API_KEY=<key> uv run iris --cluster=cw-rno2a job run --no-wait \\
        --priority batch --enable-extra-resources --cpu=2 --memory=6GB --disk=16GB \\
        --extra gpu -e WANDB_API_KEY <key> \\
        -- python -m train_qwen_3b_contacts_v1_sweep

The driver is a tiny CPU coordinator that submits the GPU gang(s). Do the FIRST
run as a single-LR, ~50-step smoke test to confirm the batch fits on one node,
that the job reports the batch band, and that the S3 tokenize-on-the-fly cache
builds: add ``-e EXP108_LRS 1e-3 -e EXP108_MAX_STEPS 50``.
"""

import dataclasses
import math
import os

from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config

from contacts_v1_train_common import CONTACTS_V1_S3_PREFIX, PROTEIN_RESOURCES_H100
from dispatch_train import dispatch_training_run

# Attention backend. On GPU levanter defaults to NVTE (Transformer Engine), but
# TE is NOT installed in the `--extra gpu` env, so it silently falls back to the
# VANILLA O(seq^2) reference kernel — catastrophic at seq 8192 (~52k tok/s in the
# first smoke run). JAX_FLASH is levanter's pallas/Triton flash kernel (O(seq)
# memory, no TE dependency). Override with EXP108_ATTN=nvte|vanilla|default.
_ATTN = os.environ.get("EXP108_ATTN", "jax_flash").upper()
ATTN_BACKEND = AttentionBackend[_ATTN] if _ATTN else None

# Gradient (activation) checkpointing. Default ON (Qwen3 default) — needed at
# high per-GPU batch. Turning it OFF removes the ~25-33% recompute tax but stores
# all layer activations (much more memory) — only viable at low per-GPU batch
# (e.g. multi-node). Toggle with EXP108_GRAD_CKPT=0.
_GRAD_CKPT = os.environ.get("EXP108_GRAD_CKPT", "1") != "0"

# --- Qwen3 ~3B shape (#75's 1.5B width, depth doubled 24 → 48) ---------------
protein_qwen3_3b = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=48,  # 2× #75's 24 → ~2.9B params; head_dim defaults to 2048/32=64 (= #75)
    rope=Llama3RotaryEmbeddingsConfig(),
    attn_backend=ATTN_BACKEND,
    gradient_checkpointing=_GRAD_CKPT,
)

# --- Recipe (tracks #75) -----------------------------------------------------
SEQ_LEN = 8192
EPOCHS = 16
WEIGHT_DECAY = 0.2  # #75 winner
WARMUP = 0.1        # 10%, as #75
# contacts-v1 train split (exp53 generation_stats.json): 4,129,682 docs /
# 4,672,623,743 tokens.
TRAIN_TOKENS = 4_672_623_743

TRAIN_BATCH = int(os.environ.get("EXP108_TRAIN_BATCH", "128"))
REPLICAS = int(os.environ.get("EXP108_REPLICAS", "1"))

# 16 epochs, computed from tokens so it stays correct if the batch changes.
STEPS_PER_EPOCH = math.ceil(TRAIN_TOKENS / (TRAIN_BATCH * SEQ_LEN))
NUM_TRAIN_STEPS = EPOCHS * STEPS_PER_EPOCH
_max_steps_env = os.environ.get("EXP108_MAX_STEPS")
if _max_steps_env:
    NUM_TRAIN_STEPS = min(NUM_TRAIN_STEPS, int(_max_steps_env))

# LR sweep around #75's winning 1e-3.
_lrs_env = os.environ.get("EXP108_LRS", "5e-4,1e-3,2e-3")
SWEEP_LRS = [float(x) for x in _lrs_env.split(",") if x.strip()]

resources = PROTEIN_RESOURCES_H100
if REPLICAS != 1:
    resources = dataclasses.replace(PROTEIN_RESOURCES_H100, replicas=REPLICAS)

# W&B routing. The pod does NOT inherit the launcher's shell, so forward the key
# from the driver env (set at launch with `-e WANDB_API_KEY <key>`). Never hard-coded.
_env_vars = {"WANDB_ENTITY": "open-athena"}
if os.environ.get("WANDB_API_KEY"):
    _env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]


def _lr_tag(lr: float) -> str:
    """`1e-3`-style tag for run names (mirrors #75's `...-lr1e-3-wd0p2`)."""
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


# Optional suffix to isolate a run's output path (e.g. throughput-probe runs, so
# they don't resume from an earlier run's checkpoint at the shared path).
_RUN_SUFFIX = os.environ.get("EXP108_RUN_SUFFIX", "")


def run_name_for(lr: float) -> str:
    base = f"plm-exp108-cv1-3b-e{EPOCHS}-lr{_lr_tag(lr)}-wd0p2"
    return f"{base}-{_RUN_SUFFIX}" if _RUN_SUFFIX else base


def main() -> None:
    print(
        f"[exp108] Qwen3-3B contacts-v1 sweep (direct batch dispatch): "
        f"LRs={[_lr_tag(l) for l in SWEEP_LRS]} wd={WEIGHT_DECAY} batch={TRAIN_BATCH} "
        f"seq={SEQ_LEN} replicas={REPLICAS} ({REPLICAS * 8} H100/run) | "
        f"{STEPS_PER_EPOCH} steps/epoch × {EPOCHS} epochs = {EPOCHS * STEPS_PER_EPOCH} steps"
        + (f" (capped to {NUM_TRAIN_STEPS} for smoke run)" if _max_steps_env else "")
        + f" | {len(SWEEP_LRS)} job(s)"
    )
    jobs = []
    for lr in SWEEP_LRS:
        name = run_name_for(lr)
        output_path = f"{CONTACTS_V1_S3_PREFIX}/checkpoints/{name}"
        job = dispatch_training_run(
            run_name=name,
            model_config=protein_qwen3_3b,
            learning_rate=lr,
            num_train_steps=NUM_TRAIN_STEPS,
            train_batch_size=TRAIN_BATCH,
            seq_len=SEQ_LEN,
            weight_decay=WEIGHT_DECAY,
            warmup=WARMUP,
            output_path=output_path,
            resources=resources,
            env_vars=_env_vars,
            wandb_name=name,
            tags=("protein", "contacts-v1", "qwen3", "3b", "unmasked", "coreweave",
                  f"e{EPOCHS}", f"bs{TRAIN_BATCH}", f"lr{_lr_tag(lr)}"),
            wait=False,  # submit only; we block on all of them below
        )
        print(f"  submitted: {name} -> {output_path}")
        jobs.append((name, job))
    print(f"[exp108] submitted {len(jobs)} gang(s) at iris batch priority; awaiting completion.")

    # CRITICAL: the training gangs are CHILDREN of this driver job. If the driver
    # exits first, iris finalizes (kills) them (that's what happened with the
    # first wait=False launch). So the driver must stay alive until every gang
    # finishes. Submitting all first, then waiting, lets the N sweep gangs run
    # concurrently while the driver blocks. raise_on_failure per-job so one sweep
    # member failing doesn't tear down the others.
    failures = 0
    for name, job in jobs:
        try:
            job.wait(raise_on_failure=True)
            print(f"[exp108] {name}: SUCCEEDED")
        except Exception as e:  # noqa: BLE001 — report, keep waiting on the rest
            failures += 1
            print(f"[exp108] {name}: FAILED — {e}")
    if failures:
        raise SystemExit(f"[exp108] {failures}/{len(jobs)} run(s) failed")


if __name__ == "__main__":
    main()

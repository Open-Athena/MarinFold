# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Warm-restart the #67 contacts-v1 1.5B run for ~1 more epoch (issue #85).

Reloads the final #67 checkpoint (``step-11999`` of
``protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2``), re-heats the
learning rate, and trains ~1 more epoch on a freshly-shuffled pass of the same
contacts-v1 corpus. This is a **warm restart**, not a resume:
``initialize_from_checkpoint_path`` loads model weights only → fresh step 0,
fresh optimizer state, fresh cosine LR schedule (so ``learning_rate`` + warmup
define the re-heat), and a fresh shuffled data loader.

Everything else matches the #67 recipe (unmasked next-token loss, Feistel
shuffle, full held-out-val eval, us-east5-a, seq 8192). The token caches are
reused from #67 via the shared ``MARIN_PREFIX`` (see ``contacts_v1_train_common``),
so no re-tokenization happens.

Knobs that differ from #67:
* ``initialize_from_checkpoint_path`` — warm-start from #67 ``step-11999``.
* **slice** — **v5p-32** (vs #67's v5p-8). The v5p-8 preemptible pool was
  thrashing (10 preemptions in 11 min, never reached step 0); v5p-32/64 have
  capacity. 4× the chips.
* ``train_batch_size`` — **512** (4×, matching the 4× chips ⇒ per-chip batch
  stays 32, identical to #67 — same memory footprint, no OOM).
* ``learning_rate`` — re-heat peak **4.0e-4** = 2× the batch-128 value of
  2.0e-4 (√ of the 4× batch increase, the standard LR-vs-batch scaling).
* ``num_train_steps`` — **1125** (~1 epoch at batch 512; ¼ of the 4,490 steps
  it takes at batch 128).
* ``data_seed`` — **1** (a different Feistel permutation than #67's seed 0, so the
  extra epoch isn't the identical token order).

Usage::

    WANDB_API_KEY=<key> uv run iris --cluster marin job run --no-wait \\
        --enable-extra-resources --memory=16GB --disk=16GB --cpu=1 \\
        --extra=tpu --zone=us-east5-a \\
        -- python -m train_protein_1_5b_contacts_v1_reheat
"""

from fray import ResourceConfig
from levanter.models.llama import LlamaConfig
from marin.execution import executor_main

from contacts_v1_train_common import build_train_step

# --- TEMP DIAGNOSTIC (issue #85): which marin-levanter does the pod resolve? ---
# Runs at import on the DRIVER pod, whose env is built by the same
# `uv sync --all-packages --frozen` as the training worker. #6014 removed
# `_exemplar_for`; its presence ⇒ the OLD buggy cache reader (input_ids/0).
try:
    import importlib.metadata as _md
    import inspect as _ins

    import levanter.store.cache as _lc

    _buggy = "_exemplar_for" in _ins.getsource(_lc)
    print(
        f"DIAG85 marin-levanter={_md.version('marin-levanter')} "
        f"marin-core={_md.version('marin-core')} cache={_lc.__file__} "
        f"has_exemplar_for={_buggy} (True=OLD/buggy reader)",
        flush=True,
    )
except Exception as _e:  # diagnostics must never break the run
    print(f"DIAG85 failed: {_e!r}", flush=True)

# 1.5B shape — identical to the #67 run (matches Pythia-1.4B: h=2048, l=24,
# dff=8192, heads=32). MUST match the checkpoint we warm-start from.
protein_llama_1_5b = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
)

# The #67 run's final levanter checkpoint (full training state — model weights are
# read for the warm start). Written under the W&B-run-name dir, NOT the executor
# hash dir; see exp67's export_protein_1_5b_contacts_v1.py for why. Available
# steps: 2000/4000/6000/8000/10000/11999; 11999 is the final (= step 12000,
# 0-indexed).
INIT_CHECKPOINT = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/checkpoints/"
    "protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2/checkpoints/step-11999"
)

# Bigger, less-contended slice. The v5p-8 preemptible pool was thrashing
# (10 preemptions in 11 min, never reached step 0); v5p-32/64 have capacity now.
# v5p-32 = 4× the chips of #67's v5p-8, so we scale the GLOBAL batch 4× (128→512)
# — per-chip batch stays 32 (identical to #67 ⇒ same memory, no OOM) — and scale
# the re-heat peak LR by √4 = 2× (2.0e-4 → 4.0e-4, per the standard sqrt rule).
RESOURCES_V5P32 = ResourceConfig.with_tpu(
    "v5p-32", slice_count=1, cpu=32, ram="128g", disk="50g", zone="us-east5-a",
)
# Re-heat peak LR: 2× the batch-128 value of 2.0e-4 (sqrt of the 4× batch increase).
REHEAT_PEAK_LR = 4.0e-4
TRAIN_BATCH = 512
# ~1 epoch. At 512 × 8192 = 4.19M tok/step over ~4.7B train tokens ⇒ ~1,123 steps
# (¼ of the 4,490 steps it would take at batch 128).
CONTINUE_STEPS = 1_125

RUN_NAME = "protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512"

protein_model_1_5b_contacts_v1_reheat = build_train_step(
    name=RUN_NAME,
    model_config=protein_llama_1_5b,
    learning_rate=REHEAT_PEAK_LR,
    num_train_steps=CONTINUE_STEPS,
    train_batch_size=TRAIN_BATCH,
    # data_seed=2 (was 1): bumped to bust the executor step hash so marin can't
    # reuse any cached state/env from the earlier frozen-wheel 726d1794 attempts.
    # Still a fresh Feistel permutation distinct from #67's seed 0.
    data_seed=2,
    extra_tags=("1_5b", "continue", "reheat", "v5p32", "bs512"),
    wandb_name=RUN_NAME,
    resources=RESOURCES_V5P32,
    # Short run (1,125 steps) on a preemptible pool — keep frequent permanent
    # checkpoints so a late preemption can't lose much (vs the 2000 default).
    steps_per_export=250,
    initialize_from_checkpoint_path=INIT_CHECKPOINT,
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_1_5b_contacts_v1_reheat])

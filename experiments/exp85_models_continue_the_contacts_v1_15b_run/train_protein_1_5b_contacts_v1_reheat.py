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
shuffle, full held-out-val eval, v5p-8 @ us-east5-a, batch 128, seq 8192). The
token caches are reused from #67 via the shared ``MARIN_PREFIX`` (see
``contacts_v1_train_common``), so no re-tokenization happens.

Knobs that differ from #67:
* ``initialize_from_checkpoint_path`` — warm-start from #67 ``step-11999``.
* ``learning_rate`` — re-heat peak **2.0e-4** (≈0.57× the original 3.5e-4): a
  moderate restart that perturbs the converged solution without blowing it up in
  a single epoch. Bump to ``3.5e-4`` for a full-strength re-heat.
* ``num_train_steps`` — **4500** (~1 epoch; 1 epoch ≈ 4,490 steps).
* ``data_seed`` — **1** (a different Feistel permutation than #67's seed 0, so the
  extra epoch isn't the identical token order).

Usage::

    WANDB_API_KEY=<key> uv run iris --cluster marin job run --no-wait \\
        --enable-extra-resources --memory=16GB --disk=16GB --cpu=1 \\
        --extra=tpu --zone=us-east5-a \\
        -- python -m train_protein_1_5b_contacts_v1_reheat
"""

from levanter.models.llama import LlamaConfig
from marin.execution import executor_main

from contacts_v1_train_common import build_train_step

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

# Re-heat peak LR (key knob). 2.0e-4 ≈ 0.57× the #67 peak of 3.5e-4.
REHEAT_PEAK_LR = 2.0e-4
# ~1 epoch (1 epoch ≈ 4,490 steps at 128 × 8192 = 1.05M tok/step over ~4.7B tok).
CONTINUE_STEPS = 4_500

RUN_NAME = "protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3"

protein_model_1_5b_contacts_v1_reheat = build_train_step(
    name=RUN_NAME,
    model_config=protein_llama_1_5b,
    learning_rate=REHEAT_PEAK_LR,
    num_train_steps=CONTINUE_STEPS,
    data_seed=1,
    extra_tags=("1_5b", "continue", "reheat"),
    wandb_name=RUN_NAME,
    initialize_from_checkpoint_path=INIT_CHECKPOINT,
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_1_5b_contacts_v1_reheat])

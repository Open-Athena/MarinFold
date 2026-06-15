# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a ~1.47B-param Llama on the contacts-v1 corpus — the quick #67 run.

The "quick / simple" 1.5B run requested in issue #67: get a fast sense of what
contact-prediction accuracy is reachable on the new ``contacts-v1`` corpus,
while @eric-czech does the carefully-tuned variant (#61). Reuses the 1.5B shape
and TPU/LR recipe from ``marin/protein-training-1b`` (mirrored in
``exp0_models_protein_docs_initial_port/train_protein_1_5b_distance_masked.py``)
but with two deliberate changes for contacts-v1:

* **Unmasked next-token loss** over the whole document — contacts-v1 has no
  ``<distance>`` statements, so there is no distance-bin loss mask to apply.
* **Shuffled** training (full Feistel permutation) because the corpus shards are
  round-descending (highest-pLDDT last). Full held-out val split each eval.

Length: 1 epoch ≈ 4,490 steps (train ≈ 4.7B tok / (128 × 8192 = 1.05M tok/step));
``num_train_steps=12_000`` is ~2.7 epochs.

All executor output (token caches + checkpoints) is pinned under
``gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b`` via
``MARIN_PREFIX`` (force-set in ``contacts_v1_train_common``).

Usage::

    uv run iris --cluster marin job run --no-wait --enable-extra-resources \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a \\
        -- python -m train_protein_1_5b_contacts_v1
"""

from levanter.models.llama import LlamaConfig
from marin.execution import executor_main

from contacts_v1_train_common import build_train_step

# 1.5B shape — matches Pythia-1.4B (h=2048, l=24, dff=8192, heads=32), identical
# to exp0's ``protein_llama_1_5b``.
protein_llama_1_5b = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
)

RUN_NAME = "protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked"

protein_model_1_5b_contacts_v1 = build_train_step(
    name=RUN_NAME,
    model_config=protein_llama_1_5b,
    learning_rate=3.5e-4,
    num_train_steps=12_000,
    extra_tags=("1_5b",),
    wandb_name=RUN_NAME,
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_1_5b_contacts_v1])

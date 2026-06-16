# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the exp85 warm-restart 1.5B run to HuggingFace format (issue #85).

Targets the run produced by ``train_protein_1_5b_contacts_v1_reheat.py``.
``CHECKPOINT_STEP`` selects the ``checkpoints/step-{N}`` input and labels the
output dir (``hf/...-step-{N}``); the contacts-v1 tokenizer is co-located with
the exported weights (hard rule: tokenizer travels with the model).

Like exp67's export, ``CHECKPOINT_PATH`` is set to the run's actual checkpoint
directory (the **W&B-run-name** dir — the ``-<runid>`` suffix is generated at
runtime by ``wandb.init``), NOT the executor-hash output dir that
``output_path_of`` resolves to. FILL IN the real ``-<runid>`` suffix and the
final step number after the run launches (read them from W&B / the GCS
``checkpoints/`` listing).

Usage (after the target ``step-{CHECKPOINT_STEP}`` checkpoint exists on GCS)::

    uv run iris --cluster marin job run --no-wait --enable-extra-resources \\
        --memory=32GB --disk=16GB --cpu=4 \\
        -- python -m export_protein_1_5b_contacts_v1_reheat
"""

from marin.execution import executor_main

from contacts_v1_train_common import build_hf_export_step
from train_protein_1_5b_contacts_v1_reheat import (
    protein_llama_1_5b,
    protein_model_1_5b_contacts_v1_reheat,
)

# Final step of the ~1,125-step continuation (= num_train_steps - 1, like #67's
# 11999). Confirm against the GCS checkpoints/ listing once the run finishes.
CHECKPOINT_STEP = 1_124
# The run's real checkpoint dir (W&B-run-name dir). Reuses exp67's MARIN_PREFIX
# (shared cache); the run is namespaced by its own W&B run name. Replace
# ``REPLACE_RUNID`` with the actual wandb run-id suffix once the run starts.
_RUN_CHECKPOINTS_DIR = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/checkpoints/"
    "protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512-REPLACE_RUNID/checkpoints"
)
CHECKPOINT_PATH: str | None = f"{_RUN_CHECKPOINTS_DIR}/step-{CHECKPOINT_STEP}"

hf_export = build_hf_export_step(
    train_step=protein_model_1_5b_contacts_v1_reheat,
    model_config=protein_llama_1_5b,
    checkpoint_step=CHECKPOINT_STEP,
    name_prefix="protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512",
    checkpoint_path_override=CHECKPOINT_PATH,
)


if __name__ == "__main__":
    executor_main(steps=[hf_export])

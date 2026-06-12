# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the 1.5B contacts-v1 training run to HuggingFace format.

Targets the run produced by ``train_protein_1_5b_contacts_v1.py``.
``CHECKPOINT_STEP`` selects the exact ``checkpoints/step-{N}`` input and labels
the output directory (``hf/...-step-{N}``). The contacts-v1 tokenizer is
co-located with the exported weights (hard rule: tokenizer travels with the
model).

``CHECKPOINT_PATH`` is set to the run's actual checkpoint directory rather than
left to the dependency-derived default, because levanter writes checkpoints under
the **W&B-run-name** dir (the ``-3b5cf2`` suffix is the W&B run id, generated at
runtime by ``wandb.init``), NOT the executor-hash output dir that
``output_path_of(train_step, ...)`` resolves to. The dependency path therefore
can never find these weights (verified: ``…-23f72e/`` has only ``.executor_info``;
the checkpoints live under ``…-3b5cf2/checkpoints/step-*``). This mirrors exp0's
export scripts, which also hard-code the run-name path. NOTE: this path is
specific to *this* launched run — a relaunch gets a new ``-<runid>`` suffix.

Usage (run after the target ``step-{CHECKPOINT_STEP}`` checkpoint exists on GCS)::

    uv run iris --cluster marin job run --no-wait --enable-extra-resources \\
        --memory=32GB --disk=16GB --cpu=4 \\
        -- python -m export_protein_1_5b_contacts_v1
"""

from marin.execution import executor_main

from contacts_v1_train_common import build_hf_export_step
from train_protein_1_5b_contacts_v1 import (
    protein_llama_1_5b,
    protein_model_1_5b_contacts_v1,
)

# Which permanent (steps_per_export=2000) checkpoint to export; selects the exact
# input ``step-{N}`` and labels the output dir. Must already exist on GCS
# (currently step-2000, step-4000, …; the final is step-12000).
CHECKPOINT_STEP = 12_000
# The run's real checkpoint directory (W&B-run-name dir — see module docstring).
# CHECKPOINT_PATH is derived from CHECKPOINT_STEP so the two stay in sync; with
# the override set, ``build_hf_export_step`` reads this exact step dir
# (discover_latest=False) instead of the unreachable dependency-derived path.
_RUN_CHECKPOINTS_DIR = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/checkpoints/"
    "protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2/checkpoints"
)
CHECKPOINT_PATH: str | None = f"{_RUN_CHECKPOINTS_DIR}/step-{CHECKPOINT_STEP}"

hf_export = build_hf_export_step(
    train_step=protein_model_1_5b_contacts_v1,
    model_config=protein_llama_1_5b,
    checkpoint_step=CHECKPOINT_STEP,
    name_prefix="protein-contacts-1_5b-contacts-v1-unmasked",
    checkpoint_path_override=CHECKPOINT_PATH,
)


if __name__ == "__main__":
    executor_main(steps=[hf_export])

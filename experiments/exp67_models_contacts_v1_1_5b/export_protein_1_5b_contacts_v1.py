# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the 1.5B contacts-v1 training run to HuggingFace format.

Targets the run produced by ``train_protein_1_5b_contacts_v1.py``.
``CHECKPOINT_STEP`` selects the exact ``checkpoints/step-{N}`` input and labels
the output directory (``hf/...-step-{N}``). The contacts-v1 tokenizer is
co-located with the exported weights (hard rule: tokenizer travels with the
model).

By default this depends on the training step reaching SUCCEEDED. To snapshot a
checkpoint mid-run, set ``CHECKPOINT_PATH`` to the literal
``gs://.../checkpoints`` dir (printed in the run's logs / discoverable under
``gs://marin-us-east5/checkpoints/<run>-<hash>/checkpoints``) and pass it as
``checkpoint_path_override``.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=32GB --disk=16GB --cpu=4 \\
        -- python -m export_protein_1_5b_contacts_v1
"""

from marin.execution import executor_main

from contacts_v1_train_common import build_hf_export_step
from train_protein_1_5b_contacts_v1 import (
    protein_llama_1_5b,
    protein_model_1_5b_contacts_v1,
)

# Selects the exact input checkpoint and labels the output directory.
CHECKPOINT_STEP = 12_000
# Set to a literal gs://.../checkpoints path to snapshot mid-run without waiting
# for the training step to finish; leave None to depend on the train step.
CHECKPOINT_PATH: str | None = None

hf_export = build_hf_export_step(
    train_step=protein_model_1_5b_contacts_v1,
    model_config=protein_llama_1_5b,
    checkpoint_step=CHECKPOINT_STEP,
    name_prefix="protein-contacts-1_5b-contacts-v1-unmasked",
    checkpoint_path_override=CHECKPOINT_PATH,
)


if __name__ == "__main__":
    executor_main(steps=[hf_export])

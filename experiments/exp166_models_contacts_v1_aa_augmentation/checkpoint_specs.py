# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint and artifact locations for the exp166 evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointSpec:
    """One immutable Hugging Face checkpoint selection."""

    key: str
    repo_id: str
    subfolder: str
    output_name: str


CHECKPOINTS = {
    spec.key: spec
    for spec in (
        CheckpointSpec(
            key="exp166",
            repo_id="open-athena/marinfold-exp166",
            subfolder=(
                "prot-exp166-cv1-aaaug-1_5b-e8-lr3p162e-3-wd0p1-bs128-"
                "exp117-init-us-east1/hf/step-35679"
            ),
            output_name="exp166-aaaug-step-35679",
        ),
        CheckpointSpec(
            key="exp117-control",
            repo_id="open-athena/marinfold-exp117",
            subfolder=(
                "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-"
                "europe-west4/hf/step-35679"
            ),
            output_name="exp117-control-step-35679",
        ),
    )
}

HF_BUCKET_ROOT = "buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp166"
TARGETS_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "data/contacts-v1-model-eval-exp169/eval_targets.parquet"
)
GROUND_TRUTH_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "data/contacts-v1-model-eval-exp169/gt_universe.jsonl"
)
GROUND_TRUTH_SHA256 = "3ff6eb4e383582595ad6f9811c77e2839ebcc0030a050b9c1f15d020163331c9"
MARINFOLD_REVISION = "0dcb7f56b1ea03ebd38e2337d69c1fff5203b426"

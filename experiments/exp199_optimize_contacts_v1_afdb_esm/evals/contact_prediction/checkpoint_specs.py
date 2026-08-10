# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable checkpoints selected for the exp199 contact evaluations."""

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CheckpointSpec:
    """One checkpoint, its evaluation placement, and optional reference score."""

    key: str
    run_name: str
    step: int
    region: str | None
    checkpoint_uri: str | None = None
    tokenizer_repo: str | None = None
    hf_repo_id: str | None = None
    hf_subfolder: str | None = None
    hf_revision: str | None = None
    reference_r_all: float | None = None
    reference_tolerance: float | None = None

    def __post_init__(self) -> None:
        native = self.checkpoint_uri is not None
        hf_values = (self.hf_repo_id, self.hf_subfolder, self.hf_revision)
        hf = all(value is not None for value in hf_values)
        if native == hf:
            raise ValueError("checkpoint must have exactly one complete source")
        if any(value is not None for value in hf_values) and not hf:
            raise ValueError(
                "Hugging Face sources require repo, subfolder, and revision"
            )
        if native and self.tokenizer_repo is None:
            raise ValueError("native Levanter checkpoints require a tokenizer repo")
        if (self.reference_r_all is None) != (self.reference_tolerance is None):
            raise ValueError("reference R-precision and tolerance must be set together")


def validate_run_tag(run_tag: str | None) -> str | None:
    """Validate an optional path and job-name component for an isolated run."""

    if run_tag is not None and re.fullmatch(r"[a-z0-9][a-z0-9-]*", run_tag) is None:
        raise ValueError(f"invalid run tag: {run_tag!r}")
    return run_tag


def checkpoint_manifest(spec: CheckpointSpec) -> dict[str, Any]:
    """Serialize a spec without changing manifests for existing native runs."""

    return {key: value for key, value in asdict(spec).items() if value is not None}


EXP199_GCS_ROOT = (
    "gs://marin-us-east1/protein-structure/MarinFold/"
    "exp199_optimize_contacts_v1_afdb_esm/checkpoints/protein"
)
EXP199_VERSION = "2026.08.07.1"
EXP199_TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
EXP199_FINAL_STEP = 72599
EXP199_HF_REPO_ID = "open-athena/marinfold-exp199"
EXP199_HF_REVISION = "ed7103bfd7dac3f75ba759e5ec827da3d75ff0ed"


def exp199_checkpoint(
    *, key: str, run_name: str, checkpoint_run_name: str, step: int
) -> CheckpointSpec:
    """Build one us-east1 exp199 native-checkpoint selection."""

    return CheckpointSpec(
        key=key,
        run_name=run_name,
        step=step,
        checkpoint_uri=(
            f"{EXP199_GCS_ROOT}/{checkpoint_run_name}/{EXP199_VERSION}/"
            f"checkpoints/step-{step}"
        ),
        region="us-east1",
        tokenizer_repo=EXP199_TOKENIZER,
    )


def exp199_final_checkpoint(trial: str) -> CheckpointSpec:
    """Build the final-checkpoint selection for one finished exp199 trial."""

    checkpoint_run_name = f"prot-exp199-cv1-{trial}"
    return exp199_checkpoint(
        key=f"{trial}-step{EXP199_FINAL_STEP}",
        run_name=f"{checkpoint_run_name}-us-east1",
        checkpoint_run_name=checkpoint_run_name,
        step=EXP199_FINAL_STEP,
    )


def exp199_hf_checkpoint(*, key: str, run_name: str, step: int) -> CheckpointSpec:
    """Build one revision-pinned exp199 HF-checkpoint selection."""

    return CheckpointSpec(
        key=key,
        run_name=run_name,
        step=step,
        region=None,
        hf_repo_id=EXP199_HF_REPO_ID,
        hf_subfolder=f"{run_name}/hf/step-{step}",
        hf_revision=EXP199_HF_REVISION,
    )


P03_RUN_NAME = "prot-exp199-cv1-s01-m1-p03-aug-us-east1"
P03_CHECKPOINT_RUN_NAME = "prot-exp199-cv1-s01-m1-p03-aug"
P03_PERMANENT_STEPS = (
    8920,
    17840,
    26760,
    35680,
    44600,
    53520,
    62440,
    71360,
    EXP199_FINAL_STEP,
)


CHECKPOINTS = {
    spec.key: spec
    for spec in (
        exp199_checkpoint(
            key="s01-m1-p06-base-step26760",
            run_name="prot-exp199-cv1-s01-m1-p06-base",
            checkpoint_run_name="prot-exp199-cv1-s01-m1-p06-base",
            step=26760,
        ),
        *(
            exp199_checkpoint(
                key=f"s01-m1-p03-aug-step{step}",
                run_name=P03_RUN_NAME,
                checkpoint_run_name=P03_CHECKPOINT_RUN_NAME,
                step=step,
            )
            for step in P03_PERMANENT_STEPS
        ),
        exp199_final_checkpoint("s01-m1-p06-aug"),
        exp199_hf_checkpoint(
            key="s01-m1-p03-base-step72599",
            run_name="prot-exp199-cv1-s01-m1-p03-base-us-east5",
            step=72599,
        ),
        exp199_hf_checkpoint(
            key="cw-s02-m1-p06-aug-step145199",
            run_name="prot-exp199-cw-cv1-s02-m1-p06-aug",
            step=145199,
        ),
        CheckpointSpec(
            key="exp117-control-step35679",
            run_name=(
                "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4"
            ),
            step=35679,
            region=None,
            hf_repo_id="open-athena/marinfold-exp117",
            hf_subfolder=(
                "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-"
                "europe-west4/hf/step-35679"
            ),
            hf_revision="f07366720aee0f62d7629ad3bd91dbcacc80ddef",
            reference_r_all=0.5335961341539802,
            reference_tolerance=0.006,
        ),
    )
}

HF_BUCKET_ROOT = "buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp199"
TARGETS_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "data/contacts-v1-model-eval-exp169/eval_targets.parquet"
)
GROUND_TRUTH_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "data/contacts-v1-model-eval-exp169/gt_universe.jsonl"
)
GROUND_TRUTH_SHA256 = "3ff6eb4e383582595ad6f9811c77e2839ebcc0030a050b9c1f15d020163331c9"
MARINFOLD_REVISION = "d79c99f17a7c9abd0d3717ec35cee90bf6649752"

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable checkpoint catalog for the exp117 and exp146 trajectories."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelGeometry:
    """Levanter Qwen3 dimensions needed to restore and export a checkpoint."""

    hidden_dim: int
    intermediate_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int | None = None


@dataclass(frozen=True)
class RunSpec:
    """One training run and the permanent checkpoints selected from it."""

    key: str
    run_name: str
    model_label: str
    region: str
    checkpoint_root: str
    selected_steps: tuple[int, ...]
    validation_losses: tuple[float, ...]
    epochs: tuple[int, ...]
    tokens_per_step: int
    geometry: ModelGeometry


@dataclass(frozen=True)
class CheckpointSpec:
    """One permanent training checkpoint selected for production evaluation."""

    key: str
    run_key: str
    run_name: str
    model_label: str
    region: str
    step: int
    epoch: int
    training_tokens: int
    validation_loss: float
    checkpoint_uri: str
    geometry: ModelGeometry
    tokenizer_repo: str = "eczech/contacts-v1-tokenizer-5d68a24a899f"


RUNS = {
    spec.key: spec
    for spec in (
        RunSpec(
            key="exp146-3b-e8",
            run_name=("prot-exp146-cv1-s01-3b-e8-lr3p162e-3-wd0p4-bs256-us-east1"),
            model_label="3B E8",
            region="us-east1",
            checkpoint_root=(
                "gs://marin-us-east1/checkpoints/protein/"
                "prot-exp146-cv1-s01-3b-e8-lr3p162e-3-wd0p4-bs256-us-east1/"
                "2026.07.23.01/checkpoints"
            ),
            selected_steps=(2230, 4460, 6690, 8920, 11150, 13380, 15610, 17839),
            validation_losses=(
                3.602867841720581,
                3.184345006942749,
                3.026846408843994,
                2.8735108375549316,
                2.807288885116577,
                2.746565818786621,
                2.714020013809204,
                2.7024784088134766,
            ),
            epochs=(1, 2, 3, 4, 5, 6, 7, 8),
            tokens_per_step=256 * 8192,
            geometry=ModelGeometry(
                hidden_dim=2560,
                intermediate_dim=10240,
                num_layers=30,
                num_heads=48,
                num_kv_heads=16,
                head_dim=64,
            ),
        ),
        RunSpec(
            key="exp117-1_5b-e16",
            run_name=(
                "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4"
            ),
            model_label="1.5B E16",
            region="europe-west4",
            checkpoint_root=(
                "gs://marin-eu-west4/checkpoints/protein/"
                "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-"
                "europe-west4/2026.07.13.02/checkpoints"
            ),
            # Every second permanent checkpoint, including the final checkpoint.
            selected_steps=(4460, 8920, 13380, 17840, 22300, 26760, 31220, 35679),
            validation_losses=(
                3.031233072280884,
                2.968775510787964,
                2.8245866298675537,
                2.768697500228882,
                2.7385761737823486,
                2.705306053161621,
                2.696971893310547,
                2.7037086486816406,
            ),
            epochs=(2, 4, 6, 8, 10, 12, 14, 16),
            tokens_per_step=256 * 8192,
            geometry=ModelGeometry(
                hidden_dim=2048,
                intermediate_dim=8192,
                num_layers=24,
                num_heads=32,
                num_kv_heads=8,
            ),
        ),
        RunSpec(
            key="exp117-1_5b-e8-bs64",
            run_name=("prot-exp117-cv1-s02-1_5b-e8-lr3p162e-3-wd0p2-bs64-europe-west4"),
            model_label="1.5B E8, BS64",
            region="europe-west4",
            checkpoint_root=(
                "gs://marin-eu-west4/checkpoints/protein/"
                "prot-exp117-cv1-s02-1_5b-e8-lr3p162e-3-wd0p2-bs64-"
                "europe-west4/2026.07.13.02/checkpoints"
            ),
            selected_steps=(8920, 17840, 26760, 35680, 44600, 53520, 62440, 71359),
            validation_losses=(
                3.224363088607788,
                2.997375249862671,
                2.9025375843048096,
                2.844377279281616,
                2.801011323928833,
                2.768310785293579,
                2.724397659301758,
                2.7130589485168457,
            ),
            epochs=(1, 2, 3, 4, 5, 6, 7, 8),
            tokens_per_step=64 * 8192,
            geometry=ModelGeometry(
                hidden_dim=2048,
                intermediate_dim=8192,
                num_layers=24,
                num_heads=32,
                num_kv_heads=8,
            ),
        ),
    )
}


def checkpoint_specs() -> tuple[CheckpointSpec, ...]:
    """Expand the runs into the ordered checkpoint evaluation catalog."""

    checkpoints = []
    for run in RUNS.values():
        if not (
            len(run.selected_steps) == len(run.validation_losses) == len(run.epochs)
        ):
            raise ValueError(f"mismatched checkpoint metadata for {run.key}")
        for step, loss, epoch in zip(
            run.selected_steps, run.validation_losses, run.epochs, strict=True
        ):
            checkpoints.append(
                CheckpointSpec(
                    key=f"{run.key}-step{step}",
                    run_key=run.key,
                    run_name=run.run_name,
                    model_label=run.model_label,
                    region=run.region,
                    step=step,
                    epoch=epoch,
                    training_tokens=step * run.tokens_per_step,
                    validation_loss=loss,
                    checkpoint_uri=f"{run.checkpoint_root}/step-{step}",
                    geometry=run.geometry,
                )
            )
    return tuple(checkpoints)


CHECKPOINTS = {spec.key: spec for spec in checkpoint_specs()}

HF_BUCKET_ROOT = (
    "buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp169/trajectory"
)
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

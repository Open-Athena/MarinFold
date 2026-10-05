"""Continue exp277 from its last pre-cooldown checkpoint for one new epoch."""

import os
from dataclasses import replace
from datetime import timedelta

import click
from fray.types import ResourceConfig
from levanter.data.text.datasets import ConcatDatasetComponent
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    DECAY,
    GLOBAL_BATCH_SIZE,
    LR_SCHEDULE,
    MIN_LR_RATIO,
    MODEL_CONFIG,
    MODEL_SEED,
    SEQ_LEN,
    SHUFFLE,
    STEPS_PER_EVAL,
    WANDB_WATCH,
    WARMUP,
    existing_cache,
)
from experiments.exp277_models_single_mpnn_pilot.config import (
    CORPORA,
    EPOCH_PACKED_EXAMPLES,
    EPOCH_TRAIN_STEPS,
    PREFIX,
    RUN_ID,
    VALIDATION_CACHE,
    VERSION,
)
from experiments.exp277_models_single_mpnn_pilot.epoch_data import (
    FULL_CORPUS,
    continuation_epoch_data,
)
from experiments.exp277_models_single_mpnn_pilot.prepare import verify_cache
from experiments.exp277_models_single_mpnn_pilot.runtime import run_train_job

SOURCE_CHECKPOINT_STEP = 213_072
SOURCE_RESUME_STEP = SOURCE_CHECKPOINT_STEP + 1
CONTINUATION_DATA_SEED = 1
CONTINUATION_RUN_ID = "contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B"


def source_checkpoint() -> ArtifactStep[LevanterCheckpoint]:
    """Adopt the completed first run as the immutable continuation source."""
    return ArtifactStep[LevanterCheckpoint].adopt(
        "input/full-epoch2/source-from213072",
        VERSION,
        source=f"{PREFIX}/runs/{RUN_ID}",
        kind=LevanterCheckpoint,
        config={
            "source_run": RUN_ID,
            "checkpoint_step": SOURCE_CHECKPOINT_STEP,
        },
    )


def build_continuation_run(
    *, smoke: bool, nodes: int, attempt: int
) -> ArtifactStep[LevanterCheckpoint]:
    """Restore full state, reshuffle the corpus, and add one WSD epoch.

    The smoke's identity includes its attempt number. A step whose output path
    already holds a `SUCCESS` status is skipped outright, and recipe drift only
    warns, so a repeated smoke at a stable path would be served from cache
    rather than rerun. Production keeps its reserved identity so that a
    restarted production job resumes from its own checkpoints.
    """
    if nodes not in (1, 2, 4, 8, 16):
        raise ValueError(f"Unsupported H100 gang size: {nodes}")
    per_device = min(8, GLOBAL_BATCH_SIZE // (8 * nodes))
    run_id = (
        f"{CONTINUATION_RUN_ID}-smoke-a{attempt:02d}" if smoke else CONTINUATION_RUN_ID
    )
    additional_steps = 10 if smoke else EPOCH_TRAIN_STEPS
    end_step = SOURCE_RESUME_STEP + additional_steps
    env = {
        "MARIN_PREFIX": PREFIX,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
    }
    datasets = {
        existing_cache(
            name=(
                f"input/full-epoch2-smoke/{corpus.name}"
                if smoke
                else f"input/full-epoch2/{corpus.name}"
            ),
            version=VERSION,
            source=corpus.cache,
            tags=["contacts-v1", "decontaminated", corpus.name],
        ): 1.0
        for corpus in CORPORA
    }
    validation = existing_cache(
        name="input/validation",
        version="2026.07.25",
        source=VALIDATION_CACHE,
        tags=["contacts-v1", "validation"],
    )
    step = train_lm(
        name=f"runs/{run_id}",
        run_id=run_id,
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=1e-3,
            weight_decay=0.2,
            warmup=WARMUP,
            rewarmup=0.0,
            decay=DECAY,
            cycle_length=[SOURCE_RESUME_STEP, additional_steps],
            min_lr_ratio=MIN_LR_RATIO,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets=datasets,
        validation=[validation],
        init_from=source_checkpoint(),
        batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=end_step,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_gpu(
            "H100", count=8, replicas=nodes, cpu=32, ram="256g", disk="256g"
        ),
        tensor_parallel_size=1,
        steps_per_eval=additional_steps if smoke else STEPS_PER_EVAL,
        wandb_project="MarinFold",
        wandb_group="exp277-single-mpnn-pilot",
        tags=[
            "exp277",
            "contacts-v1",
            "decontaminated",
            "mpnn",
            "m2",
            "p06",
            "continuation",
            "full-corpus-one-more-epoch",
            f"source-step={SOURCE_CHECKPOINT_STEP}",
            f"data-seed={CONTINUATION_DATA_SEED}",
            "smoke" if smoke else "production",
            f"nodes={nodes}",
        ],
        env_vars=env,
    )
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        source_checkpoint_dir = pod.train_config.initialize_from_checkpoint_path
        if not ctx.is_fingerprint and source_checkpoint_dir is None:
            raise ValueError("continuation requires the source checkpoint dependency")
        exact_checkpoint = (
            f"{source_checkpoint_dir}/step-{SOURCE_CHECKPOINT_STEP}"
            if source_checkpoint_dir is not None
            else None
        )
        trainer = replace(
            pod.train_config.trainer,
            seed=MODEL_SEED,
            initialize_from=exact_checkpoint,
            allow_partial_checkpoint=False,
            max_eval_batches=2 if smoke else None,
            watch=WANDB_WATCH,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                save_interval=timedelta(minutes=15),
                keep=[] if smoke else [{"every": EPOCH_TRAIN_STEPS // 10}],
            ),
        )
        if not ctx.is_fingerprint:
            trainer = replace(
                trainer,
                per_device_parallelism=per_device,
                per_device_eval_parallelism=per_device,
            )
        components = pod.train_config.data.components
        children = {
            dataset.name: replace(components[dataset.name], pack=True)
            for dataset in datasets
        }
        data = replace(
            pod.train_config.data,
            auto_build_caches=False,
            shuffle=SHUFFLE,
            components={
                FULL_CORPUS: ConcatDatasetComponent(children=children),
                validation.name: replace(components[validation.name], pack=True),
            },
            train_weights={FULL_CORPUS: 1.0, validation.name: 0.0},
            block_cross_document_attention=True,
        )
        config = replace(
            pod.train_config,
            trainer=trainer,
            data=continuation_epoch_data(
                data,
                start_step=SOURCE_RESUME_STEP,
                source_augmentation_step=SOURCE_RESUME_STEP,
                augmentation_schedule_steps=EPOCH_TRAIN_STEPS,
                expected_packed_examples=EPOCH_PACKED_EXAMPLES,
            ),
            data_seed=CONTINUATION_DATA_SEED,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            hf_save_steps=end_step + 1,
        )
        return replace(pod, train_config=config)

    return replace(
        step,
        build_config=build_config,
        run=run_train_job,
        override_path=f"{PREFIX}/runs/{run_id}",
    )


@click.command()
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    smoke = os.environ.get("SMOKE") == "1"
    for corpus in CORPORA:
        if not verify_cache(corpus):
            raise ValueError(f"Incomplete cache: {corpus.cache}")
    return build_continuation_run(
        smoke=smoke,
        nodes=int(os.environ["NODES"]),
        attempt=int(os.environ["ATTEMPT"]),
    )


if __name__ == "__main__":
    main()

"""Train exactly one scratch-initialized native/MPNN mixture configuration."""

import os
from dataclasses import replace
from datetime import timedelta

import click
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint

from experiments.exp277_models_single_mpnn_pilot.runtime import run_train_job
from experiments.exp232_sweep_cv1_decontam.training_contract import (
    DATA_SEED,
    DECAY,
    GLOBAL_BATCH_SIZE,
    LR_SCHEDULE,
    MIN_LR_RATIO,
    MODEL_CONFIG,
    MODEL_SEED,
    NUM_TRAIN_STEPS,
    PERMANENT_CHECKPOINT_EVERY,
    SEQ_LEN,
    SHUFFLE,
    STEPS_PER_EVAL,
    WANDB_WATCH,
    WARMUP,
    augment_amino_acids,
    existing_cache,
)
from experiments.exp277_models_single_mpnn_pilot.config import (
    CORPORA,
    PREFIX,
    RUN_ID,
    VALIDATION_CACHE,
    VERSION,
)
from experiments.exp277_models_single_mpnn_pilot.prepare import verify_cache


def build_run(*, smoke: bool, nodes: int) -> ArtifactStep[LevanterCheckpoint]:
    """Retain exp232 m2-p06 training semantics while replacing its mixture."""
    if nodes not in (1, 2, 4, 8, 16):
        raise ValueError(f"Unsupported H100 gang size: {nodes}")
    per_device = min(8, GLOBAL_BATCH_SIZE // (8 * nodes))
    run_id = f"{RUN_ID}-smoke" if smoke else RUN_ID
    steps = 10 if smoke else NUM_TRAIN_STEPS
    env = dict(
        MARIN_PREFIX=PREFIX, WANDB_ENTITY="open-athena", WANDB_PROJECT="MarinFold"
    )
    datasets = {
        existing_cache(
            name=f"input/{corpus.name}",
            version=VERSION,
            source=corpus.cache,
            tags=["contacts-v1", "decontaminated", corpus.name],
        ): corpus.weight
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
            decay=DECAY,
            min_lr_ratio=MIN_LR_RATIO,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets=datasets,
        validation=[validation],
        init_from=None,
        batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=steps,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_gpu(
            "H100", count=8, replicas=nodes, cpu=32, ram="256g", disk="256g"
        ),
        tensor_parallel_size=1,
        steps_per_eval=steps if smoke else STEPS_PER_EVAL,
        wandb_project="MarinFold",
        wandb_group="exp277-single-mpnn-pilot",
        tags=[
            "exp277",
            "contacts-v1",
            "decontaminated",
            "mpnn",
            "m2",
            "p06",
            "scratch",
            "smoke" if smoke else "production",
            f"nodes={nodes}",
        ],
        env_vars=env,
    )
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            seed=MODEL_SEED,
            max_eval_batches=2 if smoke else None,
            watch=WANDB_WATCH,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                save_interval=timedelta(minutes=15),
                keep=[] if smoke else [{"every": PERMANENT_CHECKPOINT_EVERY}],
            ),
        )
        if not ctx.is_fingerprint:
            trainer = replace(
                trainer,
                per_device_parallelism=per_device,
                per_device_eval_parallelism=per_device,
            )
        data = replace(
            pod.train_config.data,
            auto_build_caches=False,
            shuffle=SHUFFLE,
            components={
                key: replace(value, pack=True)
                for key, value in pod.train_config.data.components.items()
            },
            block_cross_document_attention=True,
        )
        config = replace(
            pod.train_config,
            trainer=trainer,
            data=augment_amino_acids(data, steps),
            data_seed=DATA_SEED,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            hf_save_steps=steps + 1,
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
    for corpus in CORPORA:
        if not verify_cache(corpus):
            raise ValueError(f"Incomplete cache: {corpus.cache}")
    return build_run(
        smoke=os.environ.get("SMOKE") == "1", nodes=int(os.environ["NODES"])
    )


if __name__ == "__main__":
    main()

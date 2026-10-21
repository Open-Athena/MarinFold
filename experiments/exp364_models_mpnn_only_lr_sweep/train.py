"""Train exp277's default model on only redesigns for two full epochs."""

import os
from dataclasses import replace
from datetime import timedelta

import click
from fray.types import ResourceConfig
from levanter.data.text.datasets import ConcatDatasetComponent
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    DATA_SEED,
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
from experiments.exp277_models_single_mpnn_pilot.config import Corpus
from experiments.exp277_models_single_mpnn_pilot.prepare import verify_cache
from experiments.exp364_models_mpnn_only_lr_sweep.config import (
    CORPORA,
    EPOCH_PACKED_EXAMPLES,
    LEARNING_RATES,
    PREFIX,
    TRAIN_STEPS,
    VALIDATION_CACHE,
    VERSION,
    run_id_for,
)
from experiments.exp364_models_mpnn_only_lr_sweep.epoch_data import (
    FULL_CORPUS,
    two_epoch_data,
)
from experiments.exp364_models_mpnn_only_lr_sweep.runtime import run_train_job


def build_run(
    *,
    trial: str,
    smoke: bool,
    nodes: int,
    attempt: int = 1,
    corpora: tuple[Corpus, ...] = CORPORA,
    validation_cache: str = VALIDATION_CACHE,
    prefix: str = PREFIX,
) -> ArtifactStep[LevanterCheckpoint]:
    """Preserve the default recipe, changing training subset, duration, and LR."""
    if nodes not in (1, 2, 4, 8, 16):
        raise ValueError(f"Unsupported H100 gang size: {nodes}")
    per_device = min(8, GLOBAL_BATCH_SIZE // (8 * nodes))
    run_id = run_id_for(trial, smoke=smoke, attempt=attempt)
    steps = 10 if smoke else TRAIN_STEPS
    env = {
        "MARIN_PREFIX": prefix,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
    }
    datasets = {
        existing_cache(
            name=(
                f"input/two-epoch-smoke/{corpus.name}"
                if smoke
                else f"input/two-epoch/{corpus.name}"
            ),
            version=VERSION,
            source=corpus.cache,
            tags=["contacts-v1", "decontaminated", corpus.name],
        ): 1.0
        for corpus in corpora
    }
    validation = existing_cache(
        name="input/validation",
        version="2026.07.25",
        source=validation_cache,
        tags=["contacts-v1", "validation"],
    )
    step = train_lm(
        name=f"runs/{run_id}",
        run_id=run_id,
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=LEARNING_RATES[trial],
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
        wandb_group="exp364-mpnn-only-lr-sweep",
        tags=[
            "exp364",
            "contacts-v1",
            "decontaminated",
            "mpnn",
            "m2",
            "p06",
            "scratch",
            "mpnn-only-two-epochs",
            f"trial={trial}",
            "cluster=cw-us-east-02a",
            "gpu=H100",
            "smoke" if smoke else "production",
        ],
        env_vars=env,
    )
    base_build_config = step.build_config

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            seed=MODEL_SEED,
            max_eval_batches=2 if smoke else None,
            watch=WANDB_WATCH,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                save_interval=timedelta(minutes=15),
                keep=[] if smoke else [{"every": TRAIN_STEPS // 10}],
            ),
        )
        if not ctx.is_fingerprint:
            trainer = replace(
                trainer,
                per_device_parallelism=per_device,
                per_device_eval_parallelism=per_device,
                tracker=replace(
                    trainer.tracker, tags=[*trainer.tracker.tags, f"nodes={nodes}"]
                ),
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
            data=two_epoch_data(
                data,
                num_train_steps=steps,
                expected_packed_examples=EPOCH_PACKED_EXAMPLES,
            ),
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
        override_path=f"{prefix}/runs/{run_id}",
    )


@click.command()
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    smoke = os.environ.get("SMOKE") == "1"
    for corpus in CORPORA:
        if not verify_cache(corpus):
            raise ValueError(f"Incomplete cache: {corpus.cache}")
    return build_run(
        trial=os.environ["TRIAL"],
        smoke=smoke,
        nodes=int(os.environ["NODES"]),
        attempt=int(os.environ["ATTEMPT"]),
    )


if __name__ == "__main__":
    main()

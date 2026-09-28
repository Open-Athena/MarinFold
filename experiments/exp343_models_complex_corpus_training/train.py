# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Train one scratch model for one epoch over exp277's corpus plus complexes.

Every training setting is exp277's, imported rather than restated: exp232's
m2-p06 model, optimizer, schedule, seeds and shuffle, and exp277's finite
one-epoch data config that concatenates all sources and visits each packed
example once. The only differences are the fifth training corpus and the second
validation set.
"""

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
from experiments.exp277_models_single_mpnn_pilot.epoch_data import (
    FULL_CORPUS,
    one_epoch_data,
)
from experiments.exp343_models_complex_corpus_training.config import (
    COMPLEX_VALIDATION,
    CORPORA,
    EPOCH_PACKED_EXAMPLES,
    EPOCH_TRAIN_STEPS,
)
from experiments.exp343_models_complex_corpus_training.config import (
    GLOBAL_BATCH_SIZE as CONFIG_BATCH_SIZE,
)
from experiments.exp343_models_complex_corpus_training.config import (
    PREFIX,
    RUN_ID,
    VALIDATION_CACHE,
    VERSION,
)
from experiments.exp343_models_complex_corpus_training.prepare import verify_cache
from experiments.exp343_models_complex_corpus_training.runtime import run_train_job

#: exp277's learning rate and weight decay, which are exp232's m2-p06 winners.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.2


def build_run(*, smoke: bool, nodes: int) -> ArtifactStep[LevanterCheckpoint]:
    """exp277's training recipe over its corpus plus the #294 complex corpus."""
    if nodes not in (1, 2, 4, 8, 16):
        raise ValueError(f"Unsupported H100 gang size: {nodes}")
    if CONFIG_BATCH_SIZE != GLOBAL_BATCH_SIZE:
        raise ValueError(
            f"config.py derived the step count from batch {CONFIG_BATCH_SIZE}, "
            f"the training contract uses {GLOBAL_BATCH_SIZE}"
        )
    if EPOCH_PACKED_EXAMPLES <= 0:
        raise ValueError("EPOCH_PACKED_EXAMPLES is unpinned; run audit_epoch.py first")
    per_device = min(8, GLOBAL_BATCH_SIZE // (8 * nodes))
    run_id = f"{RUN_ID}-smoke" if smoke else RUN_ID
    steps = 10 if smoke else EPOCH_TRAIN_STEPS
    env = {
        "MARIN_PREFIX": PREFIX,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
    }
    scope = "full-epoch-smoke" if smoke else "full-epoch"
    datasets = {
        existing_cache(
            name=f"input/{scope}/{corpus.name}",
            version=VERSION,
            source=corpus.cache,
            tags=["contacts-v1", "decontaminated", corpus.name],
        ): 1.0
        for corpus in CORPORA
    }
    # Two validation sets: exp277's monomer set, so the loss curve is comparable
    # with the baseline's, and the held-out complex shard, which is the only
    # measurement of whether the new corpus was learned.
    monomer_validation = existing_cache(
        name="input/validation",
        version="2026.07.25",
        source=VALIDATION_CACHE,
        tags=["contacts-v1", "validation"],
    )
    complex_validation = existing_cache(
        name="input/validation-complex",
        version=VERSION,
        source=COMPLEX_VALIDATION.cache,
        tags=["contacts-v1", "complex", "validation", "held-out"],
    )
    validations = [monomer_validation, complex_validation]
    step = train_lm(
        name=f"runs/{run_id}",
        run_id=run_id,
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup=WARMUP,
            decay=DECAY,
            min_lr_ratio=MIN_LR_RATIO,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets=datasets,
        validation=validations,
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
        wandb_group="exp343-complex-corpus-training",
        tags=[
            "exp343",
            "contacts-v1",
            "decontaminated",
            "mpnn",
            "complex",
            "m2",
            "p06",
            "scratch",
            "full-corpus-one-epoch",
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
        packed_validations = {
            validation.name: replace(components[validation.name], pack=True)
            for validation in validations
        }
        data = replace(
            pod.train_config.data,
            auto_build_caches=False,
            shuffle=SHUFFLE,
            components={
                FULL_CORPUS: ConcatDatasetComponent(children=children),
                **packed_validations,
            },
            train_weights={
                FULL_CORPUS: 1.0,
                **{validation.name: 0.0 for validation in validations},
            },
            block_cross_document_attention=True,
        )
        config = replace(
            pod.train_config,
            trainer=trainer,
            data=one_epoch_data(
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
        override_path=f"{PREFIX}/runs/{run_id}",
    )


@click.command()
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    smoke = os.environ.get("SMOKE") == "1"
    for corpus in (*CORPORA, COMPLEX_VALIDATION):
        if not verify_cache(corpus):
            raise ValueError(f"Incomplete cache: {corpus.cache}/{corpus.split}")
    return build_run(smoke=smoke, nodes=int(os.environ["NODES"]))


if __name__ == "__main__":
    main()

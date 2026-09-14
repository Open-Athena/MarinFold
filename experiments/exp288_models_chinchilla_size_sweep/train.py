"""Train one scratch-initialized size-sweep arm for one complete corpus epoch."""

import math
import os
from dataclasses import dataclass, replace
from datetime import timedelta

import click
from levanter.data.text.datasets import ConcatDatasetComponent
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    DATA_SEED,
    DECAY,
    LR_SCHEDULE,
    MIN_LR_RATIO,
    MODEL_SEED,
    SEQ_LEN,
    SHUFFLE,
    STEPS_PER_EVAL,
    WANDB_WATCH,
    WARMUP,
    existing_cache,
)
from experiments.exp288_models_chinchilla_size_sweep.config import (
    CLUSTERS,
    CORPORA,
    EPOCH_PACKED_EXAMPLES,
    EPOCH_TRAIN_STEPS,
    GLOBAL_BATCH_SIZE,
    MAX_SEQS_PER_DEVICE,
    PREFIX,
    TRIALS,
    VALIDATION_CACHE,
    VERSION,
    WANDB_GROUP,
    ClusterSpec,
    SizeTrial,
    trainable_params,
)
from experiments.exp288_models_chinchilla_size_sweep.epoch_data import (
    FULL_CORPUS,
    one_epoch_data,
)
from experiments.exp288_models_chinchilla_size_sweep.prepare import verify_cache
from experiments.exp288_models_chinchilla_size_sweep.runtime import run_train_job


@dataclass(frozen=True)
class BatchFit:
    """Concrete mesh/microbatch settings for one placement."""

    tensor_parallelism: int
    per_device_parallelism: int


def _trial_from_env() -> SizeTrial:
    trial_id = os.environ.get("TRIAL")
    if trial_id not in TRIALS:
        raise ValueError(f"TRIAL must be one of {sorted(TRIALS)}, got {trial_id!r}")
    return TRIALS[trial_id]


def _placement_from_env(trial: SizeTrial) -> tuple[str, ClusterSpec, int]:
    cluster = os.environ.get("TARGET_CLUSTER", "cw-us-east-08a")
    if cluster not in CLUSTERS:
        raise ValueError(f"TARGET_CLUSTER must be one of {sorted(CLUSTERS)}, got {cluster!r}")
    nodes = int(os.environ.get("NODES", trial.nodes))
    allowed = {1, 2, 4, 8, 16}
    if CLUSTERS[cluster].gpu_variant == "GB200":
        allowed.update({32, 64})
    if nodes not in allowed:
        raise ValueError(f"NODES must be one of {sorted(allowed)}, got {nodes}")
    if cluster == "cw-rno2a" and nodes > 4:
        raise ValueError("cw-rno2a gangs above 4 nodes are not reliable")
    return cluster, CLUSTERS[cluster], nodes


def _batch_fit(spec: ClusterSpec, *, nodes: int) -> BatchFit:
    devices = spec.gpus_per_node * nodes
    data_parallelism = math.gcd(GLOBAL_BATCH_SIZE, devices)
    tensor_parallelism = devices // data_parallelism
    sequences_per_device = GLOBAL_BATCH_SIZE // data_parallelism
    per_device = min(sequences_per_device, MAX_SEQS_PER_DEVICE[spec.gpu_variant])
    while sequences_per_device % per_device:
        per_device -= 1
    return BatchFit(tensor_parallelism=tensor_parallelism, per_device_parallelism=per_device)


def build_run(
    *, trial: SizeTrial, smoke: bool, cluster: str, spec: ClusterSpec, nodes: int
) -> ArtifactStep[LevanterCheckpoint]:
    """Retain exp232 m2-p06 optimizer settings while varying only model size."""
    batch = _batch_fit(spec, nodes=nodes)
    run_id = f"{trial.run_id}-smoke" if smoke else trial.run_id
    steps = 10 if smoke else EPOCH_TRAIN_STEPS
    env = {
        "MARIN_PREFIX": PREFIX,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "EXP288_TRIAL": trial.trial_id,
        "EXP288_MODEL_PARAMS": str(trainable_params(trial.model)),
        "EXP288_TARGET_CLUSTER": cluster,
        "EXP288_GPU_VARIANT": spec.gpu_variant,
    }
    datasets = {
        existing_cache(
            name=(
                f"input/full-epoch-smoke/{corpus.name}"
                if smoke
                else f"input/full-epoch/{corpus.name}"
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
        model=trial.model,
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
        resources=spec.resources(nodes=nodes),
        tensor_parallel_size=batch.tensor_parallelism,
        steps_per_eval=steps if smoke else STEPS_PER_EVAL,
        wandb_project="MarinFold",
        wandb_group=WANDB_GROUP,
        tags=[
            "exp288",
            "contacts-v1",
            "decontaminated",
            "mpnn",
            "m2",
            "p06",
            "scratch",
            "full-corpus-one-epoch",
            "chinchilla-size-sweep",
            f"size={trial.label}",
            f"trial={trial.trial_id}",
            f"params={trainable_params(trial.model)}",
            "smoke" if smoke else "production",
            f"cluster={cluster}",
            f"gpu={spec.gpu_variant}",
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
                per_device_parallelism=batch.per_device_parallelism,
                per_device_eval_parallelism=batch.per_device_parallelism,
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
    trial = _trial_from_env()
    cluster, spec, nodes = _placement_from_env(trial)
    for corpus in CORPORA:
        if not verify_cache(corpus):
            raise ValueError(f"Incomplete cache: {corpus.cache}")
    return build_run(smoke=smoke, trial=trial, cluster=cluster, spec=spec, nodes=nodes)


if __name__ == "__main__":
    main()

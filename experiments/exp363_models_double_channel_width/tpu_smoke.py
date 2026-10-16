"""Benchmark the 5.862B architecture on existing us-east5 native caches.

This is a hardware smoke test, not a continuation of the native/MPNN production
run. It uses native AFDB and ESM data already mirrored into the TPU region;
the full production MPNN caches and optimizer state remain on CoreWeave.
"""

import os
from dataclasses import replace

import click
from fray.types import ResourceConfig
from levanter.layers.attention import AttentionBackend
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    AFDB_TOKENS,
    DATA_SEED,
    DECAY,
    ESM_TOKENS,
    GLOBAL_BATCH_SIZE,
    LR_SCHEDULE,
    MIN_LR_RATIO,
    MODEL_SEED,
    SEQ_LEN,
    SHUFFLE,
    WANDB_WATCH,
    WARMUP,
    augment_amino_acids,
    existing_cache,
)
from experiments.exp363_models_double_channel_width.config import MODEL_CONFIG, RUN_ID, TOKENIZER
from experiments.exp363_models_double_channel_width.runtime import run_train_job

PREFIX = "gs://marin-us-east5/protein-structure/MarinFold/exp363_models_double_channel_width"
CACHE_PREFIX = "gs://marin-us-east5/protein-structure/MarinFold/exp232_train_trc/tokenized"
VERSION = "2026.10.14.1"
SMOKE_STEPS = 20
TARGET_ZONES = {"v5p-64": "us-east5-a", "v6e-32": "us-east5-b"}


def build_run(*, tpu: str, attempt: int) -> ArtifactStep[LevanterCheckpoint]:
    """Keep architecture and batch shape fixed while selecting TPU kernels."""
    zone = TARGET_ZONES[tpu]
    run_id = f"{RUN_ID}-tpu-smoke-{tpu}-a{attempt:02d}"
    datasets = {
        existing_cache(
            name=f"input/tpu-smoke/native-{name}",
            version="2026.08.14",
            source=f"{CACHE_PREFIX}/contacts_v1/{name}/2026.08.14",
            tags=["contacts-v1", "decontaminated", name],
        ): tokens / (AFDB_TOKENS + ESM_TOKENS)
        for name, tokens in (("afdb", AFDB_TOKENS), ("esm", ESM_TOKENS))
    }
    validation = existing_cache(
        name="input/tpu-smoke/validation",
        version="2026.07.25",
        source=f"{CACHE_PREFIX}/contacts-v1-val/2026.07.25",
        tags=["contacts-v1", "validation"],
    )
    step = train_lm(
        name=f"runs/{run_id}",
        run_id=run_id,
        model=replace(MODEL_CONFIG, attn_backend=AttentionBackend.SPLASH),
        optimizer=AdamConfig(
            learning_rate=1e-3, weight_decay=0.2, warmup=WARMUP,
            decay=DECAY, min_lr_ratio=MIN_LR_RATIO, lr_schedule=LR_SCHEDULE,
        ),
        datasets=datasets,
        validation=[validation],
        init_from=None,
        batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=SMOKE_STEPS,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_tpu(
            tpu, zone=zone, regions=["us-east5"], preemptible=True,
        ),
        tensor_parallel_size=1,
        steps_per_eval=SMOKE_STEPS,
        wandb_project="MarinFold",
        wandb_group="exp363-double-channel-width",
        tags=["exp363", "double-channel-width", "tpu", "smoke", "native-only", tpu],
        env_vars={
            "MARIN_PREFIX": PREFIX,
            "WANDB_ENTITY": "open-athena",
            "WANDB_PROJECT": "MarinFold",
        },
    )
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            seed=MODEL_SEED,
            max_eval_batches=2,
            watch=WANDB_WATCH,
        )
        if not ctx.is_fingerprint:
            trainer = replace(
                trainer, per_device_parallelism=1, per_device_eval_parallelism=1,
            )
        data = replace(
            pod.train_config.data,
            tokenizer=TOKENIZER,
            auto_build_caches=False,
            shuffle=SHUFFLE,
            components={
                name: replace(component, pack=True)
                for name, component in pod.train_config.data.components.items()
            },
            block_cross_document_attention=True,
        )
        config = replace(
            pod.train_config,
            trainer=trainer,
            data=augment_amino_acids(data, SMOKE_STEPS),
            data_seed=DATA_SEED,
            hf_save_steps=SMOKE_STEPS + 1,
        )
        return replace(pod, train_config=config)

    return replace(
        step, build_config=build_config, run=run_train_job,
        override_path=f"{PREFIX}/runs/{run_id}",
    )


@click.command()
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    return build_run(tpu=os.environ["TPU"], attempt=int(os.environ["ATTEMPT"]))


if __name__ == "__main__":
    main()

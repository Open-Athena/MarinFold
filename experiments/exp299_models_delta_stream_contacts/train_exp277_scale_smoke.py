# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a ten-update training smoke on the exp277-scale V2 cache path."""

import dataclasses
import logging
import os
from datetime import timedelta
from pathlib import Path

import jmp
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import (
    BlockShuffleConfig,
    DatasetComponent,
    LmDataConfig,
)
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.run_environment import extras_for_resources
from marin.training.training import (
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)

from convert_exp277_caches_to_delta_stream import VOCAB_SIZE
from delta_stream_data import PackableTokenIdsFormat
from dispatch_delta_stream_full import MODEL_CONFIG, _mesh

PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp299_contacts_delta_stream_v2_sequence_prefix"
CACHE_ROOT = os.environ.get(
    "EXP299_CACHE_ROOT", f"{PREFIX}/exp277_full_epoch_tokenized_cache/2026.09.22.1-smoke-a01/mpnn-afdb"
)
OUTPUT_PREFIX = os.environ.get("EXP299_OUTPUT_PREFIX", f"{PREFIX}/exp277_scale_runs")
TOKENIZER_PATH = str(Path(__file__).with_name("tokenizer_exp277_v2"))
RUN_NAME = os.environ.get("EXP299_NAME", "delta-v2-exp277-full-epoch-1_5b-smoke-a01")
SEQ_LEN = 8192
GLOBAL_BATCH_SIZE = 128
STEPS = 10

logger = logging.getLogger(__name__)


def data_config() -> LmDataConfig:
    """Use ordinary Levanter packing with EOS-aware attention boundaries."""
    fmt = PackableTokenIdsFormat()
    component = dataclasses.replace(
        DatasetComponent(cache_dir=f"{CACHE_ROOT}/train", format=fmt, pack=True),
        split="train",
        flat_cache=True,
    )
    return LmDataConfig(
        tokenizer=TOKENIZER_PATH,
        vocab_size=VOCAB_SIZE,
        cache_dir=None,
        auto_build_caches=False,
        components={"delta-v2-train": component},
        train_weights={"delta-v2-train": 1.0},
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel"),
        mixture_block_size=1,
        block_cross_document_attention=True,
    )


def pod_config(resources: ResourceConfig) -> TrainLmOnPodConfig:
    output_path = f"{OUTPUT_PREFIX}/{RUN_NAME}"
    env_vars = {
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "JAX_COMPILATION_CACHE_DIR": "/tmp/jax-compilation-cache",
    }
    train_config = TrainLmConfig(
        data=data_config(),
        trainer=TrainerConfig(
            id=RUN_NAME,
            tracker=WandbConfig(
                project="MarinFold",
                name=RUN_NAME,
                group="exp299-delta-v2-exp277-scale",
                tags=["exp299", "delta-v2", "exp277-corpus", "smoke", "qwen3", "1_5b"],
                replicate_path=output_path,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=GLOBAL_BATCH_SIZE,
            per_device_parallelism=8,
            per_device_eval_parallelism=8,
            num_train_steps=STEPS,
            steps_per_eval=STEPS,
            max_eval_batches=2,
            checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=15), keep=[]),
            mesh=_mesh(1),
            allow_nondivisible_batch_size=True,
        ),
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=1e-3,
            weight_decay=0.2,
            beta1=0.9,
            beta2=0.95,
            warmup=0.1,
            decay=0.2,
            lr_schedule="linear",
            min_lr_ratio=0.1,
        ),
        hf_save_steps=None,
        train_seq_len=SEQ_LEN,
        data_seed=0,
        adapter=NoAdaptorConfig(),
    )
    return TrainLmOnPodConfig(
        train_config=train_config,
        resources=resources,
        output_path=output_path,
        env_vars=env_vars,
        auto_build_caches=False,
    )


def dispatch() -> None:
    """Submit the H100 smoke as a child of an authenticated CW root job."""
    resources = ResourceConfig.with_gpu("H100", count=8, replicas=1, cpu=32, ram="256GB", disk="256GB")
    config = pod_config(resources)
    env = resolve_training_env(base_env=dict(config.env_vars or {}), resources=resources)
    for key in (
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "FSSPEC_S3",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        if value := os.environ.get(key):
            env[key] = value
    request = JobRequest(
        name=RUN_NAME,
        entrypoint=Entrypoint.from_callable(run_levanter_train_lm, args=[config]),
        resources=resources,
        environment=create_environment(env_vars=env, extras=extras_for_resources(resources)),
        priority=3,
        processes_per_task=1,
        max_retries_failure=2,
    )
    job = current_client().submit(request)
    print(getattr(job, "name", str(job)), flush=True)
    job.wait(raise_on_failure=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dispatch()

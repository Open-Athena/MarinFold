# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch exp299 delta-stream full next-token training on CoreWeave GPUs."""

import logging
import os
from collections.abc import Sequence
from datetime import timedelta

import dataclasses
import jmp
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from haliax.partitioning import ResourceAxis
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import BlockShuffleConfig, DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.training.run_environment import extras_for_resources
from marin.training.training import TrainLmOnPodConfig, resolve_training_env, run_levanter_train_lm

PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp299_contacts_delta_stream_v1"
CACHE_ROOT = os.environ.get("EXP299_CACHE_ROOT", f"{PREFIX}/packed_cache/2026.09.15.1")
VOCAB_SIZE = 2080
SEQ_LEN = 8192
GLOBAL_BATCH_SIZE = 128
DEFAULT_STEPS = 12_000
RUN_NAME = "protein-delta-stream-1_5b-1e-3-2x4gb200-a08"
IRIS_PRIORITY_BAND_BATCH = 3

logger = logging.getLogger(__name__)

MODEL_CONFIG = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
    use_qk_norm=True,
    attn_backend=AttentionBackend.JAX_FLASH,
)


def _resources() -> ResourceConfig:
    return ResourceConfig.with_gpu(
        os.environ.get("EXP299_CW_GPU_TYPE", "GB200"),
        count=int(os.environ.get("EXP299_CW_GPUS", "4")),
        replicas=int(os.environ.get("EXP299_CW_NODES", "2")),
        cpu=float(os.environ.get("EXP299_CW_CPU", "32")),
        ram=os.environ.get("EXP299_CW_RAM", "256GiB"),
        disk=os.environ.get("EXP299_CW_DISK", "256GiB"),
    )


def _data_config() -> LmDataConfig:
    fmt = PrebuiltLmDatasetFormat(input_ids_key="input_ids", loss_weights_key="loss_weights")
    train = dataclasses.replace(DatasetComponent(cache_dir=CACHE_ROOT, format=fmt, pack=True), split="train")
    validation = dataclasses.replace(
        DatasetComponent(cache_dir=CACHE_ROOT, format=fmt, pack=True), split="validation"
    )
    return LmDataConfig(
        tokenizer="passthrough",
        vocab_size=VOCAB_SIZE,
        cache_dir=None,
        auto_build_caches=False,
        components={"delta-stream": train, "delta-stream-val": validation},
        train_weights={"delta-stream": 1.0, "delta-stream-val": 0.0},
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel"),
        mixture_block_size=1,
        block_cross_document_attention=False,
    )


def _optimizer() -> AdamConfig:
    return AdamConfig(
        learning_rate=float(os.environ.get("EXP299_LEARNING_RATE", "1e-3")),
        weight_decay=0.2,
        beta1=0.9,
        beta2=0.95,
        warmup=0.1,
        lr_schedule="cosine",
        min_lr_ratio=0.1,
    )


def _mesh(tensor_parallel_size: int) -> MeshConfig:
    token_axes = (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)
    return MeshConfig(
        axes={"replica": 1, "data": -1, "model": tensor_parallel_size},
        compute_mapping={"token": token_axes, "token_repeat": token_axes},
    )


def _pod_config(run_name: str, steps: int, resources: ResourceConfig) -> TrainLmOnPodConfig:
    output_path = f"{PREFIX}/checkpoints/{run_name}"
    env_vars = {
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "EXP299_CACHE_ROOT": CACHE_ROOT,
        "EXP299_DOCUMENT_STRUCTURE": "delta-stream-v1",
        "JAX_COMPILATION_CACHE_DIR": os.environ.get("EXP299_CW_JAX_CACHE_DIR", "/tmp/jax-compilation-cache"),
    }
    for key in ("WANDB_API_KEY", "FSSPEC_S3", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if value := os.environ.get(key):
            env_vars[key] = value

    tensor_parallel_size = int(os.environ.get("EXP299_CW_TENSOR_PARALLELISM", "1"))
    steps_per_eval = int(os.environ.get("EXP299_CW_STEPS_PER_EVAL", "250"))
    keep_every_steps = int(os.environ.get("EXP299_CW_KEEP_EVERY_STEPS", "2000"))
    checkpoint_interval_minutes = int(os.environ.get("EXP299_CW_CHECKPOINT_INTERVAL_MINUTES", "10"))
    max_eval_batches = int(os.environ.get("EXP299_CW_MAX_EVAL_BATCHES", "2"))

    train_config = TrainLmConfig(
        data=_data_config(),
        trainer=TrainerConfig(
            id=run_name,
            tracker=WandbConfig(
                project="MarinFold",
                name=run_name,
                group="exp299-delta-stream-full",
                tags=["protein", "delta-stream", "contacts", "qwen3", "1_5b", "2x4gb200"],
                replicate_path=output_path,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=int(os.environ.get("EXP299_CW_BATCH_SIZE", str(GLOBAL_BATCH_SIZE))),
            per_device_parallelism=-1,
            num_train_steps=steps,
            steps_per_eval=steps_per_eval,
            max_eval_batches=max_eval_batches,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=checkpoint_interval_minutes),
                keep=[dict(every=keep_every_steps)],
            ),
            mesh=_mesh(tensor_parallel_size),
            per_device_eval_parallelism=-1,
            allow_nondivisible_batch_size=True,
        ),
        model=MODEL_CONFIG,
        optimizer=_optimizer(),
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


def _env_for_pod(pod_config: TrainLmOnPodConfig) -> dict[str, str]:
    env = resolve_training_env(base_env=dict(pod_config.env_vars or {}), resources=pod_config.resources)
    for key in (
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_NAME",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "FSSPEC_S3",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env.setdefault("WANDB_ENTITY", "open-athena")
    env.setdefault("WANDB_PROJECT", "MarinFold")
    env.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax-compilation-cache")
    return env


def dispatch(wait: bool = True):
    steps = int(os.environ.get("EXP299_TRAIN_STEPS", str(DEFAULT_STEPS)))
    run_name = os.environ.get("EXP299_NAME") or os.environ.get("WANDB_NAME") or RUN_NAME
    resources = _resources()
    pod_config = _pod_config(run_name, steps, resources)
    environment = create_environment(env_vars=_env_for_pod(pod_config), extras=extras_for_resources(resources))
    request = JobRequest(
        name=run_name,
        entrypoint=Entrypoint.from_callable(run_levanter_train_lm, args=[pod_config]),
        resources=resources,
        environment=environment,
        priority=int(os.environ.get("EXP299_CW_PRIORITY_NUM", str(IRIS_PRIORITY_BAND_BATCH))),
        processes_per_task=1,
        max_retries_failure=int(os.environ.get("EXP299_CW_MAX_RETRIES", "3")),
    )
    logger.info("Dispatching exp299 delta-stream run %s -> %s", run_name, pod_config.output_path)
    job = current_client().submit(request)
    print(getattr(job, "name", str(job)))
    if wait:
        job.wait(raise_on_failure=True)
    return job


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dispatch(wait=os.environ.get("EXP299_CW_WAIT", "1") != "0")

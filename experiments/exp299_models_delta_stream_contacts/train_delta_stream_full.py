# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a Qwen-style LM on the full exp299 delta-stream contact cache."""

import dataclasses
import os
import uuid
from collections.abc import Sequence
from functools import partial

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.datasets import BlockShuffleConfig, DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution import ExecutorStep, executor_main, versioned
from marin.training.training import TrainLmOnPodConfig, resolve_training_env, run_levanter_train_lm

from marinfold_models.defaults import default_train
from marinfold_models.simple_train_config import SimpleTrainConfig

PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp299_contacts_delta_stream_v1"
CACHE_ROOT = f"{PREFIX}/cache/2026.09.15.3"
VOCAB_SIZE = 2080
SEQ_LEN = 8192
GLOBAL_BATCH_SIZE = 128
DEFAULT_STEPS = 12_000
RUN_NAME = "protein-delta-stream-1_5b-1e-3-2x4gb200-a01"

os.environ["MARIN_PREFIX"] = PREFIX

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

RESOURCES_2X4_GB200 = ResourceConfig.with_gpu(
    "GB200",
    count=4,
    replicas=2,
    cpu=32,
    ram="256g",
    disk="256g",
)


def delta_stream_data() -> LmDataConfig:
    """Use the prebuilt passthrough cache made from delta-stream token ids."""
    fmt = PrebuiltLmDatasetFormat(input_ids_key="input_ids")
    train = dataclasses.replace(
        DatasetComponent(cache_dir=CACHE_ROOT, format=fmt, pack=True),
        split="train",
    )
    validation = dataclasses.replace(
        DatasetComponent(cache_dir=CACHE_ROOT, format=fmt, pack=True),
        split="validation",
    )
    return LmDataConfig(
        tokenizer="passthrough",
        vocab_size=VOCAB_SIZE,
        cache_dir=None,
        auto_build_caches=False,
        components={"delta-stream": train, "delta-stream-val": validation},
        train_weights={"delta-stream": 1.0, "delta-stream-val": 0.0},
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel"),
        block_cross_document_attention=False,
    )


def run_train_job(config: TrainLmOnPodConfig) -> None:
    """Submit the actual 2x4 GPU gang from a small CPU executor step."""
    env = resolve_training_env(config.env_vars, config.resources)
    for key in ("WANDB_API_KEY", "FSSPEC_S3", "HF_TOKEN"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    handle = current_client().submit(
        JobRequest(
            name=f"exp299-delta-stream-train-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(partial(run_levanter_train_lm, config)),
            resources=config.resources,
            environment=create_environment(extras=["gpu"], env_vars=env),
            priority=3,
        )
    )
    handle.wait(raise_on_failure=True)


def build_train_step(
    *,
    name: str = RUN_NAME,
    num_train_steps: int = DEFAULT_STEPS,
    learning_rate: float = 1e-3,
    resources: ResourceConfig = RESOURCES_2X4_GB200,
    extra_tags: Sequence[str] = (),
) -> ExecutorStep:
    env_vars = {
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "EXP299_CACHE_ROOT": CACHE_ROOT,
        "EXP299_DOCUMENT_STRUCTURE": "delta-stream-v1",
    }
    for key in ("WANDB_API_KEY", "FSSPEC_S3", "HF_TOKEN"):
        if os.environ.get(key):
            env_vars[key] = os.environ[key]

    train_config = SimpleTrainConfig(
        resources=resources,
        train_batch_size=GLOBAL_BATCH_SIZE,
        num_train_steps=versioned(num_train_steps),
        learning_rate=versioned(learning_rate),
        weight_decay=0.2,
        warmup=0.1,
        train_seq_len=SEQ_LEN,
        steps_per_eval=250,
        steps_per_export=2_000,
        max_eval_batches=2,
        data_seed=versioned(0),
        tensor_parallel_size=1,
        per_device_parallelism=16,
        per_device_eval_parallelism=16,
        env_vars=env_vars,
        watch=WatchConfig(watch_targets=[], interval=0),
    )
    step = default_train(
        name=name,
        tokenized=delta_stream_data(),
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=["protein", "delta-stream", "contacts", "qwen3", "1_5b", "2x4gb200", *extra_tags],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group="exp299-delta-stream-full",
        wandb_name=name,
        override_output_path=f"{PREFIX}/checkpoints/{name}",
    )
    return dataclasses.replace(
        step,
        fn=run_train_job,
        resources=ResourceConfig(cpu=4, ram="16g", disk="32g"),
    )


if __name__ == "__main__":
    steps = int(os.environ.get("EXP299_TRAIN_STEPS", str(DEFAULT_STEPS)))
    run_name = os.environ.get("WANDB_NAME", RUN_NAME)
    executor_main(steps=[build_train_step(name=run_name, num_train_steps=steps)])

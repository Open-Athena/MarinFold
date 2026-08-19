# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch exp177 next-token training on CoreWeave H100s.

CoreWeave pods cannot read the GCS premade-contact shards used by the TPU
``premade_mp`` path, so this launcher uses the contacts-v1 corpus/cache already
staged in the CoreWeave S3 bucket by exp108. It directly submits the training
job as a Fray/Iris batch-priority gang, matching the exp108 CoreWeave recipe.
"""

import dataclasses
import logging
import os

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.data.text.datasets import BlockShuffleConfig
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env, run_levanter_train_lm

from marinfold_models import build_train_lm_on_pod_config
from train import EXP117_STEPS, MODEL_CONFIG, SEQ_LEN, _optimizer

logger = logging.getLogger(__name__)

IRIS_PRIORITY_BAND_BATCH = 3

CONTACTS_V1_TOKENIZER = "timodonnell/contacts-v1-tokenizer"
CONTACTS_V1_S3_PREFIX = "s3://marin-us-east-02a/MarinFold/exp177_soft_target_loss_h2h_cw"
CONTACTS_V1_S3_CORPUS_BASE = "s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1"
CONTACTS_V1_CACHE_BASE = "s3://marin-us-east-02a/MarinFold/exp108_qwen_3b_contacts_v1/tokenized"

_FORWARD_ENV_PREFIXES = ("XLA_FLAGS", "NCCL_", "JAX_", "LIBTPU_INIT_ARGS")
_FORWARD_ENV_EXCLUDE = ("JAX_PLATFORMS", "JAX_COMPILATION_CACHE_DIR")


def _forwarded_perf_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_FORWARD_ENV_PREFIXES) and key not in _FORWARD_ENV_EXCLUDE
    }


def _data_config() -> LmDataConfig:
    train_component = DatasetComponent(
        cache_dir=f"{CONTACTS_V1_CACHE_BASE}/contacts-v1",
        pack=True,
        split="train",
    )
    val_component = DatasetComponent(
        cache_dir=f"{CONTACTS_V1_CACHE_BASE}/contacts-v1-val",
        pack=True,
        split="validation",
    )
    return LmDataConfig(
        tokenizer=CONTACTS_V1_TOKENIZER,
        cache_dir=None,
        auto_build_caches=False,
        components={"contacts-v1": train_component, "contacts-v1-val": val_component},
        train_weights={"contacts-v1": 1.0, "contacts-v1-val": 0.0},
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel"),
        mixture_block_size=1,
        block_cross_document_attention=True,
    )


def _resources() -> ResourceConfig:
    nodes = int(os.environ.get("EXP177_CW_NODES", "4"))
    return ResourceConfig.with_gpu(
        "H100",
        count=8,
        cpu=float(os.environ.get("EXP177_CW_CPU", "32")),
        ram=os.environ.get("EXP177_CW_RAM", "256g"),
        disk=os.environ.get("EXP177_CW_DISK", "256g"),
        replicas=nodes,
    )


def _pod_config(run_name: str):
    resources = _resources()
    batch_size = int(os.environ.get("EXP177_CW_BATCH_SIZE", "128"))
    steps = int(os.environ.get("EXP177_CW_STEPS", str(EXP117_STEPS * 256 // batch_size)))
    steps_per_eval = int(os.environ.get("EXP177_CW_STEPS_PER_EVAL", str(max(1, steps // 32))))
    max_eval_batches_env = os.environ.get("EXP177_CW_MAX_EVAL_BATCHES")
    max_eval_batches = int(max_eval_batches_env) if max_eval_batches_env else None
    output_path = f"{CONTACTS_V1_S3_PREFIX}/checkpoints/{run_name}/{os.environ.get('EXP177_VERSION', '2026.08.03.1')}"
    env_vars = {
        **_forwarded_perf_env(),
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "EXP177_BACKEND": "coreweave",
        "EXP177_CW_NODES": str(resources.replicas),
        "EXP177_CW_BATCH_SIZE": str(batch_size),
        # CoreWeave pods do not have GCS credentials. Set an explicit local cache
        # so resolve_training_env() does not default to marin's GCS temp bucket.
        "JAX_COMPILATION_CACHE_DIR": os.environ.get("EXP177_CW_JAX_CACHE_DIR", "/tmp/jax-compilation-cache"),
    }
    for key in ("WANDB_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        if value := os.environ.get(key):
            env_vars[key] = value

    pod_config = build_train_lm_on_pod_config(
        run_name=run_name,
        model=MODEL_CONFIG,
        optimizer=_optimizer(),
        data=_data_config(),
        resources=resources,
        output_path=output_path,
        num_train_steps=steps,
        train_batch_size=batch_size,
        seq_len=SEQ_LEN,
        steps_per_eval=steps_per_eval,
        data_seed=0,
        wandb_project="MarinFold",
        wandb_group="exp177-coreweave-next-token",
        wandb_name=run_name,
        tags=("protein", "contacts-v1", "exp177", "qwen3", "from-scratch", "next-token", "coreweave"),
        env_vars=env_vars,
    )
    if max_eval_batches is None:
        return pod_config
    trainer = dataclasses.replace(pod_config.train_config.trainer, max_eval_batches=max_eval_batches)
    train_config = dataclasses.replace(pod_config.train_config, trainer=trainer)
    return dataclasses.replace(pod_config, train_config=train_config)


def dispatch(wait: bool = True):
    batch_size = int(os.environ.get("EXP177_CW_BATCH_SIZE", "128"))
    nodes = int(os.environ.get("EXP177_CW_NODES", "4"))
    default_name = (
        "exp177-cv1-1_5b-e16-lr3p162e-3-wd0p2-"
        f"bs{batch_size}-next_token-cw-h100x{nodes}-full"
    )
    run_name = os.environ.get("EXP177_NAME", default_name)
    pod_config = _pod_config(run_name)
    resources = pod_config.resources
    env_vars = dict(pod_config.env_vars or {})
    environment = create_environment(
        env_vars=resolve_training_env(base_env=dict(env_vars), resources=resources),
        extras=extras_for_resources(resources),
    )
    request = JobRequest(
        name=run_name,
        entrypoint=Entrypoint.from_callable(run_levanter_train_lm, args=[pod_config]),
        resources=resources,
        environment=environment,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=int(os.environ.get("EXP177_CW_MAX_RETRIES", "3")),
    )
    logger.info("Dispatching CoreWeave exp177 run %s -> %s", run_name, pod_config.output_path)
    job = current_client().submit(request)
    print(getattr(job, "name", str(job)))
    if wait:
        job.wait(raise_on_failure=True)
    return job


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dispatch(wait=os.environ.get("EXP177_CW_WAIT", "1") != "0")

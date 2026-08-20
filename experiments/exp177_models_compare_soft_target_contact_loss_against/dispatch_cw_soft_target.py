# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch exp177 soft-target training on CoreWeave H100s.

Uses the exp139 partially-digested contacts-v1 ESM-Atlas rows that already live
in CoreWeave S3. The training gang is submitted directly through Fray with Iris
batch priority, so child H100 jobs do not inherit the driver's interactive band.
"""

import dataclasses
import logging
import os
import shutil
import socket
from datetime import timedelta
from pathlib import Path

import fsspec
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from levanter.callbacks.profiler import ProfilerConfig, ProfileOptionsConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, DirectDatasetComponent, LmDataConfig
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

from marinfold_models import build_train_lm_on_pod_config
from premade_contacts_dataset import (
    MPFixedQuotaSoftTargetContactsDataset,
    MPPrecomputedSoftTargetContactsDataset,
    PrecomputedSoftTargetContactsDataset,
    SparsePrecomputedSoftTargetContactsDataset,
)
from train import (
    CONTACTS_TOKENIZER,
    EXP117_STEPS,
    MODEL_CONFIG,
    SEQ_LEN,
    _optimizer,
    _run_soft_target_with_pinned_tokenizer,
)

logger = logging.getLogger(__name__)

IRIS_PRIORITY_BAND_BATCH = 3
PROFILE_LOG_DIR = Path("/tmp/exp177-levanter-logs")

CW_ANALYZED_PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp139_esm_atlas_contacts_v1/analyzed"
CW_PRECOMPUTED_PREFIX = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp177_soft_target_loss_h2h/preprocessed/soft_target_compact_v1/2026.08.05.3"
)
CW_VAL_CACHE = "s3://marin-us-east-02a/MarinFold/exp108_qwen_3b_contacts_v1/tokenized/contacts-v1-val"
CW_OUTPUT_PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/checkpoints"
CW_SHARD_TEMPLATE = "analyzed-{shard_index:05d}-of-{total_shards:05d}.parquet"
CW_PRECOMPUTED_SHARD_TEMPLATE = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet"

_FORWARD_ENV_PREFIXES = ("XLA_FLAGS", "NCCL_", "JAX_", "LIBTPU_INIT_ARGS")
_FORWARD_ENV_EXCLUDE = ("JAX_PLATFORMS", "JAX_COMPILATION_CACHE_DIR")


def _forwarded_perf_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_FORWARD_ENV_PREFIXES) and key not in _FORWARD_ENV_EXCLUDE
    }


def _resources() -> ResourceConfig:
    return ResourceConfig.with_gpu(
        os.environ.get("EXP177_CW_GPU_TYPE", "H100"),
        count=int(os.environ.get("EXP177_CW_GPUS", "1")),
        cpu=float(os.environ.get("EXP177_CW_CPU", "32")),
        ram=os.environ.get("EXP177_CW_RAM", "256g"),
        disk=os.environ.get("EXP177_CW_DISK", "256g"),
        replicas=int(os.environ.get("EXP177_CW_NODES", "4")),
    )


def _data_config() -> LmDataConfig:
    data_kind = os.environ.get("EXP177_SOFT_TARGET_DATA", "precomputed")
    if data_kind == "precomputed":
        if os.environ.get("EXP177_SOFT_TARGET_BATCH", "compact") == "sparse":
            if os.environ.get("EXP177_PRECOMPUTED_MP", "1") != "0":
                raise ValueError("Sparse precomputed soft-target batches currently require EXP177_PRECOMPUTED_MP=0")
            dataset_cls = SparsePrecomputedSoftTargetContactsDataset
        else:
            dataset_cls = (
                PrecomputedSoftTargetContactsDataset
                if os.environ.get("EXP177_PRECOMPUTED_MP", "1") == "0"
                else MPPrecomputedSoftTargetContactsDataset
            )
        kwargs = {
            "data_prefix": os.environ.get("EXP177_PRECOMPUTED_SOFT_TARGET_PREFIX", CW_PRECOMPUTED_PREFIX).rstrip("/"),
            "num_shards": int(os.environ.get("EXP177_NUM_SHARDS", "3338")),
            "total_shards": 3338,
            "examples_per_shard": int(os.environ.get("EXP177_EXAMPLES_PER_SHARD", "2650")),
            "seed": 0,
            "max_seq_len": SEQ_LEN,
            "shard_cache_size": int(os.environ.get("EXP177_SHARD_CACHE_SIZE", "8")),
            "shard_name_template": os.environ.get(
                "EXP177_PRECOMPUTED_SOFT_TARGET_SHARD_NAME_TEMPLATE", CW_PRECOMPUTED_SHARD_TEMPLATE
            ),
        }
        if dataset_cls is SparsePrecomputedSoftTargetContactsDataset:
            kwargs.update(
                max_sparse_contacts=int(os.environ.get("EXP177_MAX_SPARSE_CONTACTS", "2048")),
                max_sparse_degree=int(os.environ.get("EXP177_MAX_SPARSE_DEGREE", "32")),
            )
        if dataset_cls is MPPrecomputedSoftTargetContactsDataset:
            kwargs.update(
                transform_workers=int(os.environ.get("EXP177_PRECOMPUTED_WORKERS", "16")),
                prefetch_chunks=int(os.environ.get("EXP177_PRECOMPUTED_PREFETCH_CHUNKS", "16")),
                chunk_size=int(os.environ.get("EXP177_PRECOMPUTED_CHUNK_SIZE", "64")),
                example_cache_size=int(os.environ.get("EXP177_PRECOMPUTED_EXAMPLE_CACHE_SIZE", "4096")),
                mp_start_method=os.environ.get(
                    "EXP177_PRECOMPUTED_MP_START_METHOD",
                    os.environ.get("EXP177_MP_START_METHOD", "spawn"),
                ),
            )
        train_dataset = dataset_cls(**kwargs)
    else:
        train_dataset = MPFixedQuotaSoftTargetContactsDataset(
            data_prefix=os.environ.get("EXP177_CW_CONTACTS_PREFIX", CW_ANALYZED_PREFIX).rstrip("/"),
            num_shards=int(os.environ.get("EXP177_NUM_SHARDS", "3338")),
            total_shards=3338,
            examples_per_shard=int(os.environ.get("EXP177_EXAMPLES_PER_SHARD", "2650")),
            seed=0,
            max_seq_len=SEQ_LEN,
            transform_workers=int(os.environ.get("EXP177_TRANSFORM_WORKERS", "28")),
            prefetch_shards=int(os.environ.get("EXP177_PREFETCH_SHARDS", "28")),
            shard_cache_size=int(os.environ.get("EXP177_SHARD_CACHE_SIZE", "8")),
            mp_start_method=os.environ.get("EXP177_MP_START_METHOD", "fork"),
            shard_name_template=os.environ.get("EXP177_CONTACTS_SHARD_NAME_TEMPLATE", CW_SHARD_TEMPLATE),
        )
    return LmDataConfig(
        components={
            "contacts-v1/soft_target": DirectDatasetComponent(datasets={"train": train_dataset}),
            "tokenized/contacts-v1-val": DatasetComponent(
                cache_dir=os.environ.get("EXP177_CW_VAL_CACHE", CW_VAL_CACHE).rstrip("/"),
                pack=True,
            ),
        },
        train_weights={"contacts-v1/soft_target": 1.0, "tokenized/contacts-v1-val": 0.0},
        tokenizer=CONTACTS_TOKENIZER,
        cache_dir=None,
        auto_build_caches=False,
        shuffle=False,
        mixture_block_size=1,
        block_cross_document_attention=True,
    )


def _upload_profile_logs(local_log_dir: Path, upload_prefix: str) -> None:
    """Upload local JAX profiler artifacts to S3 after training finishes."""
    if not local_log_dir.exists():
        logger.warning("Profile log directory does not exist: %s", local_log_dir)
        return

    hostname = socket.gethostname()
    for path in local_log_dir.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(local_log_dir)
        destination = f"{upload_prefix.rstrip('/')}/{hostname}/{relative_path.as_posix()}"
        logger.info("Uploading profiler artifact %s -> %s", path, destination)
        with path.open("rb") as src, fsspec.open(destination, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _run_soft_target_with_optional_profile_upload(pod_config, profile_upload_prefix: str | None = None):
    """Run soft-target training and persist local profiler traces when requested."""
    try:
        return _run_soft_target_with_pinned_tokenizer(pod_config)
    finally:
        if profile_upload_prefix:
            _upload_profile_logs(PROFILE_LOG_DIR, profile_upload_prefix)


def _pod_config(run_name: str):
    resources = _resources()
    batch_size = int(os.environ.get("EXP177_CW_BATCH_SIZE", "128"))
    steps = int(os.environ.get("EXP177_CW_STEPS", str(EXP117_STEPS * 256 // batch_size)))
    steps_per_eval = int(os.environ.get("EXP177_CW_STEPS_PER_EVAL", str(max(1, steps // 32))))
    checkpoint_interval_minutes = int(os.environ.get("EXP177_CW_CHECKPOINT_INTERVAL_MINUTES", "10"))
    keep_every_steps = int(os.environ.get("EXP177_CW_KEEP_EVERY_STEPS", str(steps)))
    max_eval_batches = int(os.environ.get("EXP177_CW_MAX_EVAL_BATCHES", "16"))
    per_device_parallelism = int(os.environ.get("EXP177_CW_PER_DEVICE_PARALLELISM", "1"))
    profiler_enabled = os.environ.get("EXP177_CW_PROFILER", "0") == "1"
    profiler_start_step = int(os.environ.get("EXP177_CW_PROFILER_START_STEP", "5"))
    profiler_num_steps = int(os.environ.get("EXP177_CW_PROFILER_NUM_STEPS", "10"))
    profiler_perfetto_link = os.environ.get("EXP177_CW_PROFILER_PERFETTO_LINK", "0") == "1"
    version = os.environ.get("EXP177_VERSION", "2026.08.03.2")
    output_path = f"{CW_OUTPUT_PREFIX}/{run_name}/{version}"
    env_vars = {
        **_forwarded_perf_env(),
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "EXP177_LOSS": "soft_target",
        "EXP177_BACKEND": "coreweave",
        "EXP177_SOFT_TARGET_MP": "1",
        "EXP177_SOFT_TARGET_DATA": os.environ.get("EXP177_SOFT_TARGET_DATA", "precomputed"),
        "EXP177_CONTACTS_PREFIX": os.environ.get("EXP177_CW_CONTACTS_PREFIX", CW_ANALYZED_PREFIX).rstrip("/"),
        "EXP177_CONTACTS_SHARD_NAME_TEMPLATE": os.environ.get("EXP177_CONTACTS_SHARD_NAME_TEMPLATE", CW_SHARD_TEMPLATE),
        "EXP177_PRECOMPUTED_SOFT_TARGET_PREFIX": os.environ.get(
            "EXP177_PRECOMPUTED_SOFT_TARGET_PREFIX", CW_PRECOMPUTED_PREFIX
        ).rstrip("/"),
        "EXP177_PRECOMPUTED_SOFT_TARGET_SHARD_NAME_TEMPLATE": os.environ.get(
            "EXP177_PRECOMPUTED_SOFT_TARGET_SHARD_NAME_TEMPLATE", CW_PRECOMPUTED_SHARD_TEMPLATE
        ),
        "EXP177_TRANSFORM_WORKERS": os.environ.get("EXP177_TRANSFORM_WORKERS", "28"),
        "EXP177_PREFETCH_SHARDS": os.environ.get("EXP177_PREFETCH_SHARDS", "28"),
        "EXP177_SHARD_CACHE_SIZE": os.environ.get("EXP177_SHARD_CACHE_SIZE", "8"),
        "EXP177_MP_START_METHOD": os.environ.get("EXP177_MP_START_METHOD", "fork"),
        "EXP177_PRECOMPUTED_MP": os.environ.get("EXP177_PRECOMPUTED_MP", "1"),
        "EXP177_SOFT_TARGET_BATCH": os.environ.get("EXP177_SOFT_TARGET_BATCH", "compact"),
        "EXP177_CW_GPU_TYPE": os.environ.get("EXP177_CW_GPU_TYPE", "H100"),
        "EXP177_MAX_SPARSE_CONTACTS": os.environ.get("EXP177_MAX_SPARSE_CONTACTS", "2048"),
        "EXP177_MAX_SPARSE_DEGREE": os.environ.get("EXP177_MAX_SPARSE_DEGREE", "32"),
        "EXP177_PRECOMPUTED_WORKERS": os.environ.get("EXP177_PRECOMPUTED_WORKERS", "16"),
        "EXP177_PRECOMPUTED_PREFETCH_CHUNKS": os.environ.get("EXP177_PRECOMPUTED_PREFETCH_CHUNKS", "16"),
        "EXP177_PRECOMPUTED_CHUNK_SIZE": os.environ.get("EXP177_PRECOMPUTED_CHUNK_SIZE", "64"),
        "EXP177_PRECOMPUTED_EXAMPLE_CACHE_SIZE": os.environ.get("EXP177_PRECOMPUTED_EXAMPLE_CACHE_SIZE", "4096"),
        "EXP177_PRECOMPUTED_MP_START_METHOD": os.environ.get(
            "EXP177_PRECOMPUTED_MP_START_METHOD",
            os.environ.get("EXP177_MP_START_METHOD", "spawn"),
        ),
        "EXP177_DATALOADER_PREFETCH_SIZE": os.environ.get("EXP177_DATALOADER_PREFETCH_SIZE", "8"),
        "EXP177_DATALOADER_MAX_BUFFERED_BATCHES": os.environ.get("EXP177_DATALOADER_MAX_BUFFERED_BATCHES", "64"),
        # CoreWeave pods do not have GCS credentials. Set an explicit local cache
        # so resolve_training_env() does not default to marin's GCS temp bucket.
        "JAX_COMPILATION_CACHE_DIR": os.environ.get("EXP177_CW_JAX_CACHE_DIR", "/tmp/jax-compilation-cache"),
    }
    for key in (
        "WANDB_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
        "TF_GPU_ALLOCATOR",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    ):
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
        z_loss_weight=0.0,
        tensor_parallel_size=int(os.environ.get("EXP177_CW_TENSOR_PARALLELISM", "1")),
        data_seed=0,
        wandb_project="MarinFold",
        wandb_group="exp177-soft-target-loss-h2h-coreweave",
        wandb_name=run_name,
        tags=("protein", "contacts-v1", "exp177", "qwen3", "from-scratch", "loss=soft_target", "coreweave"),
        env_vars=env_vars,
    )
    trainer = dataclasses.replace(
        pod_config.train_config.trainer,
        max_eval_batches=max_eval_batches,
        per_device_parallelism=per_device_parallelism,
        per_device_eval_parallelism=per_device_parallelism,
        log_dir=PROFILE_LOG_DIR if profiler_enabled else pod_config.train_config.trainer.log_dir,
        checkpointer=CheckpointerConfig(
            save_interval=timedelta(minutes=checkpoint_interval_minutes),
            keep=[{"every": keep_every_steps}],
        ),
        profiler=ProfilerConfig(
            enabled=profiler_enabled,
            start_step=profiler_start_step,
            num_steps=profiler_num_steps,
            perfetto_link=profiler_perfetto_link,
            profile_options=ProfileOptionsConfig(
                host_tracer_level=int(os.environ["EXP177_CW_PROFILER_HOST_TRACER_LEVEL"])
                if "EXP177_CW_PROFILER_HOST_TRACER_LEVEL" in os.environ
                else None,
                python_tracer_level=int(os.environ["EXP177_CW_PROFILER_PYTHON_TRACER_LEVEL"])
                if "EXP177_CW_PROFILER_PYTHON_TRACER_LEVEL" in os.environ
                else None,
                device_tracer_level=int(os.environ["EXP177_CW_PROFILER_DEVICE_TRACER_LEVEL"])
                if "EXP177_CW_PROFILER_DEVICE_TRACER_LEVEL" in os.environ
                else None,
                include_dataset_ops=os.environ.get("EXP177_CW_PROFILER_INCLUDE_DATASET_OPS") == "1"
                if "EXP177_CW_PROFILER_INCLUDE_DATASET_OPS" in os.environ
                else None,
            ),
        ),
    )
    train_config = dataclasses.replace(pod_config.train_config, trainer=trainer, hf_save_steps=steps)
    return dataclasses.replace(pod_config, train_config=train_config, auto_build_caches=False)


def _run_name() -> str:
    batch_size = int(os.environ.get("EXP177_CW_BATCH_SIZE", "128"))
    nodes = int(os.environ.get("EXP177_CW_NODES", "4"))
    gpus = int(os.environ.get("EXP177_CW_GPUS", "1"))
    default_name = (
        "exp177-cv1-1_5b-e16-lr3p162e-3-wd0p2-"
        f"bs{batch_size}-soft_target-cw-h100x{nodes}x{gpus}-precomputed"
    )
    return os.environ.get("EXP177_NAME", default_name)


def dispatch(wait: bool = True):
    run_name = _run_name()
    pod_config = _pod_config(run_name)
    environment = create_environment(
        env_vars=resolve_training_env(base_env=dict(pod_config.env_vars or {}), resources=pod_config.resources),
        extras=extras_for_resources(pod_config.resources),
    )
    profile_upload_prefix = None
    if os.environ.get("EXP177_CW_PROFILER", "0") == "1":
        profile_upload_prefix = f"{pod_config.output_path.rstrip('/')}/profile-logs"
    request = JobRequest(
        name=run_name,
        entrypoint=Entrypoint.from_callable(
            _run_soft_target_with_optional_profile_upload,
            args=[pod_config, profile_upload_prefix],
        ),
        resources=pod_config.resources,
        environment=environment,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=int(os.environ.get("EXP177_CW_MAX_RETRIES", "3")),
    )
    logger.info("Dispatching CoreWeave exp177 soft-target run %s -> %s", run_name, pod_config.output_path)
    job = current_client().submit(request)
    print(getattr(job, "name", str(job)))
    if wait:
        job.wait(raise_on_failure=True)
    return job


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("EXP177_CW_INLINE", "0") == "1":
        run_name = _run_name()
        logger.info("Running CoreWeave exp177 soft-target inline %s", run_name)
        pod_config = _pod_config(run_name)
        profile_upload_prefix = None
        if os.environ.get("EXP177_CW_PROFILER", "0") == "1":
            profile_upload_prefix = f"{pod_config.output_path.rstrip('/')}/profile-logs"
        _run_soft_target_with_optional_profile_upload(pod_config, profile_upload_prefix)
    else:
        dispatch(wait=os.environ.get("EXP177_CW_WAIT", "1") != "0")

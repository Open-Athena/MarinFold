# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch exp177 position-loss diagnostic to one CoreWeave H100 node."""

import logging
import os
import subprocess
from pathlib import Path

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

LOGGER = logging.getLogger(__name__)
IRIS_PRIORITY_BAND_BATCH = 3


def _resources() -> ResourceConfig:
    return ResourceConfig.with_gpu(
        os.environ.get("EXP177_DIAG_GPU_TYPE", "H100"),
        count=int(os.environ.get("EXP177_DIAG_GPUS", "8")),
        cpu=float(os.environ.get("EXP177_DIAG_CPU", "32")),
        ram=os.environ.get("EXP177_DIAG_RAM", "256g"),
        disk=os.environ.get("EXP177_DIAG_DISK", "256g"),
        replicas=int(os.environ.get("EXP177_DIAG_NODES", "1")),
    )


def _run_child() -> None:
    cmd = ["uv", "run", "python", "position_loss_diagnostic.py"]
    LOGGER.info("Running %s in %s", " ".join(cmd), Path.cwd())
    subprocess.run(cmd, check=True)


def dispatch(wait: bool = True):
    resources = _resources()
    extras = extras_for_resources(resources)
    env_vars = {
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "USE_TORCH": os.environ.get("USE_TORCH", "0"),
        "USE_TF": os.environ.get("USE_TF", "0"),
        "JAX_COMPILATION_CACHE_DIR": os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/jax-compilation-cache"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
        "TF_GPU_ALLOCATOR": os.environ.get("TF_GPU_ALLOCATOR", "cuda_malloc_async"),
    }
    for key in (
        "EXP177_DIAG_CHECKPOINTS_JSON",
        "EXP177_DIAG_OUTPUT_PREFIX",
        "EXP177_DIAG_VAL_DOCS",
        "EXP177_DIAG_TRAIN_DOCS",
        "EXP177_DIAG_TRAIN_SEED",
        "EXP177_DIAG_EVAL_BATCH_SIZE",
        "EXP177_DIAG_LOG_EVERY",
        "EXP177_DIAG_TRAIN_CACHE",
        "EXP177_DIAG_VAL_CACHE",
        "EXP177_MAX_SPARSE_CONTACTS",
        "EXP177_MAX_SPARSE_DEGREE",
        "EXP177_DIAG_MAX_ABS_BINS",
        "EXP177_DIAG_PERCENTILE_BINS",
        "WANDB_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
        "FSSPEC_S3",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
        "AWS_DEFAULT_REGION",
    ):
        if value := os.environ.get(key):
            env_vars[key] = value
    environment = create_environment(
        env_vars=resolve_training_env(base_env=env_vars, resources=resources),
        extras=extras,
        setup_scripts=[default_setup_script(extras=extras), cuda_toolchain_setup_script()],
    )
    name = os.environ.get("EXP177_DIAG_JOB_NAME", "exp177-position-loss-diag-h100x8-r1")
    request = JobRequest(
        name=name,
        entrypoint=Entrypoint.from_callable(_run_child),
        resources=resources,
        environment=environment,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=int(os.environ.get("EXP177_DIAG_MAX_RETRIES", "1")),
    )
    LOGGER.info("Dispatching %s", name)
    job = current_client().submit(request)
    print(getattr(job, "name", str(job)))
    if wait:
        job.wait(raise_on_failure=True)
    return job


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dispatch(wait=os.environ.get("EXP177_DIAG_WAIT", "1") != "0")

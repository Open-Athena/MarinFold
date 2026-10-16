"""Submit the width experiment at batch priority and wait for its accelerator gang."""

import json
import os
import sys
import uuid
from functools import partial

import fsspec
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, TpuConfig, create_environment
from iris.cluster.setup_scripts import default_setup_script
from marin.training.training import (
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)

from experiments.exp277_models_single_mpnn_pilot.runtime import (
    _pinned_cuda_toolchain_setup_script,
)


def require_final_checkpoint(output_path: str, num_train_steps: int) -> None:
    """Reject a zero-exit training job that stopped before its final checkpoint."""
    expected_step = num_train_steps - 1
    path = f"{output_path}/checkpoints/step-{expected_step}/metadata.json"
    try:
        with fsspec.open(path, "rt") as handle:
            metadata = json.load(handle)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Training job exited without its final step-{expected_step} checkpoint: {path}"
        ) from error
    if metadata["step"] != expected_step:
        raise ValueError(f"Final checkpoint step mismatch at {path}: {metadata['step']}")


def run_train_job(config: TrainLmOnPodConfig) -> None:
    """Keep the CPU parent alive and allow bounded production recovery."""
    if config.output_path is None:
        raise ValueError("Training requires an output path for its completion checkpoint")
    env = resolve_training_env(config.env_vars, config.resources)
    tpu = isinstance(config.resources.device, TpuConfig)
    keys = ("WANDB_API_KEY",) if tpu else ("WANDB_API_KEY", "FSSPEC_S3")
    env.update({key: os.environ[key] for key in keys})
    extras = ["tpu"] if tpu else ["gpu"]
    setup_scripts = [
        default_setup_script(
            extras=extras,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        )
    ]
    if not tpu:
        setup_scripts.append(_pinned_cuda_toolchain_setup_script())
    smoke = "smoke" in config.train_config.trainer.id
    handle = current_client().submit(
        JobRequest(
            name=f"exp363-train-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(partial(run_levanter_train_lm, config)),
            resources=config.resources,
            environment=create_environment(
                extras=extras,
                env_vars=env,
                setup_scripts=setup_scripts,
            ),
            priority=3,
            max_task_failures=0 if smoke else 20,
            max_retries_failure=0 if smoke else 5,
        )
    )
    handle.wait(raise_on_failure=True)
    require_final_checkpoint(config.output_path, config.train_config.trainer.num_train_steps)

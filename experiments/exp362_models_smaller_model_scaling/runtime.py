"""Submit the training gang at batch priority and await its completion."""

import os
import sys
import uuid
from functools import partial

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.cluster.setup_scripts import default_setup_script
from marin.training.training import TrainLmOnPodConfig, resolve_training_env, run_levanter_train_lm

from experiments.exp277_models_single_mpnn_pilot.runtime import _pinned_cuda_toolchain_setup_script


def run_train_job(config: TrainLmOnPodConfig) -> None:
    """Forward regional storage credentials and preserve the baseline CUDA stack."""
    env = resolve_training_env(config.env_vars, config.resources)
    env.update({key: os.environ[key] for key in ("WANDB_API_KEY", "FSSPEC_S3")})
    handle = current_client().submit(
        JobRequest(
            name=f"exp362-train-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(partial(run_levanter_train_lm, config)),
            resources=config.resources,
            environment=create_environment(
                extras=["gpu"],
                env_vars=env,
                setup_scripts=[
                    default_setup_script(
                        extras=["gpu"],
                        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                    ),
                    _pinned_cuda_toolchain_setup_script(),
                ],
            ),
            priority=3,
        )
    )
    handle.wait(raise_on_failure=True)

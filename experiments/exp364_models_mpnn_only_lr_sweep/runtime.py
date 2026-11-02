"""Batch-priority GPU submission, preserving exp277 CUDA setup and resume."""

import json
import os
import sys
import uuid

import cloudpickle
import fsspec
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script
from marin.training.training import (
    TrainLmOnPodConfig,
    resolve_training_env,
)

CUDNN_X86_WHEEL = (
    "https://pypi.nvidia.com/nvidia-cudnn-cu13/"
    "nvidia_cudnn_cu13-9.26.0.17.dev59162438-"
    "py3-none-manylinux_2_27_x86_64.whl"
)
CUDNN_ARM_WHEEL = (
    "https://pypi.nvidia.com/nvidia-cudnn-cu13/"
    "nvidia_cudnn_cu13-9.26.0.17.dev59162438-"
    "py3-none-manylinux_2_27_aarch64.whl"
)


def _pinned_cuda_toolchain_setup_script() -> str:
    """Keep Iris's CUDA precedence repair deterministic for the locked cuDNN."""
    script = cuda_toolchain_setup_script()
    install_marker = '    uv pip install --python "$IRIS_VENV/bin/python" \\\n'
    package_marker = '      "$_cuda13_package==$_cuda13_version"'
    if script.count(install_marker) != 1 or script.count(package_marker) != 1:
        raise ValueError("Iris CUDA setup script changed; update the exp232 pin")
    choose_spec = f"""    if [ "$_cuda13_package" = "nvidia-cudnn-cu13" ]; then
      case "$(uname -m)" in
        x86_64) _cuda13_spec={CUDNN_X86_WHEEL!r} ;;
        aarch64) _cuda13_spec={CUDNN_ARM_WHEEL!r} ;;
        *) _cuda13_spec="$_cuda13_package==$_cuda13_version" ;;
      esac
    else
      _cuda13_spec="$_cuda13_package==$_cuda13_version"
    fi
"""
    return script.replace(install_marker, choose_spec + install_marker).replace(
        package_marker,
        '      "$_cuda13_spec"',
    )


def run_train_job(config: TrainLmOnPodConfig) -> None:
    """Submit one GPU gang and keep its CPU parent alive until completion."""
    env = resolve_training_env(config.env_vars, config.resources)
    env.update({key: os.environ[key] for key in ("WANDB_API_KEY", "FSSPEC_S3")})
    storage = json.loads(env["FSSPEC_S3"])
    env.update(
        CW_KEY_ID=storage["key"],
        CW_KEY_SECRET=storage["secret"],
        AWS_ACCESS_KEY_ID=storage["key"],
        AWS_SECRET_ACCESS_KEY=storage["secret"],
    )
    worker_name = f"exp364-train-{uuid.uuid4().hex[:8]}"
    config_path = f"{config.output_path}/launches/{worker_name}.pkl"
    with fsspec.open(config_path, "wb") as stream:
        cloudpickle.dump(config, stream)
    handle = current_client().submit(
        JobRequest(
            name=worker_name,
            # Iris's callable runner catches exceptions before sys.last_exc is
            # set, defeating Levanter's clean-exit completion hook. A normal
            # Python module leaves failures unhandled and preserves retry state.
            entrypoint=Entrypoint.from_binary(
                "python",
                [
                    "-u",
                    "-m",
                    "experiments.exp364_models_mpnn_only_lr_sweep.gpu_worker",
                    config_path,
                ],
            ),
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
            # `max_task_failures` defaults to 0: ONE failed task attempt ends the
            # whole gang. exp364-train-a02 died that way at step 150,223 after
            # 22 hours -- iris had absorbed six batch-priority preemptions
            # (`max_retries_preemption` already defaults to 100) and then a single
            # task failure in the churn was terminal. A multi-day run on a
            # contended cluster has to outlive that.
            #
            # Deliberately not unbounded: the smoke validates the configuration
            # before production, so a persistent failure here is a real defect and
            # should still surface rather than retry forever.
            max_task_failures=20,
            max_retries_failure=5,
        )
    )
    handle.wait(raise_on_failure=True)
    # An externally stopped training process can exit zero. Its parent must
    # not turn that into a cached SUCCESS that prevents later checkpoint resume.
    final_step = config.train_config.trainer.num_train_steps - 1
    metadata_path = f"{config.output_path}/checkpoints/step-{final_step}/metadata.json"
    with fsspec.open(metadata_path) as stream:
        metadata = json.load(stream)
    if metadata["step"] != final_step:
        raise ValueError(f"Training stopped without the expected step {final_step}")

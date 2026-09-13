"""Batch-priority training submission with exp232's pinned CUDA repair."""

import os
import sys
import uuid
from functools import partial

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script
from marin.training.training import (
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
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
    handle = current_client().submit(
        JobRequest(
            name=f"exp277-train-{uuid.uuid4().hex[:8]}",
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

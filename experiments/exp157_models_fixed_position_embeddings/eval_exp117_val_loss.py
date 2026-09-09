# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Directly evaluate exp117 on exp157's contacts-v1 validation cache.

This is an eval-only Levanter run: it stages the published exp117 HF export into
local pod storage, loads it as the initial model, runs one zero-learning-rate training step, and
forces the full validation-loss hook over exp157's cached
``contacts-v1-val`` split.
"""

import dataclasses
import logging
import os
import shutil
from pathlib import Path

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from huggingface_hub import snapshot_download
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env, run_levanter_train_lm

from contacts_v1_train_common import CONTACTS_V1_S3_PREFIX, PROTEIN_RESOURCES_H100
from dispatch_train import build_on_pod_config

logger = logging.getLogger(__name__)

EXP117_REPO = "open-athena/marinfold-exp117"
EXP117_SUBFOLDER = (
    "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4/"
    "hf/step-35679"
)
EXP117_FILES = (
    "config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)


def _stage_exp117_hf() -> str:
    """Stage the exp117 HF subfolder as a flat local HF checkpoint directory."""

    source_root = Path(
        snapshot_download(
            repo_id=EXP117_REPO,
            revision="main",
            allow_patterns=[f"{EXP117_SUBFOLDER}/{name}" for name in EXP117_FILES],
            local_dir="/tmp/exp117_hf_snapshot",
            max_workers=8,
            token=os.environ.get("HF_TOKEN"),
        )
    )
    destination = Path("/tmp/exp117_hf_flat")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in EXP117_FILES:
        shutil.copy2(source_root / EXP117_SUBFOLDER / name, destination / name)
    return str(destination)


def run_eval_only() -> None:
    """Run full validation loss for exp117 on the exp157 cache."""

    checkpoint_path = _stage_exp117_hf()
    run_name = os.environ.get(
        "EXP157_EXP117_EVAL_NAME",
        "exp157-exp117-direct-val-loss-exp108-cache-r1",
    )
    output_path = f"{CONTACTS_V1_S3_PREFIX}/eval_loss/{run_name}"
    pod_config = build_on_pod_config(
        run_name=run_name,
        model_config=MODEL_CONFIG,
        learning_rate=0.0,
        num_train_steps=1,
        train_batch_size=128,
        seq_len=8192,
        weight_decay=0.0,
        warmup=0.0,
        output_path=output_path,
        resources=PROTEIN_RESOURCES_H100,
        env_vars={"WANDB_ENTITY": "open-athena"},
        wandb_name=run_name,
        tags=("protein", "contacts-v1", "exp157", "exp117", "eval-only"),
        wandb_group="exp157-fixed-position-embeddings",
        steps_per_eval=1,
        max_eval_batches=None,
    )
    train_config = dataclasses.replace(
        pod_config.train_config,
        initialize_from_hf=checkpoint_path,
        pad_tokenizer_to_match_model=True,
    )
    # Keep the checkpointer configured: Marin's output-path resolver expects the
    # dataclass to be present even for this zero-step eval-only run.
    pod_config = dataclasses.replace(pod_config, train_config=train_config)
    run_levanter_train_lm(pod_config)


def dispatch() -> None:
    """Submit the eval-only H100 job from an Iris CPU driver."""

    run_name = os.environ.get(
        "EXP157_EXP117_EVAL_NAME",
        "exp157-exp117-direct-val-loss-exp108-cache-r1",
    )
    env_vars = {"WANDB_ENTITY": "open-athena"}
    for key in ("WANDB_API_KEY", "HF_TOKEN"):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    environment = create_environment(
        env_vars=resolve_training_env(base_env=env_vars, resources=PROTEIN_RESOURCES_H100),
        extras=extras_for_resources(PROTEIN_RESOURCES_H100),
    )
    request = JobRequest(
        name=run_name,
        entrypoint=Entrypoint.from_callable(run_eval_only),
        resources=PROTEIN_RESOURCES_H100,
        environment=environment,
        priority=3,
        processes_per_task=1,
        max_retries_failure=0,
        max_retries_preemption=20,
    )
    job = current_client().submit(request)
    print(job.job_id, flush=True)
    job.wait(raise_on_failure=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dispatch()

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct CoreWeave batch-priority dispatcher for exp157 training smokes."""

import dataclasses
import logging
import os

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from levanter.data.text.datasets import BlockShuffleConfig, DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.models.lm_model import LmConfig
from levanter.optim.config import AdamConfig
from marin.training.run_environment import extras_for_resources
from marin.training.training import TrainLmOnPodConfig, resolve_training_env, run_levanter_train_lm

from contacts_v1_train_common import (
    CONTACTS_V1_DATA_SEED,
    CONTACTS_V1_S3_CORPUS_BASE,
    CONTACTS_V1_S3_PREFIX,
    CONTACTS_V1_TOKEN_CACHE_BASE,
    CONTACTS_V1_TOKENIZER,
    PROTEIN_RESOURCES_H100,
)
from marinfold_models import build_train_lm_on_pod_config

logger = logging.getLogger(__name__)

IRIS_PRIORITY_BAND_BATCH = 3
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires "
    "the 0.2.x.dev fray line, not the frozen 0.99.dev build."
)

_FORWARD_ENV_PREFIXES = ("XLA_FLAGS", "NCCL_", "JAX_", "LIBTPU_INIT_ARGS", "TF_GPU_ALLOCATOR")
_FORWARD_ENV_EXCLUDE = ("JAX_PLATFORMS",)


def _forwarded_perf_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k.startswith(_FORWARD_ENV_PREFIXES) and k not in _FORWARD_ENV_EXCLUDE
    }


def build_data_config() -> LmDataConfig:
    """Build concrete-path contacts-v1 data config using CoreWeave S3 caches."""
    cache_base = CONTACTS_V1_TOKEN_CACHE_BASE
    train_source = UrlDatasetSourceConfig(
        train_urls=[f"{CONTACTS_V1_S3_CORPUS_BASE}/train/*.parquet"],
        validation_urls=[],
        cache_dir=f"{cache_base}/contacts-v1",
        format=TextLmDatasetFormat(text_key="document"),
    )
    val_source = UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[f"{CONTACTS_V1_S3_CORPUS_BASE}/val/*.parquet"],
        cache_dir=f"{cache_base}/contacts-v1-val",
        format=TextLmDatasetFormat(text_key="document"),
    )
    train_component = DatasetComponent(
        source=train_source,
        cache_dir=train_source.cache_dir,
        format=train_source.format,
        pack=True,
        split="train",
    )
    val_component = DatasetComponent(
        source=val_source,
        cache_dir=val_source.cache_dir,
        format=val_source.format,
        pack=True,
        split="validation",
    )
    return LmDataConfig(
        tokenizer=CONTACTS_V1_TOKENIZER,
        cache_dir=None,
        auto_build_caches=False,
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel"),
        block_cross_document_attention=True,
        components={"contacts-v1": train_component, "contacts-v1-val": val_component},
        train_weights={"contacts-v1": 1.0, "contacts-v1-val": 0.0},
    )


def build_on_pod_config(
    *,
    run_name: str,
    model_config: LmConfig,
    learning_rate: float,
    num_train_steps: int,
    train_batch_size: int,
    seq_len: int,
    weight_decay: float,
    warmup: float,
    output_path: str,
    resources: ResourceConfig = PROTEIN_RESOURCES_H100,
    env_vars: dict[str, str] | None = None,
    wandb_name: str | None = None,
    tags: tuple[str, ...] = ("protein", "contacts-v1", "fixed-position", "coreweave"),
    wandb_group: str | None = "protein-training",
    data_seed: int = CONTACTS_V1_DATA_SEED,
    steps_per_eval: int = 20,
    max_eval_batches: int | None = 1,
    initialize_from_checkpoint_path: str | None = None,
) -> TrainLmOnPodConfig:
    """Build one concrete Levanter training config for direct Fray dispatch."""
    optimizer = AdamConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup=warmup,
        lr_schedule="cosine",
    )
    pod_config = build_train_lm_on_pod_config(
        run_name=run_name,
        model=model_config,
        optimizer=optimizer,
        data=build_data_config(),
        resources=resources,
        output_path=output_path,
        num_train_steps=num_train_steps,
        train_batch_size=train_batch_size,
        seq_len=seq_len,
        steps_per_eval=steps_per_eval,
        max_eval_batches=max_eval_batches,
        initialize_from_checkpoint_path=initialize_from_checkpoint_path,
        data_seed=data_seed,
        wandb_project="MarinFold",
        wandb_group=wandb_group,
        wandb_name=wandb_name or run_name,
        tags=tuple(tags),
        env_vars=env_vars,
    )
    return pod_config


def dispatch_training_run(
    *,
    run_name: str,
    resources: ResourceConfig = PROTEIN_RESOURCES_H100,
    env_vars: dict[str, str] | None = None,
    max_retries_failure: int = 0,
    wait: bool = True,
    **config_kwargs,
):
    """Submit a training gang as an Iris batch-band child job."""
    env_vars = {**_forwarded_perf_env(), **(env_vars or {})}
    on_pod_config = build_on_pod_config(
        run_name=run_name,
        resources=resources,
        env_vars=env_vars,
        **config_kwargs,
    )
    environment = create_environment(
        env_vars=resolve_training_env(base_env=dict(env_vars), resources=resources),
        extras=extras_for_resources(resources),
    )
    request = JobRequest(
        name=run_name,
        entrypoint=Entrypoint.from_callable(run_levanter_train_lm, args=[on_pod_config]),
        resources=resources,
        environment=environment,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=max_retries_failure,
    )
    logger.info("Dispatching exp157 training: %s -> %s", run_name, on_pod_config.output_path)
    job = current_client().submit(request)
    if wait:
        job.wait(raise_on_failure=True)
    return job

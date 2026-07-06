# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct-dispatch launcher for the exp108 Qwen3-3B contacts-v1 GPU sweep (issue #108).

Submits each training job as a ``fray.types.JobRequest`` with ``priority=3``
(iris ``PRIORITY_BAND_BATCH``) that WE control — bypassing the marin executor,
which submits child jobs with no priority band (→ interactive). #108 requires
**batch** priority for all work; the executor offers no knob for it (its
``step_runner`` / ``remote`` build ``JobRequest`` without a band and read no env),
so we dispatch the training gangs ourselves, grug-style
(cf. marin ``experiments/grug/dispatch.py``).

The training entrypoint is marin's own ``run_levanter_train_lm`` — identical to
the executor path — but the inner ``TrainLmConfig`` is assembled here at driver
top-level via marin's ``_build_train_lm_config`` (so we reuse its AdamW /
TrainerConfig / mesh / checkpointer / mp-policy assembly), with:
  * the executor ``this_output_path()`` placeholder on ``WandbConfig.replicate_path``
    replaced by a concrete S3 path, and
  * the dataset configured to **tokenize-on-the-fly** from raw S3 parquet into a
    concrete S3 cache (``auto_build_caches=True``), so nothing routes through the
    executor's ``default_tokenize`` / ``output_path_of`` (which return
    executor-resolved ``InputName`` references, not concrete paths).

VALIDATE ON THE FIRST (smoke) RUN — this bypasses the executor's serialization
and path resolution, so confirm live:
  * the resolved fray build exposes ``JobRequest.priority`` (asserted at import;
    the frozen ``0.99.dev`` build lacks it — but exp108 pins the ``0.2.x.dev`` line);
  * ``on_pod_config`` cloudpickles across the Fray boundary (same object graph the
    executor already ships — proven, but we now build it ourselves);
  * the submitted child job reports the **batch** band
    (``iris --cluster=cw-rno2a job status <child-job-id>``);
  * caches land at ``<cache_dir>/train`` and ``<cache_dir>/validation``.
"""

from __future__ import annotations

import dataclasses
import logging

from fray import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.data.text import (
    DatasetComponent,
    LmDataConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.models.lm_model import LmConfig
from marin.training.run_environment import extras_for_resources
from marin.training.training import (
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)

from contacts_v1_train_common import (
    CONTACTS_V1_DATA_SEED,
    CONTACTS_V1_S3_CORPUS_BASE,
    CONTACTS_V1_S3_PREFIX,
    CONTACTS_V1_TOKENIZER,
    PROTEIN_RESOURCES_H100,
)
from marinfold_models.defaults import _build_train_lm_config
from marinfold_models.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)

# iris PriorityBand enum value (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3).
# fray maps JobRequest.priority (int) straight to the iris band (iris_backend.py).
IRIS_PRIORITY_BAND_BATCH = 3

# Fail loudly on the frozen 0.99.dev fray, whose JobRequest has no `priority`
# field, so `priority=3` would be silently dropped → interactive band (which
# would disrupt the very interactive users batch priority protects).
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line (exp108's pins), not the frozen 0.99.dev build."
)

# Token caches live under the same MarinFold/ prefix as everything else (#108).
# With auto_build_caches=True the training workers build them on first read; the
# 3 sweep points share these caches (same paths), so only the first run tokenizes.
CONTACTS_V1_CACHE_BASE = f"{CONTACTS_V1_S3_PREFIX}/tokenized"


def build_data_config() -> LmDataConfig:
    """Concrete-path, tokenize-on-the-fly LM data config for contacts-v1.

    Replicates marin's ``step_to_lm_mixture_component`` /
    ``TokenizeConfig.as_lm_dataset_source_config`` but with CONCRETE ``cache_dir``
    strings and raw parquet source URLs, so nothing routes through the executor.
    Caches land at ``<cache_dir>/train`` and ``<cache_dir>/validation``.
    """
    train_source = UrlDatasetSourceConfig(
        train_urls=[f"{CONTACTS_V1_S3_CORPUS_BASE}/train/*.parquet"],
        validation_urls=[],
        cache_dir=f"{CONTACTS_V1_CACHE_BASE}/contacts-v1",
        format=TextLmDatasetFormat(text_key="document"),
    )
    val_source = UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[f"{CONTACTS_V1_S3_CORPUS_BASE}/val/*.parquet"],
        cache_dir=f"{CONTACTS_V1_CACHE_BASE}/contacts-v1-val",
        format=TextLmDatasetFormat(text_key="document"),
    )
    # pack=True: avoid concat-and-split (partial protein docs are nonsensical).
    # No loss mask → every token contributes (unmasked next-token loss).
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
        tokenizer=CONTACTS_V1_TOKENIZER,  # bare id: timodonnell/contacts-v1-tokenizer
        cache_dir=None,                   # each component sets its own concrete cache_dir
        auto_build_caches=True,           # build caches on the training workers
        shuffle=True,                     # full Feistel permutation (data_seed on TrainLmConfig)
        block_cross_document_attention=True,
        components={"contacts-v1": train_component, "contacts-v1-val": val_component},
        train_weights={"contacts-v1": 1.0, "contacts-v1-val": 0.0},  # val weight 0
    )


def _with_concrete_replicate_path(inner, output_path: str | None):
    """Replace the executor ``this_output_path()`` placeholder on
    ``WandbConfig.replicate_path`` with a concrete path (or None).

    ``_build_train_lm_config`` sets ONLY this placeholder (defaults.py); the
    checkpointer / HF paths are filled later inside ``run_levanter_train_lm``
    from ``TrainLmOnPodConfig.output_path``. Replace unconditionally — the
    builder always assigns the placeholder here.
    """
    tracker = dataclasses.replace(inner.trainer.tracker, replicate_path=output_path)
    trainer = dataclasses.replace(inner.trainer, tracker=tracker)
    return dataclasses.replace(inner, trainer=trainer)


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
    tags: tuple[str, ...] = ("protein", "contacts-v1", "qwen3", "3b", "unmasked", "coreweave"),
    wandb_group: str | None = "protein-training",
    data_seed: int = CONTACTS_V1_DATA_SEED,
    steps_per_eval: int = 500,
    steps_per_export: int = 5000,
) -> TrainLmOnPodConfig:
    """Build the ``TrainLmOnPodConfig`` for one run (reused by dispatch + HF export).

    Assembles the inner ``TrainLmConfig`` with marin's builder (AdamW /
    TrainerConfig / mesh / checkpointer / mp), swaps the executor placeholder on
    ``replicate_path`` for the concrete ``output_path``, and points the data at
    the tokenize-on-the-fly S3 cache.

    Pass RAW scalars (NOT ``versioned()``): ``_build_train_lm_config`` forwards
    ``learning_rate`` / ``num_train_steps`` / ``train_batch_size`` straight into
    the levanter config without unwrapping, so a ``VersionedValue`` would leak.
    """
    train_config = SimpleTrainConfig(
        resources=resources,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup=warmup,
        train_seq_len=seq_len,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        data_seed=data_seed,
        env_vars=env_vars,
    )

    # Reuse marin's optimizer (AdamW) + TrainerConfig + mesh + checkpointer + mp.
    _name, inner_config = _build_train_lm_config(
        run_name,
        build_data_config(),
        model_config,
        train_config,
        tags=tuple(tags),
        use_default_validation=False,
        eval_harness_tasks=(),
        wandb_name=wandb_name or run_name,
        wandb_group=wandb_group,
    )
    inner_config = _with_concrete_replicate_path(inner_config, output_path)

    return TrainLmOnPodConfig(
        train_config=inner_config,
        resources=resources,
        output_path=output_path,   # run_levanter_train_lm derives checkpointer/HF/run-id from this
        env_vars=env_vars,
        auto_build_caches=True,    # force levanter to build caches on the workers
    )


def dispatch_training_run(
    *,
    run_name: str,
    resources: ResourceConfig = PROTEIN_RESOURCES_H100,
    env_vars: dict[str, str] | None = None,
    max_retries_failure: int = 3,
    wait: bool = False,
    **config_kwargs,
):
    """Build the ``TrainLmOnPodConfig`` (via :func:`build_on_pod_config`) and
    submit it as a **batch-band** Fray job. Extra kwargs are forwarded to
    :func:`build_on_pod_config`.
    """
    on_pod_config = build_on_pod_config(
        run_name=run_name, resources=resources, env_vars=env_vars, **config_kwargs
    )

    environment = create_environment(
        # resolve_training_env: hardware defaults + GIT_COMMIT + JAX compile cache.
        # (The WANDB-key check inside is TPU-only; we forward WANDB_* via env_vars.)
        env_vars=resolve_training_env(base_env=dict(env_vars or {}), resources=resources),
        extras=extras_for_resources(resources),  # GpuConfig -> ["gpu"]
    )

    request = JobRequest(
        name=run_name,  # already fray/iris-safe (alnum + hyphens, no spaces)
        entrypoint=Entrypoint.from_callable(run_levanter_train_lm, args=[on_pod_config]),
        resources=resources,                 # with_gpu("H100", count=8); replicas off the ResourceConfig
        environment=environment,
        priority=IRIS_PRIORITY_BAND_BATCH,   # → iris BATCH band (the whole point)
        processes_per_task=1,                # one JAX process driving all 8 local GPUs
        max_retries_failure=max_retries_failure,
    )

    logger.info("Dispatching exp108 training (batch band): %s -> %s", run_name, on_pod_config.output_path)
    job = current_client().submit(request)
    if wait:
        job.wait(raise_on_failure=True)
    return job

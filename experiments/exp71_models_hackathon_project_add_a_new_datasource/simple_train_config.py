# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Vendored verbatim from marin/experiments/simple_train_config.py (commit
# on the protein-training-1b branch). Marin packages `experiments/` as
# part of the `marin-root` distribution, but only `marin` (not
# `marin-root`) is published on the wheel mirror that we consume here;
# so we ship a frozen copy under `marinfold_models.` for the MarinFold
# port.

import dataclasses
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.optim import OptimizerConfig
from levanter.schedule import IntSchedule


@dataclass(frozen=True)
class SimpleTrainConfig:
    resources: ResourceConfig
    train_batch_size: int | IntSchedule
    """The batch size for training. If an IntSchedule is provided, the batch size
    will be varied according to the schedule."""
    num_train_steps: int
    learning_rate: float
    train_seq_len: int | None = None
    data_seed: int | None = None
    weight_decay: float | None = None
    beta1: float | None = None
    beta2: float | None = None
    epsilon: float | None = None
    max_grad_norm: float | None = None
    warmup: float | None = None
    decay: float | None = None
    rewarmup: float | None = None
    """Re-warmup the learning rate after a decay cycle."""
    lr_schedule: str | None = None
    min_lr_ratio: float | None = None
    cycle_length: int | list[int] | None = None
    z_loss_weight: float | None = None
    ema_beta: float | None = None
    """Exponential moving average beta."""
    skip_bad_steps: bool = False
    """If True, skip steps where the loss or grad is significantly higher than the historical mean."""

    steps_per_eval: int | None = None
    """How often to run validation losses."""
    steps_per_export: int | None = None
    """How often to keep a permanent checkpoint. None (default) keeps only the
    final checkpoint; rolling temporary checkpoints are still written for
    resumption."""
    steps_per_task_eval: int | None = None
    """How often to run task evaluations."""
    steps_per_hf_export: int | None = None
    """None means match steps_per_export; -1 disables."""
    hf_generation_eos_token_ids: list[int] | None = None
    """EOS token IDs to write to generation_config.json. None means no generation config."""
    per_device_parallelism: int = -1
    """How many examples to process in parallel on each device. -1 (default) means
    train_batch_size/num_devices (no gradient accumulation). Positive to enable
    gradient accumulation."""
    per_device_eval_parallelism: int | None = None
    """Number of examples to evaluate in parallel on each device."""
    max_eval_batches: int | None = None
    """Maximum number of batches to evaluate on. None means all batches."""

    initialize_from_checkpoint_path: str | None = None
    """If set, resume from the checkpoint at this path; otherwise start from scratch."""
    initialize_from_hf: str | None = None
    """If set, start from the HF model at this path; otherwise start from scratch."""
    reset_data_loader_on_init: bool = True
    """Pairs with initialize_from_checkpoint_path. If True, reset the data loader
    so it starts from step 0; otherwise resume from the step in the checkpoint."""

    allow_partial_checkpoint: bool = False
    """Allow loading partial checkpoints (e.g. when converting training to EMA)."""

    int8: bool = False
    """Int8 (quantized) training in Levanter."""

    pad_tokenizer_to_match_model: bool = False
    """If True, pad the tokenizer's vocab to match the model's vocab size by
    adding dummy tokens. Useful when the model checkpoint has a larger vocab
    than the tokenizer (e.g., Qwen models pad their vocab to be divisible by 4
    for TPU efficiency)."""

    optimizer_config: OptimizerConfig | None = None
    """Optimizer configuration. If not set, Adam is used."""

    watch: WatchConfig = dataclasses.field(default_factory=WatchConfig)
    """Config for watching gradients, parameters, etc. Default logs norms of gradients and parameters."""

    profiler: ProfilerConfig = dataclasses.field(default_factory=ProfilerConfig)
    """JAX profiler settings for training."""

    explicit_mesh_axes: bool = False
    """If True, build the device mesh with `AxisType.Explicit` axes. Required
    for models that call `jax.sharding.reshard(..., PartitionSpec(...))`."""

    tensor_parallel_size: int = 1
    """Number of devices to use for tensor parallelism. Default 1 (no TP)."""

    env_vars: dict[str, str] | None = None
    """Environment variables to pass to the training task (e.g. WANDB_ENTITY)."""

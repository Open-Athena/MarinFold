# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
#
# A StepContext-free training-config builder for MarinFold, modelled on
# modern marin's ``marin.experiment.train.train_lm`` (`build_config` closure).
#
# marin 0.2.38 refactored its execution framework: the old executor surface
# (``ExecutorStep`` / ``this_output_path`` / ``executor_main`` /
# ``versioned`` / ``ensure_versioned`` / ...) is gone, and ``marin.experiment
# .train.train_lm`` is now the blessed training assembler. But that assembler
# dispatches its Fray job WITHOUT a priority band (→ interactive), which
# MarinFold's #108 batch-priority requirement forbids. So instead of returning
# a lazy ``ArtifactStep`` whose ``build_config(ctx)`` needs a ``StepContext``,
# this module exposes a plain function that assembles a concrete
# ``TrainLmOnPodConfig`` from ordinary arguments — the caller then submits it
# itself as a ``fray.types.JobRequest(priority=...)`` (see exp108's
# ``dispatch_train.py``).
#
# The mesh / precision / checkpointer / trainer wiring reproduces
# ``train_lm.build_config`` field-for-field; the only differences are:
#   * ``output_path`` is a CONCRETE string (used for both ``replicate_path`` and
#     ``TrainLmOnPodConfig.output_path``) instead of ``ctx.output_path``;
#   * ``data`` is a levanter ``LmDataConfig`` passed in directly instead of
#     being assembled via ``mixture(ctx, ...)``;
#   * the optimizer is the caller's responsibility (passed in), not defaulted.

import logging
from collections.abc import Sequence
from datetime import timedelta

import jmp
from haliax.partitioning import ResourceAxis
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
# levanter >= 1.2 emptied the ``levanter.data.text`` / ``levanter.optim`` package
# __init__s (their registries discover submodules lazily now), so these configs
# must be imported from their DEFINING submodule.
from levanter.data.text.datasets import LmDataConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.optim.config import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, convert_to_levanter_task_config
from marin.training.training import TrainLmOnPodConfig

logger = logging.getLogger(__name__)

# Compute in bf16, keep master params + optimizer state in f32. The universal
# marin precision policy (see marin.experiment.train.MARIN_PRECISION).
MARIN_PRECISION = "p=f32,c=bfloat16"

# The marin token axis maps onto the data-parallel mesh. Hardware plumbing, not
# an experiment choice: how the sequence axis is laid out across the pod.
_TOKEN_AXES = (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)

# Rolling resumption-checkpoint cadence (operational, not an experiment knob).
_RESUMPTION_INTERVAL = timedelta(minutes=10)


def _marin_mesh(tensor_parallel_size: int) -> MeshConfig:
    """The standard marin training mesh: data parallel, optional tensor sharding.

    ``model`` is the tensor-parallel width (1 = no sharding); ``data`` absorbs
    the rest of the pod. The token axes ride the replica/data axes the marin
    path expects. Reproduces ``marin.experiment.train._marin_mesh``.
    """
    return MeshConfig(
        axes={"replica": 1, "data": -1, "model": tensor_parallel_size},
        compute_mapping={"token": _TOKEN_AXES, "token_repeat": _TOKEN_AXES},
    )


def build_train_lm_on_pod_config(
    *,
    run_name: str,
    model: LmConfig,
    optimizer: OptimizerConfig,
    data: LmDataConfig,
    resources,
    output_path: str,
    num_train_steps: int,
    train_batch_size: int,
    seq_len: int,
    auto_build_caches: bool = False,
    steps_per_eval: int = 1000,
    max_eval_batches: int | None = None,
    steps_per_checkpoint: int | None = None,
    z_loss_weight: float | None = None,
    tensor_parallel_size: int = 1,
    data_seed: int | None = None,
    eval_harness_tasks: Sequence[EvalTaskConfig] = (),
    eval_harness_steps: int | None = None,
    initialize_from_checkpoint_path: str | None = None,
    initialize_from_hf: str | bool = False,
    pad_tokenizer_to_match_model: bool = False,
    wandb_project: str = "MarinFold",
    wandb_group: str | None = None,
    wandb_name: str | None = None,
    tags: Sequence[str] = (),
    env_vars: dict[str, str] | None = None,
    mp: str = MARIN_PRECISION,
) -> TrainLmOnPodConfig:
    """Assemble a concrete ``TrainLmOnPodConfig`` for a MarinFold training run.

    This is the StepContext-free replacement for the old vendored
    ``_build_train_lm_config`` / ``default_train`` pair. It reproduces the inner
    ``TrainLmConfig`` assembly of ``marin.experiment.train.train_lm.build_config``
    (same mesh, precision, checkpointer, trainer fields) but:

    * takes a CONCRETE ``output_path`` string — used for both
      ``WandbConfig.replicate_path`` and ``TrainLmOnPodConfig.output_path`` (the
      checkpointer base path, HF save path, and run id are all derived from the
      latter inside ``run_levanter_train_lm``), and
    * takes ``data`` (a levanter ``LmDataConfig``) directly rather than via
      ``mixture(ctx, ...)``, and
    * takes ``optimizer`` from the caller (no baked-in default).

    The returned config is a plain dataclass: the caller submits it as its own
    ``fray.types.JobRequest`` (with a batch priority band for #108) rather than
    letting marin dispatch it interactively.

    Warm-start knobs (a fine-tune continues from an existing model; #163):
    ``initialize_from_checkpoint_path`` takes a **Levanter-native** checkpoint dir
    while ``initialize_from_hf`` takes an **HF export** (repo id or a directory
    URL); set at most one. ``pad_tokenizer_to_match_model`` pads the tokenizer up
    to a wider checkpoint embedding matrix (Eric's contacts-v1 checkpoints padded
    2846 -> 2848 for TPU efficiency), so the warm-started shapes match.

    ``auto_build_caches`` must be set here, not just on ``data``:
    ``run_levanter_train_lm`` OVERRIDES ``data.auto_build_caches`` with the outer
    ``TrainLmOnPodConfig`` value, so leaving it False would silently disable
    tokenize-on-the-fly even when the data config asked for it.

    ``steps_per_checkpoint`` adds a PERMANENT checkpoint interval on top of the
    rolling resumption checkpoints — needed when a later step (HF export, eval)
    has to read a specific ``step-N``. ``None`` keeps marin's default of forced
    checkpoints only (i.e. just the final one).
    """
    harness = (
        LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(list(eval_harness_tasks)))
        if eval_harness_tasks
        else None
    )

    inner = TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            id=run_name,
            tracker=WandbConfig(
                project=wandb_project,
                name=wandb_name or run_name,
                tags=[*tags],
                group=wandb_group,
                # Mirror metrics next to the run's output so they outlive the job.
                replicate_path=output_path,
            ),
            mp=jmp.get_policy(mp),
            train_batch_size=train_batch_size,
            per_device_parallelism=-1,
            num_train_steps=num_train_steps,
            steps_per_eval=steps_per_eval,
            max_eval_batches=max_eval_batches,
            checkpointer=CheckpointerConfig(
                save_interval=_RESUMPTION_INTERVAL,
                # `keep` entries are plain dicts (draccus can't parse the
                # CheckpointInterval dataclass); CheckpointerConfig.create turns
                # them into CheckpointInterval(every=...).
                keep=[dict(every=steps_per_checkpoint)] if steps_per_checkpoint else [],
            ),
            mesh=_marin_mesh(tensor_parallel_size),
            per_device_eval_parallelism=-1,
            allow_nondivisible_batch_size=True,
        ),
        model=model,
        optimizer=optimizer,
        z_loss_weight=z_loss_weight,
        train_seq_len=seq_len,
        data_seed=data_seed,
        initialize_from_checkpoint_path=initialize_from_checkpoint_path,
        initialize_from_hf=initialize_from_hf,
        pad_tokenizer_to_match_model=pad_tokenizer_to_match_model,
        eval_harness=harness,
        eval_harness_steps=eval_harness_steps,
        adapter=NoAdaptorConfig(),
    )

    return TrainLmOnPodConfig(
        auto_build_caches=auto_build_caches,
        train_config=inner,
        resources=resources,
        output_path=output_path,
        env_vars=env_vars,
    )


__all__ = [
    "MARIN_PRECISION",
    "build_train_lm_on_pod_config",
]

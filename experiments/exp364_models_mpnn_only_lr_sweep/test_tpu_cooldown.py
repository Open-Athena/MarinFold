"""Verify the cross-platform fork preserves scientific and resume semantics."""

from dataclasses import asdict

import pytest
from fray.types import get_tpu_topology
from levanter.layers.attention import AttentionBackend
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.training.training import apply_output_path

from experiments.exp364_models_mpnn_only_lr_sweep import tpu_cooldown as tpu
from experiments.exp364_models_mpnn_only_lr_sweep.config import (
    CORPORA,
    VALIDATION_CACHE,
    VERSION,
)
from experiments.exp364_models_mpnn_only_lr_sweep.cooldown_train import (
    EPOCH_STEPS,
    SOURCE_STEP,
    build_cooldown_run,
)
from experiments.exp364_models_mpnn_only_lr_sweep.stage_tpu_inputs import (
    check_source,
    inventory_hash,
)
from experiments.exp364_models_mpnn_only_lr_sweep.test_train import resolve


def resolve_tpu(*, smoke: bool, attempt: int = 1):
    with build_context(BuildContext(VersionCodex(VERSION))):
        step = tpu.build_run(smoke=smoke, attempt=attempt)
        fingerprint = step.fingerprint()
        ctx = StepContext.for_run(
            output_path=step.path(tpu.PREFIX),
            prefix=tpu.PREFIX,
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
        pod = step.build_config(ctx)
        return apply_output_path(pod.train_config, pod.output_path), pod, fingerprint


def normalize_paths(value):
    replacements = [
        (b.cache, a.cache) for a, b in zip(CORPORA, tpu.REGIONAL_CORPORA, strict=True)
    ]
    replacements.append((tpu.VALIDATION_CACHE, VALIDATION_CACHE))
    if isinstance(value, str):
        for regional, original in replacements:
            value = value.replace(regional, original)
        return value
    if isinstance(value, dict):
        return {key: normalize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_paths(item) for item in value]
    return value


def test_tpu_matches_gpu_fork_except_hardware_and_storage() -> None:
    gpu, _ = resolve(build_cooldown_run, smoke=False, nodes=4)
    replica, pod, _ = resolve_tpu(smoke=False)
    model = asdict(replica.model)
    assert model.pop("attn_backend") == AttentionBackend.SPLASH
    original = asdict(gpu.model)
    original.pop("attn_backend")
    assert model == original
    assert asdict(replica.optimizer) == asdict(gpu.optimizer)
    assert normalize_paths(asdict(replica.data)) == asdict(gpu.data)
    for field in (
        "data_seed",
        "train_seq_len",
        "z_loss_weight",
        "hf_save_steps",
        "initialize_from_checkpoint_path",
        "initialize_model_from_checkpoint_path",
    ):
        assert getattr(replica, field) == getattr(gpu, field)
    for field in (
        "seed",
        "mp",
        "mesh",
        "train_batch_size",
        "num_train_steps",
        "steps_per_eval",
        "max_eval_batches",
        "allow_partial_checkpoint",
    ):
        assert getattr(replica.trainer, field) == getattr(gpu.trainer, field)
    assert (
        replica.trainer.initialize_from
        == f"{tpu.SOURCE_ROOT}/checkpoints/step-{SOURCE_STEP}"
    )
    assert replica.trainer.checkpointer.base_path.startswith(tpu.PREFIX)
    assert replica.trainer.checkpointer.temporary_base_path.startswith(
        "gs://marin-us-east5/"
    )
    assert replica.trainer.checkpointer.base_path != gpu.trainer.checkpointer.base_path
    assert replica.trainer.id == tpu.RUN_ID != gpu.trainer.id
    assert replica.trainer.tracker.replicate_path == pod.output_path
    assert pod.env_vars["WANDB_PROJECT"] == "MarinFold"
    assert pod.env_vars["WANDB_ENTITY"] == "open-athena"
    assert pod.resources.zone == "us-east5-a"
    assert pod.resources.regions == ["us-east5"]
    assert pod.resources.preemptible is True
    assert pod.resources.replicas == 8
    chips = get_tpu_topology(pod.resources.device.variant).chip_count
    assert chips == 32
    assert replica.trainer.per_device_parallelism * chips == 32
    assert replica.trainer.train_batch_size == 128
    assert replica.trainer.num_train_steps == EPOCH_STEPS == 186_885


def test_retries_resume_one_identity_and_smoke_is_separate() -> None:
    first, _, fingerprint = resolve_tpu(smoke=False)
    retry, _, other = resolve_tpu(smoke=False, attempt=2)
    assert fingerprint == other
    assert asdict(first) == asdict(retry)
    smoke, _, _ = resolve_tpu(smoke=True)
    assert smoke.trainer.id != first.trainer.id
    assert smoke.trainer.checkpointer.base_path != first.trainer.checkpointer.base_path
    assert smoke.trainer.num_train_steps == SOURCE_STEP + 11
    assert smoke.trainer.initialize_from == first.trainer.initialize_from


def test_transfer_inventory_rejects_missing_or_changed_objects() -> None:
    objects = [
        {"path": "data/0", "size": 10, "etag": "abc"},
        {"path": "manifest.ocdbt", "size": 3, "etag": "def"},
    ]
    artifact = {
        "name": "checkpoint",
        "objects": 2,
        "bytes": 13,
        "inventory_sha256": inventory_hash(objects),
    }
    check_source(objects, artifact)
    with pytest.raises(ValueError, match="Source inventory changed"):
        check_source(objects[:1], artifact)
    with pytest.raises(ValueError, match="Source inventory changed"):
        check_source([dict(objects[0], etag="different"), objects[1]], artifact)

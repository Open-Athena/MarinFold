"""Verify that TPU execution cannot read or overwrite the CoreWeave run."""

from dataclasses import fields

import pytest
from levanter.layers.attention import AttentionBackend
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.training.training import apply_output_path

from experiments.exp363_models_double_channel_width.config import MODEL_CONFIG, RUN_ID
from experiments.exp363_models_double_channel_width.tpu_smoke import (
    CACHE_PREFIX,
    PREFIX,
    SMOKE_STEPS,
    TARGET_ZONES,
    VERSION,
    build_run,
)


@pytest.mark.parametrize("tpu", TARGET_ZONES)
def test_tpu_smoke_is_region_local_and_preserves_model_and_batch(tpu: str) -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        step = build_run(tpu=tpu, attempt=1)
        step.fingerprint()
        ctx = StepContext.for_run(
            output_path=step.path(PREFIX), prefix=PREFIX,
            runtime_args=step.runtime_args, deps=step.deps,
        )
        pod = step.build_config(ctx)
    config = apply_output_path(pod.train_config, pod.output_path)
    assert pod.resources.zone == TARGET_ZONES[tpu]
    assert pod.resources.regions == ["us-east5"]
    assert pod.resources.preemptible
    assert pod.resources.device.variant == tpu
    assert config.trainer.id != RUN_ID
    assert f"tpu-smoke-{tpu}-a01" in config.trainer.id
    assert config.trainer.num_train_steps == SMOKE_STEPS
    assert config.trainer.train_batch_size == 128
    assert config.trainer.per_device_parallelism == 1
    assert config.trainer.max_eval_batches == 2
    assert config.trainer.checkpointer.base_path.startswith(PREFIX)
    assert config.trainer.checkpointer.temporary_base_path.startswith("gs://marin-us-east5/")
    assert config.hf_save_path.startswith(PREFIX)
    assert config.data.block_cross_document_attention
    assert not config.data.auto_build_caches
    assert all(c.cache_dir.startswith(CACHE_PREFIX) for c in config.data.components.values())
    assert "FSSPEC_S3" not in pod.env_vars
    differences = {
        f.name for f in fields(MODEL_CONFIG)
        if getattr(config.model, f.name) != getattr(MODEL_CONFIG, f.name)
    }
    assert differences == {"attn_backend"}
    assert config.model.attn_backend == AttentionBackend.SPLASH

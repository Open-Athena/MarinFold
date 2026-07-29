# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build or launch exp124 contacts-v1-think masked-loss training runs."""

import argparse
import dataclasses
import logging
import math
import os
from collections.abc import Sequence
from datetime import timedelta

import numpy as np
from fray.types import ResourceConfig, get_tpu_topology, tpu_family, tpu_hbm_capacity_bytes
from haliax import Axis
from haliax.partitioning import ResourceAxis
from huggingface_hub import snapshot_download
from iris.client.client import get_iris_ctx
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import (
    BlockShuffleConfig,
    DatasetComponent,
    DirectDatasetComponent,
    LmDataConfig,
    PackedTokenDataset,
)
from levanter.data.text.examples import GrugLmExample
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.store.cache import TreeCache
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, run_levanter_train_lm

logger = logging.getLogger(__name__)

BUCKET = os.environ.get("EXP124_BUCKET", "gs://marin-us-east5").rstrip("/")
ROOT = f"{BUCKET}/protein-structure/MarinFold"
MARIN_PREFIX = f"{ROOT}/exp124_contacts_v1_think_loss_masked"
os.environ["MARIN_PREFIX"] = MARIN_PREFIX

THINK_CACHE_ROOT = os.environ.get(
    "EXP124_THINK_CACHE_ROOT",
    f"{MARIN_PREFIX}/cache/think-masked/2026.07.29.2",
).rstrip("/")
CONTACTS_V1_VAL_CACHE = os.environ.get(
    "EXP124_STANDARD_VAL_CACHE",
    f"{BUCKET}/tokenized/contacts-v1-val/2026.07.13.1",
).rstrip("/")

CONTACTS_TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
CONTACTS_TOKENIZER_REVISION = "5d68a24a899f"
CONTACTS_TOKENIZER = f"{CONTACTS_TOKENIZER_REPO}@{CONTACTS_TOKENIZER_REVISION}"
TOKENIZER_ALLOW_PATTERNS = (
    "tokenizer*",
    "chat_template*",
    "special_tokens*",
    "added_tokens*",
    "vocab*",
    "merges*",
    "spiece*",
    "*.tiktoken",
)
ARTIFACT_VERSION = os.environ.get("EXP124_VERSION", "2026.07.29.1")
THINK_KEY = "contacts-v1-think-masked"
STANDARD_VAL_KEY = "tokenized/contacts-v1-val"
CACHE_EXEMPLAR = {
    "input_ids": np.zeros((0,), dtype=np.int32),
    "loss_weights": np.zeros((0,), dtype=np.float32),
}
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)
VOCAB_SIZE = 2845
SEQ_LEN = 8192
# Keep exp177 / exp117's successful 16-epoch-equivalent step budget so the only
# intentional training change is the think-augmented, loss-masked data cache.
EXP177_TRAIN_TOKENS = 4_676_753_425
EXP177_STEPS = round(16 * EXP177_TRAIN_TOKENS / (256 * SEQ_LEN))

TPU_TYPE = os.environ.get("EXP124_TPU", "v5p-128")
TPU_ZONE = os.environ.get("EXP124_ZONE", "us-east5-a")
TPU_SLICE_COUNT = int(os.environ.get("EXP124_TPU_SLICE_COUNT", "1"))
RESOURCES = ResourceConfig.with_tpu(
    TPU_TYPE,
    slice_count=TPU_SLICE_COUNT,
    cpu=32,
    ram="128g",
    disk="50g",
    zone=TPU_ZONE,
)
SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")
CORRECTION_FACTORS = {"v5e": 0.5, "v6e": 0.3, "v5p": 0.45, "v4": 0.45}
TOKEN_AXES = (
    ResourceAxis.REPLICA_DCN,
    ResourceAxis.REPLICA,
    ResourceAxis.DATA,
)
def _dense_transformer_bytes(batch_size: int) -> tuple[int, int]:
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    hidden_activation_bytes = batch_size * SEQ_LEN * MODEL_CONFIG.hidden_dim * 2
    attention_activation_bytes = batch_size * SEQ_LEN * MODEL_CONFIG.hidden_dim * 4 * 2
    mlp_activation_bytes = batch_size * SEQ_LEN * MODEL_CONFIG.intermediate_dim * 2
    per_layer_activation_bytes = hidden_activation_bytes + attention_activation_bytes + mlp_activation_bytes
    saved_activation_layers = max(math.floor(MODEL_CONFIG.num_layers * 0.75), 4)
    return params * 4, per_layer_activation_bytes * saved_activation_layers


def _batch_memory_bytes(batch_size: int, correction_factor: float) -> int:
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    parameter_bytes, activation_bytes = _dense_transformer_bytes(batch_size)
    optimizer_bytes = params * 8
    return math.ceil((parameter_bytes + optimizer_bytes + activation_bytes) * correction_factor)


def _correction_factor(tpu: str) -> float:
    if raw := os.environ.get("EXP124_CORRECTION_FACTOR"):
        return float(raw)
    family = tpu_family(tpu)
    if family not in CORRECTION_FACTORS:
        raise ValueError(f"No exp117 correction factor for TPU family {family!r}; set EXP124_CORRECTION_FACTOR")
    return CORRECTION_FACTORS[family]


def _batch_fit(tpu: str, batch_size: int, slice_count: int) -> tuple[int, int, int, int]:
    chips_per_slice = get_tpu_topology(tpu).chip_count
    if batch_size % slice_count != 0:
        raise ValueError(f"batch_size {batch_size} must be divisible by slice_count {slice_count}")
    examples_per_slice = batch_size // slice_count
    data_parallelism_per_slice = math.gcd(examples_per_slice, chips_per_slice)
    data_parallelism = slice_count * data_parallelism_per_slice
    tensor_parallelism = chips_per_slice // data_parallelism_per_slice
    batch_bytes = _batch_memory_bytes(batch_size, _correction_factor(tpu))
    capacity_bytes = tpu_hbm_capacity_bytes(tpu) * slice_count
    full_per_device_batch = batch_size // data_parallelism
    if batch_bytes <= capacity_bytes:
        return data_parallelism, tensor_parallelism, full_per_device_batch, 1
    for per_device_parallelism in range(full_per_device_batch, 0, -1):
        if full_per_device_batch % per_device_parallelism != 0:
            continue
        microbatch_size = per_device_parallelism * data_parallelism
        if math.ceil(batch_bytes * microbatch_size / batch_size) <= capacity_bytes:
            return data_parallelism, tensor_parallelism, per_device_parallelism, batch_size // microbatch_size
    raise ValueError(f"Batch size {batch_size} does not fit on {tpu} x {slice_count}")


def _validation_component() -> DatasetComponent:
    return DatasetComponent(cache_dir=CONTACTS_V1_VAL_CACHE, pack=True)


def _think_component() -> DatasetComponent:
    return DatasetComponent(
        cache_dir=THINK_CACHE_ROOT,
        format=PrebuiltLmDatasetFormat(input_ids_key="input_ids", loss_weights_key="loss_weights"),
        pack=True,
    )


class LazyThinkPackedDataset(AsyncDataset[GrugLmExample]):
    """Lazily construct the packed think dataset after JAX distributed init."""

    def __init__(self, split: str):
        self.split = split
        self._dataset: PackedTokenDataset | None = None

    def _inner(self) -> PackedTokenDataset:
        if self._dataset is None:
            self._dataset = PackedTokenDataset(
                TreeCache.load(f"{THINK_CACHE_ROOT}/{self.split}", CACHE_EXEMPLAR),
                Axis("position", SEQ_LEN),
                max_segments_per_example=64,
                slice_strategy="left",
                loss_weights_key="loss_weights",
                block_cross_document_attention=True,
            )
        return self._dataset

    async def async_len(self) -> int:
        return await self._inner().async_len()

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[GrugLmExample]:
        return await self._inner().get_batch(indices)


def _with_local_runtime_data(pod_config: TrainLmOnPodConfig, tokenizer_path: str) -> TrainLmOnPodConfig:
    data = pod_config.train_config.data
    runtime_data = dataclasses.replace(
        data,
        tokenizer=tokenizer_path,
        components={
            THINK_KEY: DirectDatasetComponent(
                datasets={
                    "train": LazyThinkPackedDataset("train"),
                    "validation": LazyThinkPackedDataset("validation"),
                }
            ),
            STANDARD_VAL_KEY: data.components[STANDARD_VAL_KEY],
        },
    )
    train_config = dataclasses.replace(pod_config.train_config, data=runtime_data)
    return dataclasses.replace(pod_config, train_config=train_config)


def _run_with_pinned_tokenizer(pod_config: TrainLmOnPodConfig) -> None:
    tokenizer_path = snapshot_download(
        repo_id=CONTACTS_TOKENIZER_REPO,
        revision=CONTACTS_TOKENIZER_REVISION,
        allow_patterns=list(TOKENIZER_ALLOW_PATTERNS),
    )
    run_levanter_train_lm(_with_local_runtime_data(pod_config, tokenizer_path))


def _train_job(pod_config: TrainLmOnPodConfig) -> None:
    remote(_run_with_pinned_tokenizer, resources=pod_config.resources)(pod_config)


def _optimizer() -> AdamConfig:
    return AdamConfig(
        learning_rate=3.1623e-3,
        weight_decay=0.2,
        beta1=0.9,
        beta2=0.95,
        warmup=0.1,
        lr_schedule="cosine",
        min_lr_ratio=0.1,
    )


def _run_name(steps: int, train_batch_size: int) -> str:
    default = f"exp124-cv1-think-masked-1_5b-e16-lr3p162e-3-wd0p2-bs{train_batch_size}-exp177recipe-v5p128"
    return os.environ.get("EXP124_NAME", default if steps == EXP177_STEPS else f"{default}-{steps}s")


def _identity_config(
    *,
    name: str,
    steps: int,
    steps_per_eval: int,
    train_batch_size: int,
    per_device_parallelism: int,
    gradient_accumulation: int,
    data_parallelism: int,
    tensor_parallelism: int,
    max_eval_batches: int | None,
) -> dict[str, object]:
    return {
        "name": name,
        "model": MODEL_CONFIG,
        "optimizer": _optimizer(),
        "data": {
            "think_cache_root": THINK_CACHE_ROOT,
            "standard_validation_cache": CONTACTS_V1_VAL_CACHE,
            "tokenizer": CONTACTS_TOKENIZER,
            "think_token_id": 6,
            "loss_mask": "target_<think>_tokens_zero_weight",
            "shuffle": "exp117-block-feistel",
            "mixture_block_size": 1,
            "block_cross_document_attention": True,
        },
        "trainer": {
            "train_batch_size": train_batch_size,
            "per_device_parallelism": per_device_parallelism,
            "gradient_accumulation": gradient_accumulation,
            "tpu": TPU_TYPE,
            "slice_count": TPU_SLICE_COUNT,
            "data_parallelism": data_parallelism,
            "tensor_parallelism": tensor_parallelism,
            "correction_factor": _correction_factor(TPU_TYPE),
            "num_train_steps": steps,
            "steps_per_eval": steps_per_eval,
            "max_eval_batches": max_eval_batches,
            "precision": "p=f32,c=bfloat16",
            "mesh": {"replica": 1, "data": -1, "model": tensor_parallelism},
        },
        "hf_save_steps": steps,
        "data_seed": 0,
    }


def build_step() -> ArtifactStep[LevanterCheckpoint]:
    """Build the exp124 from-scratch training run from environment variables."""
    steps = int(os.environ.get("EXP124_STEPS", str(EXP177_STEPS)))
    steps_per_eval = int(os.environ.get("EXP124_STEPS_PER_EVAL", str(max(1, EXP177_STEPS // 32))))
    train_batch_size = int(os.environ.get("EXP124_TRAIN_BATCH_SIZE", "256"))
    data_parallelism, tensor_parallelism, fit_per_device_parallelism, gradient_accumulation = _batch_fit(
        TPU_TYPE, train_batch_size, TPU_SLICE_COUNT
    )
    per_device_parallelism = int(os.environ.get("EXP124_PER_DEVICE_PARALLELISM", str(fit_per_device_parallelism)))
    max_eval_batches_env = os.environ.get("EXP124_MAX_EVAL_BATCHES")
    max_eval_batches = int(max_eval_batches_env) if max_eval_batches_env is not None else None
    name = _run_name(steps, train_batch_size)

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig | dict[str, object]:
        identity = _identity_config(
            name=name,
            steps=steps,
            steps_per_eval=steps_per_eval,
            train_batch_size=train_batch_size,
            per_device_parallelism=per_device_parallelism,
            gradient_accumulation=gradient_accumulation,
            data_parallelism=data_parallelism,
            tensor_parallelism=tensor_parallelism,
            max_eval_batches=max_eval_batches,
        )
        if ctx.is_fingerprint:
            return identity

        data = LmDataConfig(
            components={
                THINK_KEY: _think_component(),
                STANDARD_VAL_KEY: _validation_component(),
            },
            train_weights={THINK_KEY: 1.0, STANDARD_VAL_KEY: 0.0},
            tokenizer=CONTACTS_TOKENIZER,
            cache_dir=None,
            auto_build_caches=False,
            shuffle=SHUFFLE,
            mixture_block_size=1,
            block_cross_document_attention=True,
        )
        tags = [
            "protein",
            "contacts-v1",
            "contacts-v1-think",
            "exp124",
            "qwen3",
            "from-scratch",
            "loss=next_token_masked_think",
            "exp177-recipe",
        ]
        import jmp

        trainer = TrainerConfig(
            id=name,
            tracker=WandbConfig(
                entity="open-athena",
                project="MarinFold",
                name=name,
                tags=tags,
                group="exp124-think-loss-masked",
                replicate_path=ctx.output_path,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=train_batch_size,
            per_device_parallelism=per_device_parallelism,
            num_train_steps=steps,
            steps_per_eval=steps_per_eval,
            checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10), keep=[{"every": steps}]),
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": tensor_parallelism},
                compute_mapping={"token": TOKEN_AXES, "token_repeat": TOKEN_AXES},
            ),
            per_device_eval_parallelism=-1,
            max_eval_batches=max_eval_batches,
            allow_nondivisible_batch_size=True,
        )
        train_config = TrainLmConfig(
            data=data,
            trainer=trainer,
            model=MODEL_CONFIG,
            optimizer=_optimizer(),
            z_loss_weight=0.0,
            train_seq_len=SEQ_LEN,
            hf_save_steps=steps,
            data_seed=0,
            adapter=NoAdaptorConfig(),
        )
        env_vars = {
            "WANDB_ENTITY": "open-athena",
            "WANDB_PROJECT": "MarinFold",
            "EXP124_TPU": TPU_TYPE,
            "EXP124_TPU_SLICE_COUNT": str(TPU_SLICE_COUNT),
            "EXP124_DATA_PARALLELISM": str(data_parallelism),
            "EXP124_TENSOR_PARALLELISM": str(tensor_parallelism),
            "EXP124_PER_DEVICE_PARALLELISM": str(per_device_parallelism),
            "EXP124_GRADIENT_ACCUMULATION": str(gradient_accumulation),
        }
        for key in ("WANDB_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
            if value := os.environ.get(key):
                env_vars[key] = value
        return TrainLmOnPodConfig(
            train_config=train_config,
            resources=ctx.runtime_arg("train_resources"),
            output_path=ctx.output_path,
            env_vars=env_vars,
            auto_build_caches=False,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"checkpoints/{name}", ARTIFACT_VERSION),
        version=ARTIFACT_VERSION,
        artifact_type=LevanterCheckpoint,
        run=_train_job,
        build_config=build_config,
        deps=(),
        runtime_args={"train_resources": RESOURCES},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Submit the lowered graph. Without this flag, print the plan.")
    args = parser.parse_args(argv)
    lowered = lower(build_step())
    if not args.run:
        print(lowered)
        return 0
    if get_iris_ctx() is None:
        parser.error("--run must execute inside an Iris coordinator job")
    StepRunner().run([lowered])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

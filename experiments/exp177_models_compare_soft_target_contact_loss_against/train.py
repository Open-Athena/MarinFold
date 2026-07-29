# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build or launch exp177 contacts-v1 loss head-to-head training runs."""

import argparse
import dataclasses
import logging
import os
from datetime import timedelta
from enum import StrEnum

import jmp
from fray.types import ResourceConfig
from haliax import Axis
from haliax.partitioning import ResourceAxis, round_axis_for_partitioning
from huggingface_hub import snapshot_download
from iris.client.client import get_iris_ctx
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, DirectDatasetComponent, LmDataConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmHeadModel
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import Trainer, TrainerConfig, initialize as initialize_trainer
from levanter.utils.jax_utils import parameter_count
from levanter.utils.mesh import MeshConfig
import jax.random as jrandom
import levanter.eval
import levanter.tracker
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    _apply_env_to_process,
    _prepare_training_run,
    run_levanter_train_lm,
)

from marinfold_models.document_loss import LevanterDocumentBatch, document_loss
from premade_contacts_dataset import FixedQuotaPremadeContactsDataset, FixedQuotaSoftTargetContactsDataset

logger = logging.getLogger(__name__)


class LossKind(StrEnum):
    """Training objective for an exp177 arm."""

    NEXT_TOKEN = "next_token"
    SOFT_TARGET = "soft_target"


# Existing GCS data. The analyzed-contact shards are already staged in GCS; no
# experiment launch should copy data as part of training.
BUCKET = os.environ.get("EXP177_BUCKET", "gs://marin-us-east5").rstrip("/")
ROOT = f"{BUCKET}/protein-structure/MarinFold"
MARIN_PREFIX = f"{ROOT}/exp177_soft_target_loss_h2h"
os.environ["MARIN_PREFIX"] = MARIN_PREFIX

CONTACTS_PREFIX = os.environ.get(
    "EXP177_CONTACTS_PREFIX",
    f"{ROOT}/exp147_on_the_fly_contacts_v1_pilot/pilot_data/contacts",
).rstrip("/")
# Reuse the existing exp117-compatible tokenized contacts-v1 caches for the
# stock CE arm. These live at the historical Marin root prefix because exp117
# was run from marin before MarinFold tightened its artifact-prefix policy.
CONTACTS_V1_TRAIN_CACHE = os.environ.get(
    "EXP177_TRAIN_CACHE",
    f"{BUCKET}/tokenized/contacts-v1/2026.07.13.1",
).rstrip("/")
CONTACTS_V1_VAL_CACHE = os.environ.get(
    "EXP177_VAL_CACHE",
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
ARTIFACT_VERSION = os.environ.get("EXP177_VERSION", "2026.07.20.1")
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
TRAIN_TOKENS = 4_676_753_425
EXP117_STEPS = round(16 * TRAIN_TOKENS / (256 * SEQ_LEN))

TPU_TYPE = os.environ.get("EXP177_TPU", "v5p-8")
TPU_ZONE = os.environ.get("EXP177_ZONE", "us-east5-a")
RESOURCES = ResourceConfig.with_tpu(
    TPU_TYPE,
    slice_count=1,
    cpu=32,
    ram="128g",
    disk="50g",
    zone=TPU_ZONE,
)

TOKEN_AXES = (
    ResourceAxis.REPLICA_DCN,
    ResourceAxis.REPLICA,
    ResourceAxis.DATA,
)


def _loss_kind() -> LossKind:
    return LossKind(os.environ.get("EXP177_LOSS", LossKind.NEXT_TOKEN.value))


def _train_cache_component() -> DatasetComponent:
    return DatasetComponent(cache_dir=CONTACTS_V1_TRAIN_CACHE, pack=True)


def _validation_component() -> DatasetComponent:
    return DatasetComponent(cache_dir=CONTACTS_V1_VAL_CACHE, pack=True)


def _with_local_tokenizer(pod_config: TrainLmOnPodConfig, tokenizer_path: str) -> TrainLmOnPodConfig:
    train_config = dataclasses.replace(
        pod_config.train_config,
        data=dataclasses.replace(pod_config.train_config.data, tokenizer=tokenizer_path),
    )
    return dataclasses.replace(pod_config, train_config=train_config)


def _run_next_token_with_pinned_tokenizer(pod_config: TrainLmOnPodConfig) -> None:
    tokenizer_path = snapshot_download(
        repo_id=CONTACTS_TOKENIZER_REPO,
        revision=CONTACTS_TOKENIZER_REVISION,
        allow_patterns=list(TOKENIZER_ALLOW_PATTERNS),
    )
    run_levanter_train_lm(_with_local_tokenizer(pod_config, tokenizer_path))


def _run_soft_target_with_pinned_tokenizer(pod_config: TrainLmOnPodConfig) -> None:
    tokenizer_path = snapshot_download(
        repo_id=CONTACTS_TOKENIZER_REPO,
        revision=CONTACTS_TOKENIZER_REVISION,
        allow_patterns=list(TOKENIZER_ALLOW_PATTERNS),
    )
    pod_config = _with_local_tokenizer(pod_config, tokenizer_path)
    _, train_config, env = _prepare_training_run(pod_config)
    _apply_env_to_process(env)
    _run_soft_target_train_lm(train_config)


def _train_job(pod_config: TrainLmOnPodConfig) -> None:
    loss_kind = LossKind(pod_config.env_vars["EXP177_LOSS"])
    target = _run_soft_target_with_pinned_tokenizer if loss_kind == LossKind.SOFT_TARGET else _run_next_token_with_pinned_tokenizer
    remote(target, resources=pod_config.resources)(pod_config)


def _soft_loss(model: LmHeadModel, batch: LevanterDocumentBatch, *, key=None):
    return document_loss(model, batch, key=key), {}


def _run_soft_target_train_lm(config: TrainLmConfig) -> None:
    """Minimal Levanter train loop with PR #144 document loss.

    This intentionally starts from scratch: it builds ``config.model`` directly
    and never sets ``initialize_from_hf`` or ``initialize_model_from_checkpoint``.
    """
    tokenizer = config.data.the_tokenizer
    initialize_trainer(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    with Trainer(config.trainer, optimizer, _soft_loss) as trainer:
        seed = config.trainer.seed
        data_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 3)
        if config.data_seed is not None:
            data_key = jrandom.PRNGKey(config.data_seed)

        train_length = config.train_seq_len or config.model.max_seq_len
        Pos = config.model.max_Pos.resize(train_length)
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), trainer.parameter_axis_mapping)
        train_dataset = config.data.train_set(Pos, config.trainer.batch_schedule, key=data_key)
        tagged_eval_datasets = config.data.tagged_eval_sets(Pos)
        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))
        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        max_eval_examples = config.trainer.max_eval_batches
        if max_eval_examples is not None:
            max_eval_examples *= config.trainer.eval_batch_size
        if tagged_eval_datasets:
            checkpoint_path = config.trainer.checkpointer.expanded_path(trainer.run_id)
            trainer.add_hook(
                levanter.eval.cb_tagged_lm_evaluate(
                    config.trainer.EvalBatch,
                    tagged_eval_datasets,
                    tokenizer,
                    trainer.device_mesh,
                    trainer.compute_axis_mapping,
                    max_eval_examples,
                    mp=config.trainer.mp,
                    checkpoint_path=checkpoint_path,
                ),
                every=config.trainer.steps_per_eval,
            )

        train_loader = trainer.data_loader(train_dataset)
        trainer.train(state, train_loader)


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


def _run_name(loss_kind: LossKind, steps: int, train_batch_size: int) -> str:
    default = f"exp177-cv1-1_5b-e16-lr3p162e-3-wd0p2-bs{train_batch_size}-{loss_kind}-tpu"
    return os.environ.get("EXP177_NAME", default if steps == EXP117_STEPS else f"{default}-{steps}s")


def _identity_config(
    ctx: StepContext,
    *,
    loss_kind: LossKind,
    name: str,
    steps: int,
    steps_per_eval: int,
    train_batch_size: int,
    per_device_parallelism: int,
    max_eval_batches: int | None,
    num_shards: int,
    examples_per_shard: int,
) -> dict[str, object]:
    return {
        "name": name,
        "loss_kind": loss_kind.value,
        "model": MODEL_CONFIG,
        "optimizer": _optimizer(),
        "data": {
            "contacts_prefix": CONTACTS_PREFIX,
            "num_shards": num_shards,
            "total_shards": 3338,
            "examples_per_shard": examples_per_shard,
            "seed": 0,
            "max_seq_len": SEQ_LEN,
            "train_cache": CONTACTS_V1_TRAIN_CACHE,
            "validation_cache": CONTACTS_V1_VAL_CACHE,
            "tokenizer": CONTACTS_TOKENIZER,
            "shuffle": False,
            "mixture_block_size": 1,
            "block_cross_document_attention": True,
        },
        "trainer": {
            "train_batch_size": train_batch_size,
            "per_device_parallelism": per_device_parallelism,
            "num_train_steps": steps,
            "steps_per_eval": steps_per_eval,
            "max_eval_batches": max_eval_batches,
            "precision": "p=f32,c=bfloat16",
            "mesh": {"replica": 1, "data": -1, "model": 1},
        },
        "hf_save_steps": steps,
        "data_seed": 0,
    }


def build_step() -> ArtifactStep[LevanterCheckpoint]:
    """Build one exp177 arm from environment variables."""
    loss_kind = _loss_kind()
    steps = int(os.environ.get("EXP177_STEPS", str(EXP117_STEPS)))
    steps_per_eval = int(os.environ.get("EXP177_STEPS_PER_EVAL", str(max(1, EXP117_STEPS // 32))))
    train_batch_size = int(os.environ.get("EXP177_TRAIN_BATCH_SIZE", "256"))
    per_device_parallelism = int(os.environ.get("EXP177_PER_DEVICE_PARALLELISM", "16"))
    max_eval_batches_env = os.environ.get("EXP177_MAX_EVAL_BATCHES")
    max_eval_batches = int(max_eval_batches_env) if max_eval_batches_env is not None else None
    num_shards = int(os.environ.get("EXP177_NUM_SHARDS", "3338"))
    examples_per_shard = int(os.environ.get("EXP177_EXAMPLES_PER_SHARD", "2650"))
    name = _run_name(loss_kind, steps, train_batch_size)

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig | dict[str, object]:
        identity = _identity_config(
            ctx,
            loss_kind=loss_kind,
            name=name,
            steps=steps,
            steps_per_eval=steps_per_eval,
            train_batch_size=train_batch_size,
            per_device_parallelism=per_device_parallelism,
            max_eval_batches=max_eval_batches,
            num_shards=num_shards,
            examples_per_shard=examples_per_shard,
        )
        if ctx.is_fingerprint:
            return identity

        train_key = f"contacts-v1/{loss_kind.value}"
        val_key = "tokenized/contacts-v1-val"
        if loss_kind == LossKind.SOFT_TARGET:
            train_dataset = FixedQuotaSoftTargetContactsDataset(
                data_prefix=CONTACTS_PREFIX,
                num_shards=num_shards,
                examples_per_shard=examples_per_shard,
                seed=0,
                max_seq_len=SEQ_LEN,
            )
            train_component = DirectDatasetComponent(datasets={"train": train_dataset})
        else:
            train_component = _train_cache_component()
        data = LmDataConfig(
            components={
                train_key: train_component,
                val_key: _validation_component(),
            },
            train_weights={train_key: 1.0, val_key: 0.0},
            tokenizer=CONTACTS_TOKENIZER,
            cache_dir=None,
            auto_build_caches=False,
            shuffle=False,
            mixture_block_size=1,
            block_cross_document_attention=True,
        )
        tags = [
            "protein",
            "contacts-v1",
            "exp177",
            "qwen3",
            "from-scratch",
            f"loss={loss_kind.value}",
            "exp117-recipe",
        ]
        trainer = TrainerConfig(
            id=name,
            tracker=WandbConfig(
                entity="open-athena",
                project="MarinFold",
                name=name,
                tags=tags,
                group="exp177-soft-target-loss-h2h",
                replicate_path=ctx.output_path,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=train_batch_size,
            per_device_parallelism=per_device_parallelism,
            num_train_steps=steps,
            steps_per_eval=steps_per_eval,
            checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10), keep=[{"every": steps}]),
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": 1},
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
            "EXP177_LOSS": loss_kind.value,
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

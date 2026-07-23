# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build or launch the exp147 on-the-fly contacts-v1 TPU pilot."""

import argparse
import dataclasses
import os
from datetime import timedelta

import jmp
from fray.types import ResourceConfig
from haliax.partitioning import ResourceAxis
from huggingface_hub import snapshot_download
from iris.client.client import get_iris_ctx
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import (
    DatasetComponent,
    DirectDatasetComponent,
    LmDataConfig,
)
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.data import tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    run_levanter_train_lm,
)

from premade_contacts_dataset import FixedQuotaPremadeContactsDataset

BUCKET = os.environ.get("EXP147_BUCKET", "gs://marin-us-east5").rstrip("/")
ROOT = f"{BUCKET}/protein-structure/MarinFold"
MARIN_PREFIX = f"{ROOT}/exp147_on_the_fly_contacts_v1_pilot"
os.environ["MARIN_PREFIX"] = MARIN_PREFIX

PILOT_DATA_PREFIX = f"{MARIN_PREFIX}/pilot_data"
PILOT_CONTACTS_PREFIX = f"{PILOT_DATA_PREFIX}/contacts"
CONTACTS_V1_VAL_GLOB = f"{ROOT}/exp53_contacts_v1_5x/documents/val/*.parquet"

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
ARTIFACT_VERSION = os.environ.get("EXP147_VERSION", "exp147-dev")
VALIDATION_VERSION = "2026.07.23"

MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

TPU_TYPE = os.environ.get("EXP147_TPU", "v6e-8")
TPU_ZONE = os.environ.get("EXP147_ZONE", "us-east5-b")
RESOURCES = ResourceConfig.with_tpu(
    TPU_TYPE,
    slice_count=1,
    cpu=32,
    ram="128g",
    disk="50g",
    zone=TPU_ZONE,
)

TAGS = (
    "protein",
    "contacts-v1",
    "on-the-fly",
    "esm-atlas",
    "qwen3",
    "unmasked",
    "exp147",
    "pilot",
    "exp117-reference",
)
TOKEN_AXES = (
    ResourceAxis.REPLICA_DCN,
    ResourceAxis.REPLICA,
    ResourceAxis.DATA,
)


def _validation_step() -> ArtifactStep[TokenizedCache]:
    return tokenized(
        name="tokenized/contacts-v1-val",
        paths=[CONTACTS_V1_VAL_GLOB],
        tokenizer=CONTACTS_TOKENIZER_REPO,
        version=VALIDATION_VERSION,
        validation=True,
        text_key="document",
        resources=ResourceConfig.with_cpu(
            cpu=4,
            ram="16g",
            disk="10g",
            zone=TPU_ZONE,
        ),
    )


def _validation_component(cache: TokenizedCache) -> DatasetComponent:
    return dataclasses.replace(cache.as_component(), pack=True)


def _with_local_tokenizer(
    pod_config: TrainLmOnPodConfig, tokenizer_path: str
) -> TrainLmOnPodConfig:
    train_config = dataclasses.replace(
        pod_config.train_config,
        data=dataclasses.replace(
            pod_config.train_config.data,
            tokenizer=tokenizer_path,
        ),
    )
    return dataclasses.replace(pod_config, train_config=train_config)


def _run_with_pinned_tokenizer(pod_config: TrainLmOnPodConfig) -> None:
    """Stage the pinned tokenizer on the TPU worker before starting Levanter."""
    tokenizer_path = snapshot_download(
        repo_id=CONTACTS_TOKENIZER_REPO,
        revision=CONTACTS_TOKENIZER_REVISION,
        allow_patterns=list(TOKENIZER_ALLOW_PATTERNS),
    )
    run_levanter_train_lm(_with_local_tokenizer(pod_config, tokenizer_path))


def _train_job(pod_config: TrainLmOnPodConfig) -> None:
    remote(_run_with_pinned_tokenizer, resources=pod_config.resources)(pod_config)


def _identity_config(
    ctx: StepContext,
    validation: ArtifactStep[TokenizedCache],
    *,
    name: str,
    steps: int,
    steps_per_eval: int,
    train_batch_size: int,
    per_device_parallelism: int,
    max_eval_batches: int | None,
    num_shards: int,
    examples_per_shard: int,
) -> dict[str, object]:
    """Return the stable experiment decisions used for artifact fingerprinting."""
    return {
        "name": name,
        "model": MODEL_CONFIG,
        "optimizer": AdamConfig(
            learning_rate=3.1623e-3,
            weight_decay=0.2,
            beta1=0.9,
            beta2=0.95,
            warmup=0.1,
            lr_schedule="cosine",
            min_lr_ratio=0.1,
        ),
        "data": {
            "kind": "fixed-quota-premade-contacts-v1",
            "prefix": PILOT_CONTACTS_PREFIX,
            "num_shards": num_shards,
            "total_shards": 3338,
            "examples_per_shard": examples_per_shard,
            "seed": 0,
            "max_seq_len": 8192,
            "validation": ctx.artifact_path(validation),
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
        "wandb": {
            "entity": "open-athena",
            "project": "MarinFold",
            "group": "exp147-on-the-fly-contacts-v1",
            "name": name,
            "tags": TAGS,
        },
        "hf_save_steps": steps,
        "data_seed": 0,
    }


def build_step() -> ArtifactStep[LevanterCheckpoint]:
    """Build the lazy training artifact without submitting it."""
    validation = _validation_step()
    steps = int(os.environ.get("EXP147_STEPS", "200"))
    steps_per_eval = int(os.environ.get("EXP147_STEPS_PER_EVAL", "100"))
    train_batch_size = int(os.environ.get("EXP147_TRAIN_BATCH_SIZE", "256"))
    per_device_parallelism = int(
        os.environ.get("EXP147_PER_DEVICE_PARALLELISM", "16")
    )
    max_eval_batches_env = os.environ.get("EXP147_MAX_EVAL_BATCHES")
    max_eval_batches = (
        int(max_eval_batches_env) if max_eval_batches_env is not None else None
    )
    num_shards = int(os.environ.get("EXP147_NUM_SHARDS", "16"))
    examples_per_shard = int(
        os.environ.get("EXP147_EXAMPLES_PER_SHARD", "2650")
    )
    name = os.environ.get(
        "EXP147_NAME",
        f"exp147-otf-contacts-v1-1_5b-pilot-{steps}s-bs{train_batch_size}-v6e8",
    )

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig | dict[str, object]:
        identity = _identity_config(
            ctx,
            validation,
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

        train_dataset = FixedQuotaPremadeContactsDataset(
            data_prefix=PILOT_CONTACTS_PREFIX,
            num_shards=num_shards,
            examples_per_shard=examples_per_shard,
            seed=0,
            max_seq_len=8192,
        )
        train_key = "on-the-fly/esm-atlas-contacts-v1"
        val_key = "tokenized/contacts-v1-val"
        data = LmDataConfig(
            components={
                train_key: DirectDatasetComponent(datasets={"train": train_dataset}),
                val_key: _validation_component(ctx.resolved(validation)),
            },
            train_weights={train_key: 1.0, val_key: 0.0},
            tokenizer=CONTACTS_TOKENIZER,
            cache_dir=None,
            auto_build_caches=False,
            shuffle=False,
            mixture_block_size=1,
            block_cross_document_attention=True,
        )
        trainer = TrainerConfig(
            id=name,
            tracker=WandbConfig(
                entity="open-athena",
                project="MarinFold",
                name=name,
                tags=list(TAGS),
                group="exp147-on-the-fly-contacts-v1",
                replicate_path=ctx.output_path,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=train_batch_size,
            per_device_parallelism=per_device_parallelism,
            num_train_steps=steps,
            steps_per_eval=steps_per_eval,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[{"every": steps}],
            ),
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": 1},
                compute_mapping={
                    "token": TOKEN_AXES,
                    "token_repeat": TOKEN_AXES,
                },
            ),
            per_device_eval_parallelism=-1,
            max_eval_batches=max_eval_batches,
            allow_nondivisible_batch_size=True,
        )
        train_config = TrainLmConfig(
            data=data,
            trainer=trainer,
            model=MODEL_CONFIG,
            optimizer=identity["optimizer"],
            z_loss_weight=0.0,
            train_seq_len=8192,
            hf_save_steps=steps,
            data_seed=0,
            adapter=NoAdaptorConfig(),
        )
        env_vars = {"WANDB_ENTITY": "open-athena", "WANDB_PROJECT": "MarinFold"}
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            env_vars["WANDB_API_KEY"] = wandb_api_key
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
        deps=(validation,),
        runtime_args={"train_resources": RESOURCES},
    )


def build_steps() -> list[ArtifactStep[LevanterCheckpoint]]:
    """Build the single-step launch list used by tests and the CLI."""
    return [build_step()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Submit the lowered graph. Without this flag, only print the plan.",
    )
    args = parser.parse_args(argv)
    lowered = lower(build_step())
    if not args.run:
        print(lowered)
        return 0
    if get_iris_ctx() is None:
        parser.error(
            "--run must execute inside an Iris coordinator job; use the launch "
            "command in this experiment's README"
        )
    StepRunner().run([lowered])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

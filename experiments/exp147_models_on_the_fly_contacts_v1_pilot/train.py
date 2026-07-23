# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the exp147 on-the-fly contacts-v1 TPU pilot."""

import dataclasses
import os

from fray import ResourceConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat
from levanter.data.text.datasets import DirectDatasetComponent
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution import executor_main, versioned
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

from marinfold_models.defaults import default_tokenize, default_train
from marinfold_models.simple_train_config import SimpleTrainConfig

from premade_contacts_dataset import StreamingPremadeContactsDataset


BUCKET = os.environ.get("EXP147_BUCKET", "gs://marin-us-east5").rstrip("/")
ROOT = f"{BUCKET}/protein-structure/MarinFold"
MARIN_PREFIX = f"{ROOT}/exp147_on_the_fly_contacts_v1_pilot"
os.environ["MARIN_PREFIX"] = MARIN_PREFIX

PILOT_DATA_PREFIX = f"{MARIN_PREFIX}/pilot_data"
PILOT_CONTACTS_PREFIX = f"{PILOT_DATA_PREFIX}/contacts"
CONTACTS_V1_VAL_GLOB = f"{ROOT}/exp53_contacts_v1_5x/documents/val/*.parquet"

CONTACTS_TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
CONTACTS_TOKENIZER_REVISION = "5d68a24a899f"

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
TPU_ZONE = os.environ.get("EXP147_ZONE", "us-east5-a")
RESOURCES = ResourceConfig.with_tpu(
    TPU_TYPE,
    slice_count=1,
    cpu=32,
    ram="128g",
    disk="50g",
    zone=TPU_ZONE,
)

CONTACTS_V1_VAL_TOK = default_tokenize(
    name="contacts-v1-val",
    dataset=CONTACTS_V1_VAL_GLOB,
    tokenizer=CONTACTS_TOKENIZER_REPO,
    format=TextLmDatasetFormat(text_key="document"),
    is_validation=True,
)


def _validation_component() -> DatasetComponent:
    component = step_to_lm_mixture_component(
        CONTACTS_V1_VAL_TOK,
        include_raw_paths=True,
    )
    return dataclasses.replace(component, pack=True)


def build_steps() -> list:
    steps = int(os.environ.get("EXP147_STEPS", "200"))
    steps_per_eval = int(os.environ.get("EXP147_STEPS_PER_EVAL", "100"))
    max_eval_batches_env = os.environ.get("EXP147_MAX_EVAL_BATCHES")
    max_eval_batches = (
        int(max_eval_batches_env) if max_eval_batches_env is not None else None
    )
    name = os.environ.get(
        "EXP147_NAME",
        f"exp147-otf-contacts-v1-1_5b-pilot-{steps}s-v6e8",
    )

    train_dataset = StreamingPremadeContactsDataset(
        data_prefix=PILOT_CONTACTS_PREFIX,
        num_shards=int(os.environ.get("EXP147_NUM_SHARDS", "16")),
        seed=0,
        max_seq_len=8192,
        global_batch_size=128,
    )
    train_key = "on-the-fly/esm-atlas-contacts-v1"
    val_key = "tokenized/contacts-v1-val"
    data = LmDataConfig(
        components={
            train_key: DirectDatasetComponent(datasets={"train": train_dataset}),
            val_key: _validation_component(),
        },
        train_weights={train_key: 1.0, val_key: 0.0},
        tokenizer=CONTACTS_TOKENIZER_REPO,
        cache_dir=None,
        auto_build_caches=False,
        # The streaming dataset owns shard/row shuffling and ignores external
        # index values. These two settings keep the single-component mixture
        # from reordering those values before the stream sees them.
        shuffle=False,
        mixture_block_size=1,
        block_cross_document_attention=True,
    )

    env_vars = {"WANDB_ENTITY": "open-athena"}
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        env_vars["WANDB_API_KEY"] = wandb_api_key
    train_config = SimpleTrainConfig(
        resources=RESOURCES,
        train_batch_size=128,
        num_train_steps=versioned(steps),
        learning_rate=versioned(3.1623e-3),
        lr_schedule=versioned("cosine"),
        min_lr_ratio=0.1,
        weight_decay=0.2,
        warmup=0.1,
        beta1=0.9,
        beta2=0.95,
        train_seq_len=8192,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps,
        max_eval_batches=max_eval_batches,
        data_seed=versioned(0),
        env_vars=env_vars,
    )
    train_step = default_train(
        name=name,
        tokenized=data,
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=[
            "protein",
            "contacts-v1",
            "on-the-fly",
            "esm-atlas",
            "qwen3",
            "unmasked",
            "exp147",
            "pilot",
        ],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group="exp147-on-the-fly-contacts-v1",
        wandb_name=name,
    )
    return [train_step]


if __name__ == "__main__":
    executor_main(steps=build_steps())

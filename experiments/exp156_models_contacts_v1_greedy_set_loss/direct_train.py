# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run exp156 training directly in the allocated Iris GPU task.

The experiment's normal ``train.py --run`` entrypoint is an artifact coordinator
that submits a nested accelerator task. This entrypoint instead builds the same
Levanter configuration in the already-allocated GPU task. It is required for
multi-node CoreWeave runs and makes the spawn-based telemetry writer safe.
"""

import dataclasses
import os
from datetime import timedelta

import jmp
from fray.types import ResourceConfig
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.distributed import DistributedConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import TrainLmOnPodConfig

import train as exp156


def _component(cache_path: str) -> DatasetComponent:
    """Load a legacy mirrored cache without artifact metadata."""
    return dataclasses.replace(TokenizedCache.raw_load(cache_path).as_component(), pack=False)


def main() -> None:
    """Configure and run one chosen exp156 objective directly on GPUs."""
    name = os.environ["EXP156_NAME"]
    loss_kind = os.environ["EXP156_LOSS_KIND"]
    version = os.environ.get("EXP156_VERSION", "exp156-direct")
    steps = int(os.environ.get("EXP156_STEPS", "300"))
    steps_per_eval = int(os.environ.get("EXP156_STEPS_PER_EVAL", "50"))
    train_batch_size = int(os.environ.get("EXP156_TRAIN_BATCH_SIZE", "16"))
    per_device_parallelism = int(os.environ.get("EXP156_PER_DEVICE_PARALLELISM", "1"))
    max_eval_batches = int(os.environ.get("EXP156_MAX_EVAL_BATCHES", "16"))

    train_key = "tokenized/contacts-v1-train"
    val_key = "tokenized/contacts-v1-val"
    data = LmDataConfig(
        components={
            train_key: _component(os.environ["EXP156_TOKENIZED_TRAIN_CACHE"]),
            val_key: _component(os.environ["EXP156_TOKENIZED_VAL_CACHE"]),
        },
        train_weights={train_key: 1.0, val_key: 0.0},
        tokenizer=exp156.CONTACTS_TOKENIZER,
        cache_dir=None,
        auto_build_caches=False,
        shuffle=True,
        mixture_block_size=1,
        block_cross_document_attention=True,
    )
    save_interval = None if os.environ.get("EXP156_DISABLE_TEMP_CHECKPOINTS") == "1" else timedelta(minutes=10)
    trainer = TrainerConfig(
        id=name,
        tracker=WandbConfig(
            entity="open-athena",
            project="MarinFold",
            name=name,
            tags=list(exp156.TAGS),
            group="exp156-contacts-v1-greedy-set-loss",
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=train_batch_size,
        per_device_parallelism=per_device_parallelism,
        num_train_steps=steps,
        steps_per_eval=steps_per_eval,
        checkpointer=CheckpointerConfig(save_interval=save_interval, keep=[{"every": steps}]),
        mesh=MeshConfig(
            axes={"replica": 1, "data": -1, "model": 1},
            compute_mapping={"token": exp156.TOKEN_AXES, "token_repeat": exp156.TOKEN_AXES},
        ),
        per_device_eval_parallelism=-1,
        max_eval_batches=max_eval_batches,
        allow_nondivisible_batch_size=True,
        distributed=DistributedConfig(initialize_jax_distributed=False),
    )
    train_config = TrainLmConfig(
        data=data,
        trainer=trainer,
        model=exp156.MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=float(os.environ.get("EXP156_LEARNING_RATE", "3.1623e-3")),
            weight_decay=0.2,
            beta1=0.9,
            beta2=0.95,
            warmup=0.1,
            lr_schedule="cosine",
            min_lr_ratio=0.1,
        ),
        z_loss_weight=0.0,
        train_seq_len=8192,
        hf_save_steps=steps,
        data_seed=0,
        adapter=NoAdaptorConfig(),
        initialize_from_hf=exp156.INITIALIZE_FROM_HF,
        pad_tokenizer_to_match_model=True,
    )
    if not isinstance(train_config.model, HFCompatConfig):
        raise TypeError("expected HF-compatible model config")

    resources = ResourceConfig.with_gpu(
        os.environ.get("EXP156_GPU", "H100"),
        count=int(os.environ.get("EXP156_GPU_COUNT", "1")),
        cpu=float(os.environ.get("EXP156_CPU", "16")),
        ram=os.environ.get("EXP156_RAM", "128g"),
        disk=os.environ.get("EXP156_DISK", "200g"),
    )
    env_vars = {"WANDB_ENTITY": "open-athena", "WANDB_PROJECT": "MarinFold"}
    for env_name in ("WANDB_API_KEY", "WANDB_MODE"):
        if env_value := os.environ.get(env_name):
            env_vars[env_name] = env_value
    pod_config = TrainLmOnPodConfig(
        train_config=train_config,
        resources=resources,
        output_path=f"{exp156.MARIN_PREFIX}/users/zack/checkpoints/{name}/{version}",
        env_vars=env_vars,
        auto_build_caches=False,
    )
    if loss_kind == "next-token":
        exp156._run_stock_next_token_train_with_telemetry(pod_config)
    elif loss_kind == "greedy-set":
        exp156._run_greedy_set_train_with_telemetry(pod_config)
    else:
        raise ValueError(f"unsupported EXP156_LOSS_KIND={loss_kind!r}")


if __name__ == "__main__":
    main()

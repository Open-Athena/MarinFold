# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Train delta-stream V2 for one finite pass over the full exp277 corpus."""

import dataclasses
import logging
import os
from datetime import timedelta
from pathlib import Path

import jmp
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import (
    BlockShuffleConfig,
    ConcatDatasetComponent,
    DatasetComponent,
    LmDataConfig,
)
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.run_environment import extras_for_resources
from marin.training.training import (
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)

from convert_exp277_caches_to_delta_stream import VOCAB_SIZE
from delta_stream_data import PackableTokenIdsFormat
from dispatch_delta_stream_full import MODEL_CONFIG, _mesh
from finite_delta_data import FULL_CORPUS, finite_one_epoch_data

PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp299_contacts_delta_stream_v2_sequence_prefix"
CACHE_ROOT = os.environ.get(
    "EXP299_CACHE_ROOT", f"{PREFIX}/exp277_full_epoch_tokenized_cache/2026.09.22.1"
)
OUTPUT_PREFIX = os.environ.get("EXP299_OUTPUT_PREFIX", f"{PREFIX}/checkpoints")
TOKENIZER_PATH = str(Path(__file__).with_name("tokenizer_exp277_v2"))
TRAIN_CORPORA = ("native-afdb", "native-esm", "mpnn-afdb", "mpnn-esm")
EXPECTED_PACKED_EXAMPLES = 26_942_937
TRAIN_STEPS = 210_492
SEQ_LEN = 8192
GLOBAL_BATCH_SIZE = 128
STEPS_PER_EVAL = 2_114
RUN_NAME = os.environ.get("EXP299_NAME", "delta-v2-exp277-full-epoch-1_5b-4x4gb200-a01")

logger = logging.getLogger(__name__)


def data_config() -> LmDataConfig:
    """Build finite training and independent validation components."""
    fmt = PackableTokenIdsFormat()
    children = {
        corpus: dataclasses.replace(
            DatasetComponent(cache_dir=f"{CACHE_ROOT}/{corpus}/train", format=fmt, pack=True),
            split="train",
            flat_cache=True,
            tags=["delta-v2", corpus],
        )
        for corpus in TRAIN_CORPORA
    }
    validation = dataclasses.replace(
        DatasetComponent(cache_dir=f"{CACHE_ROOT}/validation", format=fmt, pack=True),
        split="validation",
        tags=["delta-v2", "validation"],
    )
    base = LmDataConfig(
        tokenizer=TOKENIZER_PATH,
        vocab_size=VOCAB_SIZE,
        cache_dir=None,
        auto_build_caches=False,
        components={FULL_CORPUS: ConcatDatasetComponent(children=children), "delta-v2/validation": validation},
        train_weights={FULL_CORPUS: 1.0, "delta-v2/validation": 0.0},
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel"),
        mixture_block_size=1,
        block_cross_document_attention=True,
    )
    return finite_one_epoch_data(base, expected_packed_examples=EXPECTED_PACKED_EXAMPLES)


def resources(nodes: int) -> ResourceConfig:
    """Return a 4-GPU GB200 gang matching the requested node count."""
    if nodes not in (2, 4):
        raise ValueError(f"nodes must be 2 or 4, got {nodes}")
    return ResourceConfig.with_gpu("GB200", count=4, replicas=nodes, cpu=32, ram="256GB", disk="256GB")


def pod_config(run_resources: ResourceConfig, *, nodes: int) -> TrainLmOnPodConfig:
    """Construct the exact one-epoch exp277 optimizer recipe for V2."""
    output_path = f"{OUTPUT_PREFIX}/{RUN_NAME}"
    per_device = min(8, GLOBAL_BATCH_SIZE // (4 * nodes))
    train_config = TrainLmConfig(
        data=data_config(),
        trainer=TrainerConfig(
            id=RUN_NAME,
            tracker=WandbConfig(
                project="MarinFold",
                name=RUN_NAME,
                group="exp299-delta-v2-exp277-full-epoch",
                tags=[
                    "exp299",
                    "delta-v2",
                    "exp277-corpus",
                    "full-corpus-one-epoch",
                    "qwen3",
                    "1_5b",
                    f"nodes={nodes}",
                    "gb200",
                ],
                replicate_path=output_path,
            ),
            seed=0,
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=GLOBAL_BATCH_SIZE,
            per_device_parallelism=per_device,
            per_device_eval_parallelism=per_device,
            num_train_steps=TRAIN_STEPS,
            steps_per_eval=STEPS_PER_EVAL,
            max_eval_batches=None,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=15),
                keep=[{"every": TRAIN_STEPS // 10}],
            ),
            mesh=_mesh(1),
            allow_nondivisible_batch_size=True,
        ),
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=1e-3,
            weight_decay=0.2,
            beta1=0.9,
            beta2=0.95,
            warmup=0.1,
            decay=0.2,
            lr_schedule="linear",
            min_lr_ratio=0.1,
        ),
        hf_save_steps=None,
        train_seq_len=SEQ_LEN,
        data_seed=0,
        adapter=NoAdaptorConfig(),
    )
    env_vars = {
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "JAX_COMPILATION_CACHE_DIR": "/tmp/jax-compilation-cache",
        "EXP299_DOCUMENT_STRUCTURE": "delta-stream-v2-sequence-prefix-exp277-full-epoch",
    }
    return TrainLmOnPodConfig(
        train_config=train_config,
        resources=run_resources,
        output_path=output_path,
        env_vars=env_vars,
        auto_build_caches=False,
    )


def dispatch() -> None:
    """Submit the training gang from an authenticated CoreWeave root job."""
    nodes = int(os.environ.get("EXP299_NODES", "4"))
    run_resources = resources(nodes)
    config = pod_config(run_resources, nodes=nodes)
    env = resolve_training_env(base_env=dict(config.env_vars or {}), resources=run_resources)
    for key in (
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "FSSPEC_S3",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        if value := os.environ.get(key):
            env[key] = value
    request = JobRequest(
        name=RUN_NAME,
        entrypoint=Entrypoint.from_callable(run_levanter_train_lm, args=[config]),
        resources=run_resources,
        environment=create_environment(env_vars=env, extras=extras_for_resources(run_resources)),
        priority=3,
        processes_per_task=1,
        max_retries_failure=3,
    )
    logger.info("Submitting %s on %d x 4 GB200", RUN_NAME, nodes)
    job = current_client().submit(request)
    print(getattr(job, "name", str(job)), flush=True)
    job.wait(raise_on_failure=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dispatch()

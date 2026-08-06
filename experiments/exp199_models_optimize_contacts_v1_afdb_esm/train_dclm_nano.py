# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a nano Llama for ten steps on an existing DCLM token cache.

This is an infrastructure smoke test for issue #199. It deliberately uses only
published Marin packages installed by this experiment and never imports from a
Marin source checkout. The DCLM dependency is adopted as an already-tokenized
artifact: this graph has no tokenization callable and cannot rebuild the cache.

Print the plan locally::

    uv run --extra tpu python train_dclm_nano.py --version 2026.08.06

The launch command is documented in the experiment README.
"""

import os

import click
from fray.types import ResourceConfig
from levanter.data.text.datasets import DatasetComponent, UrlDatasetSourceConfig
from levanter.models.llama import LlamaConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

DCLM_CACHE_URI = "gs://marin-eu-west4/tokenized/dclm_baseline-0206f1/"
DCLM_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
RUN_ID = "exp199-dclm-nano-v6e4-smoke"

SEQ_LEN = 512
BATCH_SIZE = 32
NUM_TRAIN_STEPS = 10

TRAIN_RESOURCES = ResourceConfig.with_tpu(
    "v6e-4",
    slice_count=1,
    cpu=32,
    ram="128g",
    disk="50g",
)

LLAMA_NANO = LlamaConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=32,
    intermediate_dim=128,
    num_heads=2,
    num_kv_heads=2,
    num_layers=2,
)


class DclmTrainingCache(TokenizedCache):
    """Legacy DCLM cache with explicit training-split and packing semantics."""

    def as_component(self) -> DatasetComponent:
        source = UrlDatasetSourceConfig(
            tags=self.tags,
            train_urls=[],
            validation_urls=[],
            cache_dir=self.cache_dir,
            format=self.format,
        )
        return DatasetComponent(
            source=source,
            cache_dir=source.cache_dir,
            format=source.format,
            tags=source.tags,
            split="train",
            pack=True,
        )


def dclm_tokens() -> ArtifactStep[TokenizedCache]:
    """Return the existing regional DCLM cache as a non-computable handle."""
    return ArtifactStep[TokenizedCache].adopt(
        "external/tokenized/dclm-baseline-llama3",
        "2024.11.26",
        source=DCLM_CACHE_URI,
        kind=DclmTrainingCache,
        config={
            "tokenizer": DCLM_TOKENIZER,
            "format": {"text_key": "text"},
            "tags": ["dclm", "llama3", "pretokenized"],
        },
    )


def training_env() -> dict[str, str]:
    """Validate credentials and return W&B routing from ``marin.env``.

    Fray forwards ``HF_TOKEN`` and ``WANDB_API_KEY`` from the driver process via
    its job environment defaults. Keeping their values out of this config also
    keeps them out of artifact fingerprints and provenance records.
    """
    required = ("WANDB_API_KEY", "HF_TOKEN", "WANDB_ENTITY", "WANDB_PROJECT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    return {
        "WANDB_ENTITY": os.environ["WANDB_ENTITY"],
        "WANDB_PROJECT": os.environ["WANDB_PROJECT"],
    }


def build() -> ArtifactStep[LevanterCheckpoint]:
    """Assemble the complete ten-step smoke-training artifact."""
    tokens = dclm_tokens()
    env = training_env()
    return train_lm(
        name=f"checkpoints/{RUN_ID}",
        run_id=RUN_ID,
        model=LLAMA_NANO,
        optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
        datasets={tokens: 1.0},
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=NUM_TRAIN_STEPS,
        z_loss_weight=None,
        evals=None,
        resources=TRAIN_RESOURCES,
        steps_per_eval=NUM_TRAIN_STEPS,
        wandb_project=env["WANDB_PROJECT"],
        wandb_group="exp199",
        tags=["exp199", "dclm", "llama", "nano", "smoke", "v6e-4"],
        env_vars=env,
    )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    return build()


if __name__ == "__main__":
    main()

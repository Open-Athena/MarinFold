# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared library code for MarinFold model-training experiments.

This package intentionally keeps top-level imports lazy. Some helpers depend on
fast-moving Marin/Levanter training APIs, while lightweight submodules such as
``marinfold_models.document_loss`` should remain importable without importing the
whole training stack.
"""

from typing import Any

__all__ = [
    "MARIN_PRECISION",
    "MPQueueShardDatasetStats",
    "MPQueueShardDocumentDataset",
    "SimpleTrainConfig",
    "build_train_lm_on_pod_config",
]


def __getattr__(name: str) -> Any:
    if name in {"MARIN_PRECISION", "build_train_lm_on_pod_config"}:
        from marinfold_models import defaults

        return getattr(defaults, name)
    if name in {"MPQueueShardDatasetStats", "MPQueueShardDocumentDataset"}:
        from marinfold_models import mp_queue_shard_dataset

        return getattr(mp_queue_shard_dataset, name)
    if name == "SimpleTrainConfig":
        from marinfold_models.simple_train_config import SimpleTrainConfig

        return SimpleTrainConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared library code for MarinFold model-training experiments.

The core export is :func:`build_train_lm_on_pod_config` — a StepContext-free
builder that assembles a concrete ``TrainLmOnPodConfig`` for a training run,
modelled on modern marin's ``marin.experiment.train.train_lm`` but returning a
plain config the caller submits itself (so experiments can dispatch at iris
**batch** priority; see exp108). ``SimpleTrainConfig`` is kept as a MarinFold
convenience container but is no longer required by the builder.

Experiments under ``experiments/exp<N>_models_<name>/`` import these so each one
does not have to re-vendor the marin training plumbing.
"""

from marinfold_models.defaults import MARIN_PRECISION, build_train_lm_on_pod_config
from marinfold_models.simple_train_config import SimpleTrainConfig

__all__ = [
    "MARIN_PRECISION",
    "SimpleTrainConfig",
    "build_train_lm_on_pod_config",
]

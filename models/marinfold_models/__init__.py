# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared library code for MarinFold model-training experiments.

Vendored marin helpers (``default_train``, ``default_tokenize``,
``SimpleTrainConfig``) live here so experiments under
``experiments/exp<N>_models_<slug>/`` can import them without each one
having to vendor its own copy.
"""

from marinfold_models.defaults import default_tokenize, default_train
from marinfold_models.simple_train_config import SimpleTrainConfig

__all__ = ["SimpleTrainConfig", "default_tokenize", "default_train"]

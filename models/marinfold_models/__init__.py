# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared library code for MarinFold model-training experiments.

Vendored marin helpers (``default_train``, ``default_tokenize``,
``SimpleTrainConfig``) live here so experiments under
``experiments/exp<N>_models_<name>/`` can import them without each one
having to vendor its own copy.
"""

from marinfold_models.defaults import default_tokenize, default_train
from marinfold_models.scored_document import (
    LevanterScoredDocumentBatch,
    levanter_scored_document_batch,
    scored_document_loss,
)
from marinfold_models.simple_train_config import SimpleTrainConfig
from marinfold_models.streaming_documents import (
    StreamingDocumentDataset,
    causal_lm_example_from_documents,
    streaming_document_state_path,
)

__all__ = [
    "LevanterScoredDocumentBatch",
    "SimpleTrainConfig",
    "default_tokenize",
    "default_train",
    "levanter_scored_document_batch",
    "scored_document_loss",
    "StreamingDocumentDataset",
    "causal_lm_example_from_documents",
    "streaming_document_state_path",
]

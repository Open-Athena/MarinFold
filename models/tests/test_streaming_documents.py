# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from marinfold.document_structures.documents import (
    AttentionLayout,
    Document,
    causal_training_document,
)
from marinfold_models.streaming_documents import (
    causal_lm_example_from_documents,
    streaming_document_state_path,
)


def test_causal_adapter_masks_padding_and_document_boundaries():
    first = causal_training_document([1, 2, 3])
    second = causal_training_document([4, 5])

    example = causal_lm_example_from_documents((first, second), 8, 4)

    np.testing.assert_array_equal(np.asarray(example.tokens), [1, 2, 3, 4, 5, 0, 0, 0])
    np.testing.assert_array_equal(
        np.asarray(example.loss_weight),
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_causal_adapter_rejects_noncausal_document():
    document = Document([1, 2], attention=AttentionLayout.FULL)

    with pytest.raises(ValueError, match="only accepts causal"):
        causal_lm_example_from_documents((document,), 8, 4)


def test_streaming_state_uses_process_local_model_checkpoint_sidecar():
    assert (
        streaming_document_state_path("gs://bucket/run/step-10/", process_index=3)
        == "gs://bucket/run/step-10/input/streaming-documents-process-00003.json"
    )

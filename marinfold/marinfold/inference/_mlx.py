# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""MLX backend.

Gated by the ``[mlx]`` extra (Darwin-only wheels). Uses
``mlx_lm.load`` to read HF safetensors directly — no conversion step
is required for unquantized inference.

The HF tokenizer is loaded separately with
``transformers.AutoTokenizer`` so the surfaced object is exactly the
same ``PreTrainedTokenizerFast`` the other backends expose, rather
than mlx-lm's ``TokenizerWrapper``.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load as mlx_load
from transformers import AutoTokenizer


class MlxBackend:
    """MLX-backed implementation of the :class:`Backend` protocol."""

    def __init__(self, model_path: Path):
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        # mlx_lm.load returns (model, tokenizer); we use mlx_lm's
        # tokenizer only as a sanity check that the path is loadable
        # and rely on the HF tokenizer above for the public surface.
        self._model, _ = mlx_load(str(model_path))

    @property
    def tokenizer(self):
        return self._tokenizer

    def next_token_probs(
        self,
        prompts: list[list[int]],
        target_token_ids: list[int],
    ) -> np.ndarray:
        if not prompts:
            return np.zeros((0, len(target_token_ids)), dtype=np.float32)

        lengths = {len(p) for p in prompts}
        if len(lengths) != 1:
            raise ValueError(
                f"next_token_probs requires equal-length prompts in a single call; "
                f"got lengths {sorted(lengths)}."
            )

        input_ids = mx.array(prompts, dtype=mx.int32)
        target_ids = mx.array(target_token_ids, dtype=mx.int32)

        logits = self._model(input_ids)[:, -1, :]
        # Cast to float32 before softmax so the renormalization the
        # caller does is numerically stable, especially for tiny
        # target probabilities in the tail of the distribution.
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        target_probs = probs[:, target_ids]
        return np.asarray(target_probs, dtype=np.float32)

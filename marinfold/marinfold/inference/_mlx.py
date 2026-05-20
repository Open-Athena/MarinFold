# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""MLX backend with prefix-cache reuse.

Gated by the ``[mlx]`` extra (Darwin-only wheels). Uses
``mlx_lm.load`` to read HF safetensors directly — no conversion step
is required for unquantized inference.

The HF tokenizer is loaded separately with
``transformers.AutoTokenizer`` so the surfaced object is exactly the
same ``PreTrainedTokenizerFast`` the other backends expose, rather
than mlx-lm's ``TokenizerWrapper``.

Prefix-cache strategy (what makes this fast on Apple Silicon):

1. Run the shared prefix forward once at batch=1; the resulting
   ``KVCache`` per layer holds ``(keys, values)`` of shape
   ``[1, n_kv_heads, prefix_len, head_dim]``.
2. For each chunk of ``tail_batch_size`` tails: replicate the prefix
   cache along the batch dim to ``[B, ...]`` (cheap — ``broadcast_to``
   plus a materializing ``mx.array`` copy) and run the model on the
   tail tokens with the replicated cache. The model attends over the
   cached prefix + the current tail in one forward pass.

Mathematically identical to running each (prefix + tail) full forward
pass — verified by ``test_mlx_prefix_cache_matches_uncached`` — but
avoids re-running the heavy prefix forward for every tail.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import KVCache, make_prompt_cache
from transformers import AutoTokenizer


class MlxBackend:
    """MLX-backed implementation of the :class:`Backend` protocol.

    Args:
        model_path: Local directory containing the HF safetensors
            model + tokenizer.
        tail_batch_size: Number of tails to process in a single batched
            forward pass over the cached prefix. Larger amortizes
            per-batch launch overhead; smaller bounds peak unified-
            memory use (each tail in flight needs a per-layer copy of
            the prefix KV). Default ``64`` is comfortable on a 16 GB
            Apple Silicon machine for a 1B model.
    """

    def __init__(self, model_path: Path, *, tail_batch_size: int = 64):
        if tail_batch_size < 1:
            raise ValueError(
                f"tail_batch_size must be >= 1; got {tail_batch_size}."
            )
        self._tail_batch_size = tail_batch_size
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
        prefix_token_ids: list[int],
        tail_token_ids_batch: list[list[int]],
        target_token_ids: list[int],
    ) -> np.ndarray:
        n_targets = len(target_token_ids)
        if not tail_token_ids_batch:
            return np.zeros((0, n_targets), dtype=np.float32)

        tail_lengths = {len(t) for t in tail_token_ids_batch}
        if len(tail_lengths) != 1:
            raise ValueError(
                "next_token_probs requires equal-length tails in a single call; "
                f"got lengths {sorted(tail_lengths)}."
            )
        if not prefix_token_ids:
            raise ValueError("next_token_probs requires a non-empty prefix.")
        if tail_lengths == {0}:
            raise ValueError("next_token_probs requires tails of length >= 1.")

        # 1. Prefix forward at batch=1, populate cache.
        prefix_arr = mx.array([list(prefix_token_ids)], dtype=mx.int32)
        prefix_cache = make_prompt_cache(self._model)
        _ = self._model(prefix_arr, cache=prefix_cache)
        # Force eval so the cache tensors are materialized before we
        # start replicating them — without this, every replicate
        # below triggers an independent recompute of the prefix.
        mx.eval([layer.state for layer in prefix_cache])

        target_ids = mx.array(target_token_ids, dtype=mx.int32)

        # 2. Chunk tails, replicate cache per chunk, single forward.
        out = np.empty((len(tail_token_ids_batch), n_targets), dtype=np.float32)
        for chunk_start in range(0, len(tail_token_ids_batch), self._tail_batch_size):
            chunk = tail_token_ids_batch[chunk_start : chunk_start + self._tail_batch_size]
            b = len(chunk)
            tail_arr = mx.array(chunk, dtype=mx.int32)
            batched_cache = _replicate_cache(prefix_cache, b)
            logits = self._model(tail_arr, cache=batched_cache)[:, -1, :]
            # Cast to float32 before softmax so the renormalization the
            # caller does is numerically stable, especially for tiny
            # target probabilities in the tail of the distribution.
            probs = mx.softmax(logits.astype(mx.float32), axis=-1)
            target_probs = probs[:, target_ids]
            out[chunk_start : chunk_start + b] = np.asarray(
                target_probs, dtype=np.float32
            )
        return out


def _replicate_cache(prefix_cache: list[KVCache], batch_size: int) -> list[KVCache]:
    """Return a per-layer copy of ``prefix_cache`` tiled to ``batch_size``.

    Each layer's ``(keys, values)`` is shaped ``[1, n_kv_heads,
    prefix_len, head_dim]``; we broadcast along the batch dim then
    force materialization with ``mx.array`` so the resulting tensors
    aren't views that the model's attention kernel may reject.
    """
    out: list[KVCache] = []
    for layer in prefix_cache:
        k, v = layer.state
        target_shape = (batch_size,) + tuple(k.shape[1:])
        k_b = mx.array(mx.broadcast_to(k, target_shape))
        v_b = mx.array(mx.broadcast_to(v, target_shape))
        new = KVCache()
        new.state = (k_b, v_b)
        out.append(new)
    return out

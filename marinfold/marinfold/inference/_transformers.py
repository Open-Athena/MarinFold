# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace transformers backend with prefix-cache reuse.

Gated by the ``[transformers]`` extra (pulls torch). Picks the best
available device automatically — CUDA > MPS > CPU — so the same code
runs on a Linux box, an Apple Silicon Mac (via MPS), or anywhere
torch is installed.

Uses full-vocab softmax: vocabularies for MarinFold doc structures
are tiny (~2840 tokens), so computing the whole softmax and gathering
the target columns is cheaper than any top-k dance.

Prefix-cache strategy mirrors the MLX backend: run the prefix once
to populate a ``DynamicCache``, then for each chunk of tails replicate
the cache to batch size B and run a single forward pass over the
tail tokens with the cached prefix. ``DynamicCache.key_cache`` /
``value_cache`` are per-layer tensors of shape
``[B, n_kv_heads, seq_len, head_dim]``, so replication is just a
``repeat`` along dim 0.
"""

import copy
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(name: str) -> torch.dtype:
    """Map a dtype name to a torch dtype.

    No silent device-specific downgrades — fp16 on MPS overflows the
    residual stream for bf16-trained Llama-style models and produces
    NaNs. If a user asks for fp16, they get fp16 (and may regret it);
    bf16 stays bf16 (torch ≥ 2.4 supports it on MPS); fp32 is the
    safe default for the unsure.
    """
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unknown dtype {name!r}. Options: {sorted(table)}.")
    return table[name]


class TransformersBackend:
    """transformers-backed implementation of the :class:`Backend` protocol.

    Args:
        model_path: Local directory containing the HF model + tokenizer.
        dtype: Model dtype name (see :func:`_resolve_dtype`).
        device: ``"cuda"`` / ``"mps"`` / ``"cpu"``. Defaults to best
            available.
        tail_batch_size: Tails per cached forward pass. Bounds peak
            memory: each tail in flight needs a copy of the prefix KV
            tiled along the batch dim.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
        tail_batch_size: int = 64,
    ):
        if tail_batch_size < 1:
            raise ValueError(
                f"tail_batch_size must be >= 1; got {tail_batch_size}."
            )
        self._device = device or _best_device()
        self._tail_batch_size = tail_batch_size
        torch_dtype = _resolve_dtype(dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = (
            AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch_dtype)
            .to(self._device)
            .eval()
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._device

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

        target_ids = torch.tensor(
            target_token_ids, dtype=torch.long, device=self._device
        )

        with torch.inference_mode():
            # 1. Prefix forward (batch=1), populate cache.
            prefix_ids = torch.tensor(
                [list(prefix_token_ids)], dtype=torch.long, device=self._device
            )
            prefix_cache = DynamicCache()
            self._model(
                input_ids=prefix_ids,
                past_key_values=prefix_cache,
                use_cache=True,
            )

            # 2. Chunk tails, replicate cache per chunk, batched forward.
            out = np.empty((len(tail_token_ids_batch), n_targets), dtype=np.float32)
            for chunk_start in range(
                0, len(tail_token_ids_batch), self._tail_batch_size
            ):
                chunk = tail_token_ids_batch[
                    chunk_start : chunk_start + self._tail_batch_size
                ]
                b = len(chunk)
                tail_ids = torch.tensor(
                    chunk, dtype=torch.long, device=self._device
                )
                # ``cache_position`` tells the model where the new
                # tokens land relative to the cached prefix. Without
                # it, transformers picks the wrong absolute positions
                # for RoPE.
                prefix_len = prefix_ids.shape[1]
                cache_position = torch.arange(
                    prefix_len,
                    prefix_len + tail_ids.shape[1],
                    device=self._device,
                )
                batched_cache = _replicate_cache(prefix_cache, b)
                logits = self._model(
                    input_ids=tail_ids,
                    past_key_values=batched_cache,
                    cache_position=cache_position,
                    use_cache=False,
                ).logits[:, -1, :]
                probs = torch.softmax(logits.float(), dim=-1)
                target_probs = probs.index_select(dim=-1, index=target_ids)
                out[chunk_start : chunk_start + b] = (
                    target_probs.detach().cpu().numpy().astype(np.float32, copy=False)
                )
        return out


def _replicate_cache(prefix_cache: DynamicCache, batch_size: int) -> DynamicCache:
    """Return a per-layer batch-tiled copy of ``prefix_cache``.

    transformers 5.x exposes the cache as a list of ``DynamicLayer``
    objects each holding ``[1, n_kv_heads, prefix_len, head_dim]``
    ``keys`` / ``values`` tensors. We deep-copy so the prefix cache
    stays usable for the next chunk, then call the built-in
    ``batch_repeat_interleave`` which does ``repeat_interleave(B, dim=0)``
    on each layer's keys + values in place.
    """
    out = copy.deepcopy(prefix_cache)
    out.batch_repeat_interleave(batch_size)
    return out

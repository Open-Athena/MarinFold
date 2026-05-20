# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM backend.

Gated by the ``[vllm]`` optional-dep. The module-level ``vllm``
import is intentional: this file is only imported when
``load_backend('vllm', ...)`` is called, so its import cost is
deferred until then.

vLLM exposes logprobs as a top-k dict per generated position. We use
``max_tokens=1, logprobs=top_k_logprobs`` to get the next-token
distribution without actually sampling. Target tokens that fall
outside the top-k get probability 0 — the caller renormalizes.

Prefix-cache strategy: vLLM has automatic prefix caching built into
its scheduler. We just submit each ``(prefix + tail)`` as a separate
prompt; vLLM detects the shared prefix at scheduling time and reuses
the KV blocks for it across all rows in the batch. No explicit cache
plumbing needed on our side.
"""

from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams, TokensPrompt


class VllmBackend:
    """vLLM-backed implementation of the :class:`Backend` protocol."""

    def __init__(
        self,
        model_path: Path,
        *,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        top_k_logprobs: int = 128,
        tail_batch_size: int = 64,
    ):
        if tail_batch_size < 1:
            raise ValueError(
                f"tail_batch_size must be >= 1; got {tail_batch_size}."
            )
        self._tail_batch_size = tail_batch_size
        self._top_k_logprobs = top_k_logprobs
        self._llm = LLM(
            model=str(model_path),
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            trust_remote_code=True,
            max_logprobs=max(top_k_logprobs, 128),
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            logprobs=top_k_logprobs,
            n=1,
        )

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
        if not prefix_token_ids:
            raise ValueError("next_token_probs requires a non-empty prefix.")
        tail_lengths = {len(t) for t in tail_token_ids_batch}
        if tail_lengths == {0}:
            raise ValueError("next_token_probs requires tails of length >= 1.")

        target_to_col = {tid: c for c, tid in enumerate(target_token_ids)}
        target_id_set = set(target_token_ids)

        out = np.zeros((len(tail_token_ids_batch), n_targets), dtype=np.float32)
        prefix = list(prefix_token_ids)
        for chunk_start in range(0, len(tail_token_ids_batch), self._tail_batch_size):
            chunk = tail_token_ids_batch[
                chunk_start : chunk_start + self._tail_batch_size
            ]
            full_prompts = [
                TokensPrompt(prompt_token_ids=prefix + list(tail))
                for tail in chunk
            ]
            # vLLM's automatic prefix caching reuses KV blocks for the
            # shared prefix across these prompts, while chunking keeps
            # each scheduler request bounded on long proteins.
            outputs = self._llm.generate(full_prompts, self._sampling, use_tqdm=False)
            for row_offset, result in enumerate(outputs):
                lp_dict = result.outputs[0].logprobs[0] if result.outputs[0].logprobs else {}
                for tok_id, lp in lp_dict.items():
                    tid = int(tok_id)
                    if tid in target_id_set:
                        out[chunk_start + row_offset, target_to_col[tid]] = float(
                            np.exp(float(lp.logprob))
                        )
        return out

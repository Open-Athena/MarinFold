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
    ):
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
        prompts: list[list[int]],
        target_token_ids: list[int],
    ) -> np.ndarray:
        if not prompts:
            return np.zeros((0, len(target_token_ids)), dtype=np.float32)
        target_to_col = {tid: c for c, tid in enumerate(target_token_ids)}
        target_id_set = set(target_token_ids)

        vllm_prompts = [TokensPrompt(prompt_token_ids=p) for p in prompts]
        outputs = self._llm.generate(vllm_prompts, self._sampling, use_tqdm=False)

        out = np.zeros((len(prompts), len(target_token_ids)), dtype=np.float32)
        for row_idx, result in enumerate(outputs):
            lp_dict = result.outputs[0].logprobs[0] if result.outputs[0].logprobs else {}
            for tok_id, lp in lp_dict.items():
                tid = int(tok_id)
                if tid in target_id_set:
                    out[row_idx, target_to_col[tid]] = float(np.exp(float(lp.logprob)))
        return out

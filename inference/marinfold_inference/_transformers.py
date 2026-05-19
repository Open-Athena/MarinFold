# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace transformers backend.

Gated by the ``[transformers]`` extra (pulls torch). Picks the best
available device automatically — CUDA > MPS > CPU — so the same code
runs on a Linux box, an Apple Silicon Mac (via MPS), or anywhere
torch is installed.

Uses full-vocab softmax: vocabularies for MarinFold doc structures
are tiny (~2840 tokens), so computing the whole softmax and gathering
the target columns is cheaper than any top-k dance.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    """transformers-backed implementation of the :class:`Backend` protocol."""

    def __init__(
        self,
        model_path: Path,
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
    ):
        self._device = device or _best_device()
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
        prompts: list[list[int]],
        target_token_ids: list[int],
    ) -> np.ndarray:
        if not prompts:
            return np.zeros((0, len(target_token_ids)), dtype=np.float32)

        # Caller guarantees equal-length prompts within a call. Cheap
        # assertion catches misuse early.
        lengths = {len(p) for p in prompts}
        if len(lengths) != 1:
            raise ValueError(
                f"next_token_probs requires equal-length prompts in a single call; "
                f"got lengths {sorted(lengths)}."
            )

        input_ids = torch.tensor(prompts, dtype=torch.long, device=self._device)
        target_ids = torch.tensor(target_token_ids, dtype=torch.long, device=self._device)

        with torch.inference_mode():
            logits = self._model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            target_probs = probs.index_select(dim=-1, index=target_ids)

        return target_probs.detach().cpu().numpy().astype(np.float32, copy=False)

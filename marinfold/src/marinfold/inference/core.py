# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend protocol + factory for MarinFold inference.

A backend wraps one model + tokenizer and exposes a single primitive:
given a batch of equal-length token-id prompts and a list of target
token ids, return the probability mass on each target token at the
next position. That's all the document-structure impls need.

Backends are loaded by name through :func:`load_backend`; their
concrete classes (e.g. ``VllmBackend``) live in sibling modules
behind optional-dep gates and are only imported when actually used.
"""

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np

from marinfold.registry import resolve_model

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


BackendName = Literal["vllm", "transformers", "mlx"]


@runtime_checkable
class Backend(Protocol):
    """Common surface every backend implements.

    Implementations:

    - Load the model + tokenizer from a local directory in their
      ``__init__``.
    - Expose the HuggingFace tokenizer via the :attr:`tokenizer`
      property so callers can encode prompts with it.
    - Implement :meth:`next_token_probs` against their underlying
      runtime (vLLM, torch, MLX).

    The contract for ``next_token_probs`` is intentionally narrow:
    one forward pass per prompt, no autoregressive generation, no
    sampling. Returned probabilities are over the *target token set*
    only and are not renormalized — the caller decides whether and
    how to renormalize.
    """

    @property
    def tokenizer(self) -> "PreTrainedTokenizerFast":
        ...

    def next_token_probs(
        self,
        prompts: list[list[int]],
        target_token_ids: list[int],
    ) -> np.ndarray:
        """Probability mass on each target token at the next position.

        Args:
            prompts: One token-id sequence per row. All sequences in a
                single call must have the same length — the caller is
                expected to batch by prompt length.
            target_token_ids: Token ids whose next-position
                probability the caller wants. Order is preserved in
                the output columns.

        Returns:
            Float array of shape ``(len(prompts), len(target_token_ids))``.
            Backends with full-vocab logits return real softmax mass
            on each target. The vLLM backend uses top-k logprobs and
            returns 0 for target tokens that fall outside the top-k.
            The caller renormalizes if it wants a renormalized
            distribution.
        """
        ...


def load_backend(
    name: BackendName,
    *,
    model: str | None,
    **kwargs,
) -> Backend:
    """Load one of the registered backends against a resolved model.

    ``model`` is passed through :func:`resolve_model`: a local
    directory, a ``MODELS.yaml`` nickname, or ``None`` to use the
    entry marked ``default: true``. The resolved local path is
    handed to the backend's constructor along with any backend-
    specific ``**kwargs``.

    Lazy-imports the backend's module so the optional runtime
    (``vllm`` / ``torch`` / ``mlx``) is only loaded when actually
    used.
    """
    model_path = resolve_model(model)
    if name == "vllm":
        from marinfold.inference._vllm import VllmBackend
        return VllmBackend(model_path, **kwargs)
    if name == "transformers":
        from marinfold.inference._transformers import TransformersBackend
        return TransformersBackend(model_path, **kwargs)
    if name == "mlx":
        from marinfold.inference._mlx import MlxBackend
        return MlxBackend(model_path, **kwargs)
    raise ValueError(
        f"Unknown backend {name!r}. Expected one of: 'vllm', 'transformers', 'mlx'."
    )

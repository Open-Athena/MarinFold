# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend protocol + factory for MarinFold inference.

A backend wraps one model + tokenizer and exposes a single primitive:
given a shared prefix and a batch of equal-length tails, return the
probability mass on a target token set at the position immediately
after each (prefix + tail) prompt.

The shared-prefix shape is intentional. Every MarinFold doc-structure
workload — distance prediction, contact querying — is "one prompt
prefix per protein, many short tails for each (i, j) pair." Surfacing
the prefix explicitly lets backends with KV-cache support (MLX,
transformers, vLLM-via-its-own-prefix-cache) compute the prefix
forward pass once instead of recomputing it for every pair. On the
MLX backend this is a ~50-100× speedup over naive concat-and-rerun.

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

    The :meth:`next_token_probs` contract is intentionally narrow:
    one forward pass per tail, no autoregressive generation, no
    sampling. Returned probabilities are over the *target token set*
    only and are not renormalized — the caller decides whether and
    how to renormalize.

    Internal batching across tails (memory-aware chunking, prefix-
    cache replication) is the backend's responsibility. Callers pass
    every tail they want scored against the prefix in a single call;
    the backend produces one row per tail in the returned array.
    """

    @property
    def tokenizer(self) -> "PreTrainedTokenizerFast":
        ...

    def next_token_probs(
        self,
        prefix_token_ids: list[int],
        tail_token_ids_batch: list[list[int]],
        target_token_ids: list[int],
    ) -> np.ndarray:
        """Probability mass on each target token after each (prefix + tail).

        Args:
            prefix_token_ids: Shared prompt prefix (token ids). Same
                prefix is used for every row of ``tail_token_ids_batch``.
                Backends with prefix caching compute this once.
            tail_token_ids_batch: One token-id sequence per row. All
                tails in a single call must have the same length
                (≥ 1). May be empty — the backend returns a
                ``(0, len(target_token_ids))`` array.
            target_token_ids: Token ids whose next-position
                probability the caller wants. Order is preserved in
                the output columns.

        Returns:
            Float array of shape
            ``(len(tail_token_ids_batch), len(target_token_ids))``.
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

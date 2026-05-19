# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend abstraction for running MarinFold models.

Three backends are available behind one :class:`Backend` protocol:

- ``"vllm"``       — Linux + GPU production path.
- ``"transformers"`` — HuggingFace transformers; runs on Apple
  Silicon (MPS), CPU, and CUDA. Lowest-effort local-eval option.
- ``"mlx"``        — Apple Silicon native via MLX.

Each backend's heavy runtime is lazy-imported inside its module, so
importing :mod:`marinfold_inference` itself requires only the base
deps and works with zero backends installed.

The library is protein-unaware. Document-structure impls under
``experiments/exp<N>_document_structures_<name>/`` build prompts as
opaque token-id lists and pass them in.
"""

from marinfold_inference.core import Backend, load_backend
from marinfold_inference.registry import resolve_model

__all__ = ["Backend", "load_backend", "resolve_model"]

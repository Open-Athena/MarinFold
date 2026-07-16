# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend abstraction for running MarinFold models.

Three backends are available behind one :class:`Backend` protocol:

- ``"vllm"``         — Linux + GPU production path.
- ``"transformers"`` — HuggingFace transformers; runs on Apple
  Silicon (MPS), CPU, and CUDA. Lowest-effort local-eval option.
- ``"mlx"``          — Apple Silicon native via MLX.

Each backend's heavy runtime is lazy-imported inside its module, so
importing :mod:`marinfold.inference` itself requires only the base
deps and works with zero backends installed.

The library is protein-unaware. Document-structure impls
(``marinfold.document_structures`` and the impl packages under
``document_structures/``) build prompts as opaque token-id
lists and pass them in.
"""

from marinfold.inference.core import Backend, load_backend

__all__ = ["Backend", "load_backend"]

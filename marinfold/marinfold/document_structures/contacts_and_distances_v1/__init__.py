# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v1 document structure.

The first MarinFold document structure: serialize each protein as a
sequence of AA-residue tokens followed by a stream of statements
(distances + contacts) over residue pairs. Each ``<d_X.X>`` token
covers a 0.5 Å bin (64 bins, 0.5–32.0 Å).

Public surface:

- :data:`NAME`, :data:`CONTEXT_LENGTH`, :func:`all_domain_tokens` —
  from :mod:`.vocab`.
- :func:`generate_documents`, :class:`GenerationConfig` —
  from :mod:`.generate`.
- :func:`predict`, :func:`evaluate`, :class:`InferenceConfig` —
  from :mod:`.inference`. These are what the top-level ``marinfold``
  CLI dispatches to.
- :func:`plot_infer_pdf`, :func:`plot_evaluate_pdf` — from
  :mod:`.plots`. Optional PDF writers used by ``marinfold infer
  --out-plots`` / ``marinfold evaluate --out-plots``. Matplotlib is
  imported lazily inside the helpers.
"""

from .generate import GenerationConfig, generate_documents
from .inference import InferenceConfig, evaluate, predict
from .parse import structure_from_sequence
from .plots import plot_evaluate_pdf, plot_infer_pdf
from .vocab import CONTEXT_LENGTH, NAME, all_domain_tokens

__all__ = [
    "CONTEXT_LENGTH",
    "GenerationConfig",
    "InferenceConfig",
    "NAME",
    "all_domain_tokens",
    "evaluate",
    "generate_documents",
    "plot_evaluate_pdf",
    "plot_infer_pdf",
    "predict",
    "structure_from_sequence",
]

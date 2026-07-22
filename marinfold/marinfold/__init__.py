# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""MarinFold: backends + document-structure toolkit + CLI.

Public API is re-exported from this module so callers can write::

    from marinfold import Backend, load_backend, resolve_model
    from marinfold import EvalResult, build_tokenizer
    from marinfold import write_docs, write_predictions, write_eval

The submodules are still importable directly
(:mod:`marinfold.inference`, :mod:`marinfold.document_structures`)
when you need something not in the re-export list.
"""

from marinfold.document_structures import (
    EvalResult,
    Token,
    TokenFamily,
    Vocabulary,
    VocabularyBuilder,
    VocabularyIdentity,
    build_tokenizer,
    write_docs,
    write_eval,
    write_predictions,
)
from marinfold.inference import Backend, load_backend
from marinfold.registry import (
    ModelEntry,
    default_model_nickname,
    list_model_entries,
    resolve_model,
    resolve_model_entry,
)

__all__ = [
    "Backend",
    "EvalResult",
    "ModelEntry",
    "Token",
    "TokenFamily",
    "Vocabulary",
    "VocabularyBuilder",
    "VocabularyIdentity",
    "build_tokenizer",
    "default_model_nickname",
    "list_model_entries",
    "load_backend",
    "resolve_model",
    "resolve_model_entry",
    "write_docs",
    "write_eval",
    "write_predictions",
]

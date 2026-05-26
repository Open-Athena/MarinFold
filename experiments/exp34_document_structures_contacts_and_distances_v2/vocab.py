# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical vocabulary for the contacts-and-distances-v2 format.

v2 reuses every v1 token in its existing order, then appends the two
new tokens this structure introduces:

- ``<contacts-and-distances-v2>`` — the doc-structure marker that
  appears as the first token of every v2 document.
- ``<think>`` — the pause / scratch token whose insertions are the
  whole point of v2 (see ``generate.py``).

Append-only is load-bearing: it keeps every v1 token ID stable, so a
checkpoint pretrained on v1 can be warm-started on v2 simply by
growing the embedding table by 2 rows. Reordering would be a v3
event.

The v1 vocab is imported directly from the marinfold kind library so
this file stays a true delta — there is no risk of v1 and v2 silently
disagreeing on the position of a shared token, and the published v1
tokenizer (``timodonnell/protein-docs-tokenizer@83f597d88e9b``)
remains the source of truth for the shared prefix.
"""

from marinfold.document_structures.contacts_and_distances_v1.vocab import (
    AMINO_ACIDS,
    ATOM_NAMES,
    CONTACT_TYPES,
    CONTROL_TOKENS as V1_CONTROL_TOKENS,
    DISTANCE_BINS,
    DISTANCE_MARKER,
    MAX_POSITION,
    PLDDT_BINS,
    POSITION_TOKENS,
    UNK_TOKEN,
)
from marinfold.document_structures.contacts_and_distances_v1.vocab import (
    all_domain_tokens as v1_all_domain_tokens,
)


NAME = "contacts-and-distances-v2"
CONTEXT_LENGTH = 8192

THINK_TOKEN = "<think>"


# The two tokens v2 introduces, in append order. Kept as a module-
# level constant so tests can assert ``v2 == v1 + V2_NEW_TOKENS``.
V2_NEW_TOKENS = [
    f"<{NAME}>",
    THINK_TOKEN,
]


def all_domain_tokens() -> list[str]:
    """Return the canonical v2 domain vocabulary (2840 tokens).

    Layout:

    - ids 0..2837: the v1 domain vocab, byte-for-byte, in its
      original order (CONTROL → CONTACT_TYPES → DISTANCE_MARKER →
      DISTANCE_BINS → PLDDT_BINS → AAs → atoms → positions → UNK).
    - id 2838: ``<contacts-and-distances-v2>``.
    - id 2839: ``<think>``.

    Build the published tokenizer via
    :func:`marinfold.build_tokenizer(all_domain_tokens())`; that
    prepends ``<pad>`` (id 0) and ``<eos>`` (id 1), so the encoded
    tokenizer is 2842 entries total.
    """
    out = list(v1_all_domain_tokens())
    out.extend(V2_NEW_TOKENS)
    return out


# Re-export the v1 categories so callers that want to introspect the
# vocab (tests, downstream tools) don't have to know that v2's
# vocabulary is "v1 plus two tokens" — they just import from here.
__all__ = [
    "AMINO_ACIDS",
    "ATOM_NAMES",
    "CONTACT_TYPES",
    "CONTEXT_LENGTH",
    "DISTANCE_BINS",
    "DISTANCE_MARKER",
    "MAX_POSITION",
    "NAME",
    "PLDDT_BINS",
    "POSITION_TOKENS",
    "THINK_TOKEN",
    "UNK_TOKEN",
    "V1_CONTROL_TOKENS",
    "V2_NEW_TOKENS",
    "all_domain_tokens",
    "v1_all_domain_tokens",
]

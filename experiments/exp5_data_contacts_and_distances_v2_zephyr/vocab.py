# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical vocabulary for the contacts-and-distances-v2 format.

Identical to exp34's vocab.py — v2 reuses every v1 token in its existing
order, then appends two new tokens (``<contacts-and-distances-v2>`` and
``<think>``). Append-only is load-bearing: it keeps every v1 token ID
stable, so a checkpoint pretrained on v1 can be warm-started on v2 simply
by growing the embedding table by 2 rows.

The v1 vocab is imported directly from the marinfold kind library, so v1
and v2 can never silently disagree on the position of a shared token.
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

V2_NEW_TOKENS = [
    f"<{NAME}>",
    THINK_TOKEN,
]


def all_domain_tokens() -> list[str]:
    """v2 domain vocabulary = v1 domain vocab ++ [``<v2>``, ``<think>``]."""
    out = list(v1_all_domain_tokens())
    out.extend(V2_NEW_TOKENS)
    return out


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

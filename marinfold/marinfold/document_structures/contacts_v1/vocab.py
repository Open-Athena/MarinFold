# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical vocabulary for the contacts-v1 format.

Local to this document-structure package. Imported by ``generate.py``,
``parse.py``, and ``cli.py``.

Order is **load-bearing**. Token IDs derived from this list must stay
stable for every checkpoint trained against the contacts-v1 vocab.
Append-only — never reorder. (Reordering is a v2 event: new structure,
new tokenizer, new experiment.)

The vocab is the union of three groups, in this fixed order:

1. contacts-v1's own control tokens, then the 2000 unpadded position
   tokens ``<pos-0>`` .. ``<pos-1999>``, then ``<think>``.
2. every token in the contacts-and-distances-v1
   :func:`~marinfold.document_structures.contacts_and_distances_v1.vocab.all_domain_tokens`
   list, deduplicated against group 1. Per SPEC.md these are carried as
   *additional* tokens so a contacts-v1 model can later be fine-tuned on
   contacts-and-distances-v1 documents without changing the tokenizer.
   This group is also where contacts-v1 gets its amino-acid tokens
   (uppercase ``<ALA>`` .. ``<VAL>``) and ``<UNK>`` from — contacts-v1
   intentionally reuses them rather than minting a parallel set.

The only token shared between groups 1 and 2 is ``<end>``; dedup keeps
the group-1 (lower-id) copy so the two structures share one ``<end>``.
"""

from marinfold.document_structures.contacts_and_distances_v1.vocab import (
    all_domain_tokens as _cd_v1_all_domain_tokens,
)


NAME = "contacts-v1"
CONTEXT_LENGTH = 8192

# Residues are indexed into 2000 position tokens with wrap-around (see
# generate.py). This caps the longest single chain we can serialize:
# structures with more residues than indices can't be uniquely numbered
# and are dropped at generation time.
NUM_POSITION_INDICES = 2000

# contacts-v1's own control tokens. ``<end>`` is shared with
# contacts-and-distances-v1 (see module docstring).
CONTROL_TOKENS = [
    "<contacts-v1>",
    "<begin-sequence>",
    "<begin-structure>",
    "<n-term>",
    "<c-term>",
    "<contact>",
    "<end>",
]

# Unpadded position tokens <pos-0> .. <pos-1999>.
POSITION_TOKENS = [f"<pos-{i}>" for i in range(NUM_POSITION_INDICES)]

# Reasoning scratch token reserved by SPEC.md. Unused by the generator.
THINK_TOKEN = ["<think>"]


def contacts_v1_native_tokens() -> list[str]:
    """The tokens contacts-v1 mints itself (control + positions + think).

    Group 1 of the vocab. Amino-acid / ``<UNK>`` tokens are *not* here —
    they come from the contacts-and-distances-v1 block (group 2).
    """
    return [*CONTROL_TOKENS, *POSITION_TOKENS, *THINK_TOKEN]


def additional_tokens() -> list[str]:
    """contacts-and-distances-v1 tokens carried as additional vocab (group 2).

    Deduplicated against :func:`contacts_v1_native_tokens` (only ``<end>``
    overlaps). Returned in contacts-and-distances-v1's canonical order with
    the overlapping token removed.
    """
    native = set(contacts_v1_native_tokens())
    return [t for t in _cd_v1_all_domain_tokens() if t not in native]


def all_domain_tokens() -> list[str]:
    """Return the full ordered contacts-v1 domain vocabulary.

    ``build_tokenizer`` prepends the ``<pad>`` / ``<eos>`` specials (ids 0
    and 1); this function returns the domain tokens only, starting at id 2.

    The group order (native control → positions → ``<think>`` → the
    deduplicated contacts-and-distances-v1 block) and the within-group
    order are both load-bearing.
    """
    return [*contacts_v1_native_tokens(), *additional_tokens()]

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical vocabulary for the contacts-v1 format.

Local to this document-structure package. Imported by ``generate.py``,
``parse.py``, and ``cli.py``.

Order is **load-bearing**. Token IDs derived from this list must stay
stable for every checkpoint trained against the contacts-v1 vocab.
Append-only — never reorder. (Reordering is a v2 event: new structure,
new tokenizer, new experiment.)

contacts-v1 **reuses contacts-and-distances-v1 tokens wherever a token
with the same meaning already exists**, so the two structures share token
IDs / embeddings and a contacts-v1 model can later be fine-tuned on
contacts-and-distances-v1 documents without a tokenizer change. Concretely
a contacts-v1 document emits:

- ``<contacts-v1>``, ``<n-term>``, ``<c-term>``, ``<contact>`` — minted
  here (no contacts-and-distances-v1 analog), plus the unused ``<think>``.
- ``<begin_sequence>`` / ``<begin_statements>`` — the section markers,
  reused from contacts-and-distances-v1 (its underscore spelling).
- ``<p0>`` .. ``<p1999>`` — residue position indices, reused from
  contacts-and-distances-v1's ``<p0>`` .. ``<p2700>``.
- ``<ALA>`` .. ``<VAL>`` (uppercase amino acids), ``<UNK>``, and ``<end>``
  — reused from contacts-and-distances-v1.

The vocab is therefore: the 5 native tokens, then the entire
contacts-and-distances-v1 ``all_domain_tokens()`` list (which supplies
every reused token plus the rest of its vocab, carried so the fine-tuning
path keeps a single tokenizer). The two groups are disjoint.
"""

from marinfold.document_structures.contacts_and_distances_v1.vocab import (
    MAX_POSITION as _CD_V1_MAX_POSITION,
)
from marinfold.document_structures.contacts_and_distances_v1.vocab import (
    all_domain_tokens as _cd_v1_all_domain_tokens,
)


NAME = "contacts-v1"
CONTEXT_LENGTH = 8192

# Residues are indexed into NUM_POSITION_INDICES position tokens with
# wrap-around (see generate.py). This caps the longest single chain we can
# serialize: structures with more residues than indices can't be uniquely
# numbered and are dropped at generation time.
NUM_POSITION_INDICES = 2000

# --- Tokens contacts-v1 mints itself (no contacts-and-distances-v1 analog) ---
DOC_TYPE_TOKEN = "<contacts-v1>"
N_TERM_TOKEN = "<n-term>"
C_TERM_TOKEN = "<c-term>"
CONTACT_TOKEN = "<contact>"
# Reasoning scratch token reserved by SPEC.md. Unused by the generator.
THINK_TOKEN = "<think>"

NATIVE_TOKENS = [
    DOC_TYPE_TOKEN,
    N_TERM_TOKEN,
    C_TERM_TOKEN,
    CONTACT_TOKEN,
    THINK_TOKEN,
]

# --- Tokens reused from contacts-and-distances-v1 (emitted, not minted) ---
BEGIN_SEQUENCE_TOKEN = "<begin_sequence>"      # start of the sequence section
BEGIN_STRUCTURE_TOKEN = "<begin_statements>"   # start of the structure section
END_TOKEN = "<end>"


def position_token(index: int) -> str:
    """Token for a residue position index — reused ``<pX>`` from c-and-d-v1."""
    return f"<p{index}>"


def _validate_reuse() -> None:
    """Fail loudly if a reused token isn't actually in the c-and-d-v1 vocab.

    Guards against the two vocabs drifting apart (e.g. a c-and-d-v1 rename
    silently turning a "reused" token into a contacts-v1-only token).
    """
    if NUM_POSITION_INDICES > _CD_V1_MAX_POSITION + 1:
        raise ValueError(
            f"contacts-v1 needs <p0>..<p{NUM_POSITION_INDICES - 1}> but "
            f"contacts-and-distances-v1 only defines up to "
            f"<p{_CD_V1_MAX_POSITION}>"
        )
    cd_v1 = set(_cd_v1_all_domain_tokens())
    reused = {BEGIN_SEQUENCE_TOKEN, BEGIN_STRUCTURE_TOKEN, END_TOKEN,
              position_token(0), position_token(NUM_POSITION_INDICES - 1)}
    missing = reused - cd_v1
    if missing:
        raise ValueError(
            f"contacts-v1 reuses tokens absent from contacts-and-distances-v1: "
            f"{sorted(missing)}"
        )


_validate_reuse()


def contacts_v1_native_tokens() -> list[str]:
    """The tokens contacts-v1 mints itself (no contacts-and-distances-v1 analog)."""
    return list(NATIVE_TOKENS)


def additional_tokens() -> list[str]:
    """The full contacts-and-distances-v1 vocab, carried in this tokenizer.

    Supplies every token contacts-v1 reuses (section markers, ``<p*>``,
    amino acids, ``<UNK>``, ``<end>``) plus the rest of the
    contacts-and-distances-v1 vocab (distance bins, atoms, …) so the
    fine-tuning path keeps a single tokenizer. Disjoint from
    :func:`contacts_v1_native_tokens`.
    """
    native = set(NATIVE_TOKENS)
    return [t for t in _cd_v1_all_domain_tokens() if t not in native]


def all_domain_tokens() -> list[str]:
    """Return the full ordered contacts-v1 domain vocabulary.

    ``build_tokenizer`` prepends the ``<pad>`` / ``<eos>`` specials (ids 0
    and 1); this function returns the domain tokens only, starting at id 2.

    The group order (the native tokens, then the contacts-and-distances-v1
    block) and the within-group order are both load-bearing.
    """
    return [*contacts_v1_native_tokens(), *additional_tokens()]

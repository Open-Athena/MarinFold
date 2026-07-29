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
  here (no contacts-and-distances-v1 analog), plus ``<think>`` (emitted only
  in ``think=True`` generation; see ``generate.py`` / #123).
- ``<begin_sequence>`` / ``<begin_statements>`` — the section markers,
  reused from contacts-and-distances-v1 (its underscore spelling).
- ``<p0>`` .. ``<p1999>`` — residue position indices, reused from
  contacts-and-distances-v1's ``<p0>`` .. ``<p2700>``.
- ``<ALA>`` .. ``<VAL>`` (uppercase amino acids), ``<UNK>``, and ``<end>``
  — reused from contacts-and-distances-v1.

The vocab is therefore: the 5 native tokens, then the entire
contacts-and-distances-v1 ``all_domain_tokens()`` list (which supplies
every reused token plus the rest of its vocab, carried so the fine-tuning
path keeps a single tokenizer), then two trailing tokens appended in the
order they were introduced: the sequence-only variant
(``<contacts-v1.sequence_only>``; see :data:`SEQUENCE_ONLY_TOKENS`) and the
retraction token (``<retract>``; see :data:`RETRACT_TOKENS`, issue #158),
and the retraction-mode doc type (``<contacts-v1.backtracking>``; see
:data:`BACKTRACKING_TOKENS`, issue #175).
The groups are disjoint, and each trailing token was appended **last** when
added so introducing it left every pre-existing token id unchanged
(append-only) — an existing checkpoint only grows its embedding table by
one row per trailing token.

The sibling ``contacts-and-coordinates-v1`` format inherits this vocab as
its own leading block but freezes it at the pre-retraction 2844 tokens (it
has no retraction), so ``<retract>`` lives in the contacts-v1 tokenizer
only and no coordinate-format token id moves. See that format's
``vocab.inherited_tokens``.
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
# Pause / reasoning-scratch token. Emitted in the structure section only when
# the generator runs with GenerationConfig(think=True) (issue #123); the
# default generator leaves it unused. No vocab change was needed — it was
# reserved here from the start for exactly this.
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

# --- Sequence-only variant (contacts-v1.sequence_only) ---
# Doc type that emits ONLY the sequence section (no structure section). It
# lets a sequence-only corpus (e.g. UniRef50; see exp64) live in the
# contacts-v1 token space so it can be mixed with the contacts-v1 corpus
# under one tokenizer. Minted by contacts-v1 but deliberately kept OUT of
# NATIVE_TOKENS and appended LAST in all_domain_tokens(): that way adding it
# left every pre-existing contacts-v1 / contacts-and-distances-v1 token id
# unchanged (append-only — see the module docstring).
SEQUENCE_ONLY_DOC_TYPE_TOKEN = "<contacts-v1.sequence_only>"
SEQUENCE_ONLY_TOKENS = [SEQUENCE_ONLY_DOC_TYPE_TOKEN]

# --- Retraction extension (issue #158) ---
# ``<retract> <pX> <pY>`` takes back a previously emitted ``<contact> <pX>
# <pY>``: the structure section becomes an ordered edit list, and the final
# contact set is whatever is still live at ``<end>`` (see ``read.py`` for the
# fold). Native-minted by contacts-v1 (no analog in any prior format), but —
# exactly like SEQUENCE_ONLY_TOKENS — deliberately kept OUT of NATIVE_TOKENS
# and appended LAST in all_domain_tokens(): adding it left every pre-existing
# contacts-v1 / contacts-and-distances-v1 / sequence-only id unchanged
# (append-only), so an existing checkpoint just grows its embedding by one
# row. The generator never emits it (retraction documents are synthesised by
# the model-in-the-loop corpus job, issue #159); it is reserved here so the
# tokenizer and inference fold are ready for retraction-trained models.
RETRACT_TOKEN = "<retract>"
RETRACT_TOKENS = [RETRACT_TOKEN]

# --- Retraction-mode doc type (issue #175) ---
# Doc type for a document that MAY contain ``<retract>`` statements. The
# statements themselves are unchanged; this only tells the model, at token 0,
# which mode it is generating in.
#
# #160 trained on a 50:50 mixture of retraction-bearing and clean documents
# that began with the *identical* prefix, so nothing distinguished them --
# and 20.1% of the retraction half happened to contain no ``<retract>`` at
# all, making that fifth indistinguishable in the body too. A model in that
# position has to marginalise over "may I take this back later?" at every
# step, which shows up as retracting on only 43% of rollouts and, more
# expensively, as a -0.0251 R-precision regression in *emission* quality
# before any retraction is honoured: in retraction mode the optimal policy is
# more speculative, and with no marker that speculativeness leaks into clean
# generation.
#
# Same treatment as SEQUENCE_ONLY_TOKENS and RETRACT_TOKENS: minted here,
# deliberately OUT of NATIVE_TOKENS, and appended LAST in all_domain_tokens()
# so every pre-existing id is unchanged (append-only) and an existing
# checkpoint grows by exactly one embedding row.
BACKTRACKING_DOC_TYPE_TOKEN = "<contacts-v1.backtracking>"
BACKTRACKING_TOKENS = [BACKTRACKING_DOC_TYPE_TOKEN]


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


def sequence_only_tokens() -> list[str]:
    """The token(s) the sequence-only variant mints, appended second-to-last.

    Currently just ``<contacts-v1.sequence_only>``. Kept as its own
    trailing group (rather than folded into the leading native tokens) so
    that adding it preserved every pre-existing token id. The retraction
    token (:func:`retract_tokens`) was appended after it, so this group is
    no longer the very last one — but its id is unchanged (append-only).
    """
    return list(SEQUENCE_ONLY_TOKENS)


def retract_tokens() -> list[str]:
    """The retraction token(s), appended last (issue #158).

    Currently just ``<retract>``. Its own trailing group, after the
    sequence-only token, so introducing it left every pre-existing id
    (including the sequence-only token's) unchanged. The
    contacts-and-coordinates-v1 format excludes this group from the vocab it
    inherits (it has no retraction), so no coordinate-format id moves.
    """
    return list(RETRACT_TOKENS)


def backtracking_tokens() -> list[str]:
    """The retraction-mode doc type, appended last (issue #175).

    Currently just ``<contacts-v1.backtracking>``. Its own trailing group,
    after the retraction token, so introducing it left every pre-existing id
    (including ``<retract>``'s) unchanged.

    The two coordinate formats exclude this group from the vocab they inherit
    for the same reason they exclude :func:`retract_tokens` — see their
    ``inherited_tokens``. Anything appended here must be added to that filter
    too, or the whole xyz/crop block shifts up by one id.
    """
    return list(BACKTRACKING_TOKENS)


def all_domain_tokens() -> list[str]:
    """Return the full ordered contacts-v1 domain vocabulary.

    ``build_tokenizer`` prepends the ``<pad>`` / ``<eos>`` specials (ids 0
    and 1); this function returns the domain tokens only, starting at id 2.

    The group order (the native tokens, then the contacts-and-distances-v1
    block, then the trailing sequence-only token, then the trailing
    retraction token, then the trailing retraction-mode doc type) and the
    within-group order are both load-bearing.
    """
    return [
        *contacts_v1_native_tokens(),
        *additional_tokens(),
        *sequence_only_tokens(),
        *retract_tokens(),
        *backtracking_tokens(),
    ]

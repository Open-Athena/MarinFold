# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical vocabulary for the contacts-and-coordinates-v1 format.

Local to this document-structure package. Imported by ``generate.py``,
``parse.py``, and ``cli.py``.

Order is **load-bearing**. Token IDs derived from this list must stay
stable for every checkpoint trained against this vocab. Append-only —
never reorder. (Reordering is a v2 event: new structure, new tokenizer,
new experiment.)

This format extends contacts-v1 (see ``SPEC.md``): it emits the identical
sequence section and the identical ``<contact>`` statements, plus a new
coordinate section built from a fresh ``<xyz-DDD>`` vocabulary. It carries
forward **contacts-v1's entire ``all_domain_tokens()`` list unchanged** —
so every amino acid, atom name (``<CA>`` …), position token, section
marker, and the ``<contact>`` token is reused by *emitting* it, not by
re-minting. A model trained on any of the three shared-vocab formats
(contacts-and-distances-v1, contacts-v1, this one) shares embeddings and
can be fine-tuned across them without a tokenizer change.

Unlike contacts-v1's own precedent — which puts its 5 native tokens
*before* the inherited contacts-and-distances-v1 block — this format puts
the **inherited contacts-v1 block first and its own 1001 native tokens
last**. That keeps every one of contacts-v1's token ids identical to what
it has in contacts-v1's own standalone tokenizer (both prepend
``<pad>``/``<eos>`` at ids 0-1, so contacts-v1's tokens occupy ids 2-2845
in either tokenizer). A pretrained contacts-v1 checkpoint's embedding
matrix can then be warm-started here by *appending* 1001 new rows rather
than remapping every existing embedding to a new id.

The 1001 native tokens are the doc-type token ``<contacts-and-coordinates-v1>``
followed by the 1000 coordinate tokens ``<xyz-000>`` .. ``<xyz-999>`` in
numeric order. Total domain vocab: 3845 tokens (3847 with ``<pad>``/``<eos>``).
"""

from marinfold.document_structures.contacts_and_distances_v1.vocab import (
    ATOM_NAMES as _CD_V1_ATOM_NAMES,
)
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_SEQUENCE_TOKEN as _CONTACTS_V1_BEGIN_SEQUENCE_TOKEN,
)
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_STRUCTURE_TOKEN as _CONTACTS_V1_BEGIN_STRUCTURE_TOKEN,
)
from marinfold.document_structures.contacts_v1.vocab import (
    CONTACT_TOKEN as _CONTACTS_V1_CONTACT_TOKEN,
)
from marinfold.document_structures.contacts_v1.vocab import (
    C_TERM_TOKEN as _CONTACTS_V1_C_TERM_TOKEN,
)
from marinfold.document_structures.contacts_v1.vocab import (
    END_TOKEN as _CONTACTS_V1_END_TOKEN,
)
from marinfold.document_structures.contacts_v1.vocab import (
    NUM_POSITION_INDICES as _CONTACTS_V1_NUM_POSITION_INDICES,
)
from marinfold.document_structures.contacts_v1.vocab import (
    N_TERM_TOKEN as _CONTACTS_V1_N_TERM_TOKEN,
)
from marinfold.document_structures.contacts_v1.vocab import (
    all_domain_tokens as _contacts_v1_all_domain_tokens,
)
from marinfold.document_structures.contacts_v1.vocab import (
    position_token as _contacts_v1_position_token,
)


NAME = "contacts-and-coordinates-v1"

# Larger than contacts-v1 / contacts-and-distances-v1's shared 8192:
# coordinates need far more room than contacts ever did. The token
# *vocabulary* is shared with those formats; the context length is not.
CONTEXT_LENGTH = 32768

# Residues are indexed into NUM_POSITION_INDICES position tokens with
# wrap-around (reused verbatim from contacts-v1). Caps the longest single
# chain we can serialize.
NUM_POSITION_INDICES = _CONTACTS_V1_NUM_POSITION_INDICES

# Number of coordinate tokens: one per (x, y, z) digit triple at a single
# decimal place, for all 1000 possible triples 000..999.
NUM_XYZ_TOKENS = 1000

# --- Tokens this format mints itself (no analog in any prior format) ---
DOC_TYPE_TOKEN = "<contacts-and-coordinates-v1>"


def xyz_token(triple: int) -> str:
    """Token for one (x, y, z) digit triple at a single decimal place.

    ``triple`` is the three digits read as a base-10 integer in
    ``[0, 999]``: the hundreds digit is x's digit at that place, the tens
    digit is y's, the ones digit is z's. So ``xyz_token(210)`` is the token
    whose x-digit is 2, y-digit is 1, z-digit is 0.
    """
    if not 0 <= triple < NUM_XYZ_TOKENS:
        raise ValueError(f"xyz triple {triple} out of range [0, {NUM_XYZ_TOKENS})")
    return f"<xyz-{triple:03d}>"


def xyz_token_for_digits(x_digit: int, y_digit: int, z_digit: int) -> str:
    """Token for the (x, y, z) digits at one decimal place."""
    return xyz_token(x_digit * 100 + y_digit * 10 + z_digit)


# All 1000 coordinate tokens, in numeric order (load-bearing).
XYZ_TOKENS = [xyz_token(i) for i in range(NUM_XYZ_TOKENS)]

# This format's own native tokens: the doc type, then the 1000 xyz tokens.
NATIVE_TOKENS = [DOC_TYPE_TOKEN, *XYZ_TOKENS]

# --- Tokens reused (emitted, not minted) from the inherited block ---
# Section markers + framing, carried forward from contacts-v1 (which in
# turn reuses contacts-and-distances-v1's underscore spellings).
BEGIN_SEQUENCE_TOKEN = _CONTACTS_V1_BEGIN_SEQUENCE_TOKEN
BEGIN_STRUCTURE_TOKEN = _CONTACTS_V1_BEGIN_STRUCTURE_TOKEN
END_TOKEN = _CONTACTS_V1_END_TOKEN
N_TERM_TOKEN = _CONTACTS_V1_N_TERM_TOKEN
C_TERM_TOKEN = _CONTACTS_V1_C_TERM_TOKEN
CONTACT_TOKEN = _CONTACTS_V1_CONTACT_TOKEN


def position_token(index: int) -> str:
    """Token for a residue position index — reused ``<pX>`` from contacts-v1."""
    return _contacts_v1_position_token(index)


def atom_token(atom_name: str) -> str:
    """Token for a heavy-atom name — reused ``<CA>`` / ``<CB>`` / … block."""
    return f"<{atom_name}>"


def inherited_tokens() -> list[str]:
    """The full contacts-v1 domain vocab, carried forward unchanged.

    This is contacts-v1's entire ``all_domain_tokens()`` list (its 5 native
    tokens, the contacts-and-distances-v1 block, and its trailing
    sequence-only token) — 2844 tokens — reused verbatim so every id is
    byte-stable against contacts-v1's own tokenizer.
    """
    return _contacts_v1_all_domain_tokens()


def native_tokens() -> list[str]:
    """The 1001 tokens this format mints: the doc type, then the xyz tokens."""
    return list(NATIVE_TOKENS)


def _validate_reuse() -> None:
    """Fail loudly if the inherited block drifted from what we emit.

    Guards against a contacts-v1 / contacts-and-distances-v1 rename silently
    turning a token this format *emits* into one it would have to mint.
    """
    inherited = set(inherited_tokens())
    reused = {
        BEGIN_SEQUENCE_TOKEN, BEGIN_STRUCTURE_TOKEN, END_TOKEN,
        N_TERM_TOKEN, C_TERM_TOKEN, CONTACT_TOKEN,
        position_token(0), position_token(NUM_POSITION_INDICES - 1),
        *(atom_token(a) for a in _CD_V1_ATOM_NAMES),
    }
    missing = reused - inherited
    if missing:
        raise ValueError(
            f"contacts-and-coordinates-v1 reuses tokens absent from the "
            f"inherited contacts-v1 block: {sorted(missing)}"
        )
    # The native block must be disjoint from the inherited one (we mint it,
    # nobody else does) so appending it introduces no duplicate ids.
    clashes = set(NATIVE_TOKENS) & inherited
    if clashes:
        raise ValueError(
            f"contacts-and-coordinates-v1 native tokens collide with the "
            f"inherited block: {sorted(clashes)}"
        )


_validate_reuse()


def all_domain_tokens() -> list[str]:
    """Return the full ordered contacts-and-coordinates-v1 domain vocabulary.

    ``build_tokenizer`` prepends the ``<pad>`` / ``<eos>`` specials (ids 0
    and 1); this function returns the domain tokens only, starting at id 2.

    The group order — the entire inherited contacts-v1 block first, then
    this format's 1001 native tokens (doc type, then ``<xyz-000>`` ..
    ``<xyz-999>``) — and the within-group order are both load-bearing.
    """
    return [*inherited_tokens(), *native_tokens()]

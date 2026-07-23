# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical vocabulary for the contacts-and-crops-v1 format.

Local to this document-structure package. Imported by ``generate.py``,
``parse.py``, and ``cli.py``.

Order is **load-bearing**. Token IDs derived from this list must stay
stable for every checkpoint trained against this vocab. Append-only —
never reorder. (Reordering is a v2 event: new structure, new tokenizer,
new experiment.)

This format is contacts-and-coordinates-v1 (ccoord) with a smaller,
bounded coordinate section that fits 8192 tokens (see ``SPEC.md``). It
emits the identical sequence section and ``<contact>`` statements, and
reuses ccoord's ``<xyz-DDD>`` vocabulary verbatim, adding only two native
tokens of its own: the doc-type ``<contacts-and-crops-v1>`` and ``<crop>``.

The token order is chosen so a checkpoint trained on **either** contacts-v1
**or** ccoord warm-starts by *appending* rows, with no id remapping:

- the entire contacts-v1 ``all_domain_tokens()`` list comes first, byte for
  byte, so every one of its ids (2-2845, after the ``<pad>``/``<eos>``
  specials) is identical to what it has in contacts-v1's own tokenizer;
- then the doc-type token (id 2846), then the 1000 ``<xyz-DDD>`` tokens
  (ids 2847-3846) — **exactly the ids ccoord gives them** (ccoord also puts
  its doc-type at 2846 and the xyz block at 2847-3846), so a ccoord
  checkpoint's xyz embeddings transfer at their own ids;
- then ``<crop>`` (id 3847) — the single genuinely new embedding row, the
  only token ccoord's tokenizer does not already have.

So ``contacts-and-crops-v1`` is literally ccoord's native block with one
``<crop>`` token appended. Total domain vocab: 3846 tokens (3848 with
``<pad>``/``<eos>``).
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


NAME = "contacts-and-crops-v1"

# The whole point of the format: a coordinate-bearing document that fits the
# same 8192-token context as the contacts formats (vs ccoord's 32768), so it
# can be trained cheaply and mixed 1:1 with them.
CONTEXT_LENGTH = 8192

# Residues are indexed into NUM_POSITION_INDICES position tokens with
# wrap-around (reused verbatim from contacts-v1). Caps the longest single
# chain we can serialize.
NUM_POSITION_INDICES = _CONTACTS_V1_NUM_POSITION_INDICES

# Number of coordinate tokens: one per (x, y, z) digit triple at a single
# decimal place, for all 1000 possible triples 000..999. Reused from ccoord.
NUM_XYZ_TOKENS = 1000

# --- Tokens this format mints itself (no analog in any prior format) ---
DOC_TYPE_TOKEN = "<contacts-and-crops-v1>"
# Opens a Pass-2 crop: `<crop> <xyz-HHH> <xyz-TTT>` names a 10 Å box exactly.
CROP_TOKEN = "<crop>"


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

# This format's own native tokens: the doc type, then ccoord's 1000 xyz
# tokens (at ccoord's own ids), then the single new `<crop>` token last.
NATIVE_TOKENS = [DOC_TYPE_TOKEN, *XYZ_TOKENS, CROP_TOKEN]

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
    """The 1002 tokens this format mints: doc type, the xyz tokens, ``<crop>``."""
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
            f"contacts-and-crops-v1 reuses tokens absent from the "
            f"inherited contacts-v1 block: {sorted(missing)}"
        )
    # The native block must be disjoint from the inherited one (we mint it,
    # nobody else does) so appending it introduces no duplicate ids.
    clashes = set(NATIVE_TOKENS) & inherited
    if clashes:
        raise ValueError(
            f"contacts-and-crops-v1 native tokens collide with the "
            f"inherited block: {sorted(clashes)}"
        )


_validate_reuse()


def all_domain_tokens() -> list[str]:
    """Return the full ordered contacts-and-crops-v1 domain vocabulary.

    ``build_tokenizer`` prepends the ``<pad>`` / ``<eos>`` specials (ids 0
    and 1); this function returns the domain tokens only, starting at id 2.

    The group order — the entire inherited contacts-v1 block first, then
    this format's native tokens (doc type, then ``<xyz-000>`` ..
    ``<xyz-999>``, then ``<crop>``) — and the within-group order are both
    load-bearing.
    """
    return [*inherited_tokens(), *native_tokens()]

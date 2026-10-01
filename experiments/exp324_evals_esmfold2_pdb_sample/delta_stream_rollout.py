"""Delta-stream V2 rollout serialization and vote parsing.

This module deliberately has no model backend dependency. The rollout worker
loads a native Levanter checkpoint, samples constrained token suffixes, then
uses this parser to turn each rollout into the same undirected pair-vote
matrices consumed by exp82/exp89.
"""

from collections.abc import Iterable

STOP_TOKEN_ID = 0
AA_BASE_TOKEN_ID = 1
DOC_START_TOKEN_ID = 21
CONTACTS_BEGIN_TOKEN_ID = 22
DOC_END_TOKEN_ID = 23
DELTA_BASE_TOKEN_ID = 32
MAX_ABS_DELTA = 1024
MIN_SEQ_SEPARATION = 6
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_TOKEN_ID = {aa: AA_BASE_TOKEN_ID + index for index, aa in enumerate(AA_ORDER)}
UNKNOWN_AA_TOKEN_ID = 24


def sequence_prefix(sequence: str) -> list[int]:
    """Encode a V2 sequence-conditioned contact prefix."""
    amino_acids = [AA_TO_TOKEN_ID.get(aa, UNKNOWN_AA_TOKEN_ID) for aa in sequence]
    return [DOC_START_TOKEN_ID, *amino_acids, CONTACTS_BEGIN_TOKEN_ID]


def delta_from_token(token_id: int) -> int:
    """Decode one signed nonzero delta token."""
    offset = token_id - DELTA_BASE_TOKEN_ID
    if not 0 <= offset < 2 * MAX_ABS_DELTA:
        raise ValueError(f"not a delta token: {token_id}")
    if offset < MAX_ABS_DELTA:
        return -(offset + 1)
    return offset - MAX_ABS_DELTA + 1


def parse_contact_suffix(tokens: Iterable[int], length: int, *, strict: bool = True) -> set[tuple[int, int]]:
    """Parse a V2 suffix into canonical undirected pairs.

    ``strict=False`` matches contacts-v1 rollout semantics: malformed statements
    and out-of-universe positions are ignored while valid pairs still vote. This
    matters because the canonical exp82 worker parses valid contact regex matches
    from unfinished or otherwise malformed generations rather than dropping the
    entire rollout.
    """
    pairs: set[tuple[int, int]] = set()
    residue = 0
    ended = False
    for token_id in tokens:
        if token_id == STOP_TOKEN_ID:
            residue += 1
            if residue > length:
                if strict:
                    raise ValueError("suffix has more STOP segments than residues")
                break
            continue
        if token_id == DOC_END_TOKEN_ID:
            if strict and residue != length:
                raise ValueError(f"DOC_END after {residue}/{length} segments")
            ended = True
            break
        if residue >= length:
            if strict:
                raise ValueError("delta after final STOP segment")
            continue
        try:
            delta = delta_from_token(token_id)
        except ValueError:
            if strict:
                raise
            continue
        partner = residue + delta
        if not 0 <= partner < length:
            if strict:
                raise ValueError(f"out-of-range delta {delta} at residue {residue}")
            continue
        if abs(partner - residue) >= MIN_SEQ_SEPARATION:
            pairs.add((min(residue, partner), max(residue, partner)))
    if strict and not ended:
        raise ValueError("suffix did not terminate with DOC_END")
    return pairs

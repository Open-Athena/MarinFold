"""Strict statement grammar shared by corpus construction and inference.

Histories contain repeated hypothesis sections. FINAL abandons the current
hypothesis and starts synthesis; it is legal after any complete contact triple.
END terminates only the final answer. Parsing never silently salvages malformed
generated documents. Budget truncation is the one deliberate prefix operation.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

BEGIN = "<begin_statements>"
CONTACT = "<contact>"
END = "<end>"
FINAL = "<final-prediction>"
MULTI = "<contacts-v1.multi>"
POSITION = re.compile(r"<p(\d+)>")


@dataclass(frozen=True)
class ParsedHistory:
    """Parsed token offsets and position-token pairs, before ring decoding."""

    hypotheses: tuple[tuple[tuple[int, int], ...], ...]
    final: tuple[tuple[int, int], ...] | None
    final_index: int | None
    finished: bool
    boundaries: tuple[int, ...]


def parse_history(tokens: Sequence[str], *, allow_prefix: bool = False) -> ParsedHistory:
    """Parse a history starting with BEGIN or FINAL, rejecting broken triples.

    Args:
        tokens: Whitespace-independent vocabulary tokens, excluding the sequence.
        allow_prefix: Accept a statement-aligned prefix without FINAL/END.
    """
    sections: list[list[tuple[int, int]]] = []
    final: list[tuple[int, int]] | None = None
    final_index = None
    boundaries = [0]
    i = 0
    finished = False
    while i < len(tokens):
        token = tokens[i]
        if token == BEGIN and final is None:
            if sections and not sections[-1]:
                raise ValueError("empty hypothesis section")
            sections.append([])
            i += 1
        elif token == FINAL and final is None:
            if sections and not sections[-1]:
                raise ValueError("final marker after empty hypothesis section")
            final_index = i
            final = []
            i += 1
        elif token == END and final is not None and i == len(tokens) - 1:
            finished = True
            i += 1
        elif token == CONTACT and (sections or final is not None):
            if i + 2 >= len(tokens):
                raise ValueError("incomplete contact triple")
            matches = [POSITION.fullmatch(t) for t in tokens[i + 1:i + 3]]
            if any(m is None for m in matches):
                raise ValueError("contact must be followed by two position tokens")
            pair = tuple(int(m.group(1)) for m in matches if m is not None)
            if any(p >= 2000 for p in pair):
                raise ValueError("position outside contacts-v1 ring")
            destination = final if final is not None else sections[-1]
            destination.append((pair[0], pair[1]))
            i += 3
            if final is None:
                boundaries.append(i)
        else:
            raise ValueError(f"unexpected token {token!r} at history offset {i}")
    if not allow_prefix and not finished:
        raise ValueError("missing final prediction or end marker")
    if sections and not sections[-1]:
        raise ValueError("empty trailing hypothesis")
    return ParsedHistory(tuple(tuple(s) for s in sections),
                         None if final is None else tuple(final),
                         final_index, finished, tuple(boundaries))


def truncate_history(tokens: Sequence[str], budget: int) -> list[str]:
    """Keep the longest valid hypothesis prefix within a token budget.

    A trailing partial triple or section marker is discarded only because it
    crosses the requested budget. Malformed tokens inside the retained budget
    raise. A naturally emitted FINAL ends the hypothesis prefix early.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    prefix: list[str] = []
    i = 0
    last_boundary = 0
    while i < min(budget, len(tokens)):
        if tokens[i] == FINAL:
            break
        if tokens[i] == BEGIN:
            prefix.append(BEGIN)
            i += 1
        elif tokens[i] == CONTACT:
            if i + 3 > min(budget, len(tokens)):
                break
            prefix.extend(tokens[i:i + 3])
            i += 3
            last_boundary = len(prefix)
        else:
            raise ValueError(f"invalid hypothesis token {tokens[i]!r}")
    result = prefix[:last_boundary]
    parse_history(result, allow_prefix=True)
    return result

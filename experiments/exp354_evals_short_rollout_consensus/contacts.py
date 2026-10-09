"""Exact emitted-contact cutoffs, including retractions and premature EOS.

The budget counts complete emitted contact statements, including duplicates
and invalid/out-of-sequence pairs. Votes count unique live valid pairs. This
keeps the generation stopping rule independent of evaluation ground truth.
"""

import re

POSITION = re.compile(r"<p(\d+)>")


def limits(length: int) -> dict[str, int]:
    """Return the five requested contact budgets."""
    return {"5": 5, "10": 10, "20": 20, "L5": max(1, length // 5), "L2": max(1, length // 2)}


def snapshots(tokens: list[str], budgets: list[int]) -> tuple[dict[int, dict], int]:
    """Return live position pairs and token counts at each exact cutoff."""
    wanted = set(budgets)
    found: dict[int, dict] = {}
    live: set[tuple[int, int]] = set()
    emitted = 0
    cursor = 0
    end = len(tokens)
    while cursor < len(tokens):
        kind = tokens[cursor]
        if kind == "<end>":
            end = cursor + 1
            break
        if kind in {"<contact>", "<retract>"} and cursor + 2 < len(tokens):
            a = POSITION.fullmatch(tokens[cursor + 1])
            b = POSITION.fullmatch(tokens[cursor + 2])
            if a is not None and b is not None:
                pair = tuple(sorted((int(a[1]), int(b[1]))))
                if kind == "<contact>":
                    live.add(pair)
                    emitted += 1
                else:
                    live.discard(pair)
                cursor += 3
                if emitted in wanted and emitted not in found:
                    found[emitted] = {"pairs": frozenset(live), "tokens": cursor, "emitted": emitted}
                continue
        cursor += 1
    for budget in wanted - found.keys():
        found[budget] = {"pairs": frozenset(live), "tokens": end, "emitted": emitted}
    return found, emitted


def sequence_pairs(pairs: frozenset, n_term: int, length: int) -> set[tuple[int, int]]:
    """Map ring positions to sequence pairs, applying the fixed separation cut."""
    result = set()
    for a, b in pairs:
        i, j = sorted(((a - n_term) % 2000, (b - n_term) % 2000))
        if i < length and j < length and j - i >= 6:
            result.add((i, j))
    return result

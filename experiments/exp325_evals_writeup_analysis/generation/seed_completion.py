"""Truth-independent selection and prompt rendering for seeded contact completion."""

import hashlib
import re

import numpy as np

BUDGETS = ("0", "5", "10", "L5")
CASES = [("0", 0)] + [(budget, replicate) for budget in BUDGETS[1:] for replicate in range(2)]
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")


def render_seeds(pairs: list[list[int]], n_term: int, rollout: int) -> str:
    """Randomize seed order/orientation and express sequence indices on the token ring."""
    rng = np.random.default_rng(325_000 + rollout)
    statements = []
    for index in rng.permutation(len(pairs)):
        i, j = pairs[index]
        if rng.integers(2):
            i, j = j, i
        statements.append(f"<contact> <p{(n_term + i) % 2000}> <p{(n_term + j) % 2000}>")
    return " " + " ".join(statements) if statements else ""


def parse_pairs(text: str, position_map: dict[int, int]) -> set[tuple[int, int]]:
    """Decode unique valid sequence pairs, with the training separation cutoff."""
    result = set()
    for a, b in CONTACT_RE.findall(text):
        i, j = position_map.get(int(a)), position_map.get(int(b))
        if i is not None and j is not None and abs(i - j) >= 6:
            result.add(tuple(sorted((i, j))))
    return result


def complete_pairs(seeds: list[list[int]], votes: list[list[int]], mapping: dict[int, int],
                   total: int) -> list[tuple[int, int, int]]:
    """Choose new pairs by vote only; retain seeds and require an exact total budget.

    Inputs contain no ground-truth contact labels. Candidates must map to resolved
    residues and satisfy separation >=6 in both coordinate systems. Zero-vote
    pairs are never invented to fill a shortage. Equal votes break by (i,j).
    """
    given = {tuple(sorted(pair)) for pair in seeds}
    if len(given) != len(seeds) or len(given) > total:
        raise ValueError("Invalid seed contact budget")
    if any(i not in mapping or j not in mapping or j - i < 6 or abs(mapping[j] - mapping[i]) < 6
           for i, j in given):
        raise ValueError("Seed pair cannot be mapped")
    eligible = [(int(i), int(j), int(count)) for i, j, count in votes
                if count > 0 and (i, j) not in given and i in mapping and j in mapping
                and j - i >= 6 and abs(mapping[j] - mapping[i]) >= 6]
    if len({(i, j) for i, j, _ in eligible}) != len(eligible):
        raise ValueError("Duplicate vote pairs")
    ordered = sorted(eligible, key=lambda row: (-row[2], row[0], row[1]))
    needed = total - len(given)
    if len(ordered) < needed:
        raise ValueError(f"Only {len(ordered)} voted pairs for {needed} requested additions")
    return ordered[:needed]


def state_digest(state: np.ndarray) -> str:
    """Identify the exact three-state Helico conditioning bytes."""
    return hashlib.sha256(state.tobytes()).hexdigest()

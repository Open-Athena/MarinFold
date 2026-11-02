"""Deterministic random alanine masks for contacts-v1 prompts."""

import hashlib
import math
import random
from collections.abc import Sequence

AA_TOKENS = {
    f"<{name}>"
    for name in (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "UNK",
    )
}


def stable_seed(value: str) -> int:
    """Return a reproducible positive 31-bit seed."""
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") & 0x7FFFFFFF


def mutation_count(n_candidates: int, fraction: float) -> int:
    """Return the exact mask size for a fraction of non-alanine residues."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"mutation fraction must lie in [0, 1], got {fraction}")
    if n_candidates == 0 or fraction == 0.0:
        return 0
    return min(n_candidates, max(1, math.floor(fraction * n_candidates + 0.5)))


def ranked_non_alanine_positions(sequence: str, stem: str, rollout: int) -> list[int]:
    """Rank mutable positions once so masks are nested across fractions."""
    positions = [index for index, residue in enumerate(sequence) if residue != "A"]
    random.Random(stable_seed(f"{stem}:r{rollout}:alanine-mask")).shuffle(positions)
    return positions


def alanine_mask(
    sequence: str, stem: str, rollout: int, fraction: float
) -> tuple[str, list[int]]:
    """Replace an exact deterministic random subset of non-A residues with A."""
    ranked = ranked_non_alanine_positions(sequence, stem, rollout)
    selected = sorted(ranked[:mutation_count(len(ranked), fraction)])
    mutated = list(sequence)
    for index in selected:
        mutated[index] = "A"
    return "".join(mutated), selected


def validate_nested_masks(
    sequence: str, masks: Sequence[tuple[float, Sequence[int]]]
) -> None:
    """Require exact, non-alanine, nested masks in increasing fraction order."""
    previous: set[int] = set()
    previous_fraction = -1.0
    n_candidates = sum(residue != "A" for residue in sequence)
    for fraction, raw_positions in masks:
        positions = {int(value) for value in raw_positions}
        if fraction < previous_fraction:
            raise ValueError("masks are not sorted by mutation fraction")
        if len(positions) != mutation_count(n_candidates, fraction):
            raise ValueError("mask size does not match mutation fraction")
        if any(sequence[index] == "A" for index in positions):
            raise ValueError("native alanine included in mutation mask")
        if not previous <= positions:
            raise ValueError("mutation masks are not nested")
        previous = positions
        previous_fraction = fraction


def validate_prompt_mutation(
    native_prefix: str, mutated_prefix: str, expected_mutations: int
) -> None:
    """Require two prompts to differ only at the requested amino-acid slots."""
    native_tokens = native_prefix.split()
    mutated_tokens = mutated_prefix.split()
    if len(native_tokens) != len(mutated_tokens):
        raise ValueError("alanine masking changed prompt token count")
    changed = 0
    for native, mutated in zip(native_tokens, mutated_tokens, strict=True):
        if native == mutated:
            continue
        changed += 1
        if native not in AA_TOKENS or mutated != "<ALA>":
            raise ValueError("alanine masking changed a non-amino-acid prompt token")
    if changed != expected_mutations:
        raise ValueError(f"expected {expected_mutations} prompt mutations, found {changed}")

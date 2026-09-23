"""Pure policy helpers for sequence-contrastive rollout guidance."""

import hashlib
import random
from collections.abc import Sequence


AA_THREE = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
)


def stable_seed(value: str) -> int:
    """Return a reproducible 31-bit seed independent of Python hash state."""
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") & 0x7FFFFFFF


def null_sequence(sequence: str, kind: str, stem: str) -> str:
    """Build a same-length null sequence for one target.

    The shuffled null is fixed per protein rather than per document realization,
    so document resampling does not introduce an extra background-sequence
    nuisance.
    """
    if kind == "polyala":
        return "A" * len(sequence)
    if kind == "polylys":
        return "K" * len(sequence)
    if kind == "shuffle":
        values = list(sequence)
        random.Random(stable_seed(f"{stem}:composition-null")).shuffle(values)
        return "".join(values)
    raise ValueError(f"unknown null kind: {kind}")


def validate_prompt_pair(
    native_ids: Sequence[int],
    null_ids: Sequence[int],
    amino_acid_ids: set[int],
    expected_residues: int,
) -> None:
    """Require paired prompts to differ only at amino-acid token slots."""
    if len(native_ids) != len(null_ids):
        raise ValueError("native and null prompts have different token lengths")
    changed = 0
    aa_slots = 0
    for native, background in zip(native_ids, null_ids):
        native_is_aa = native in amino_acid_ids
        background_is_aa = background in amino_acid_ids
        if native_is_aa:
            aa_slots += 1
        if native != background:
            changed += 1
            if not native_is_aa or not background_is_aa:
                raise ValueError("paired prompts differ outside an amino-acid slot")
    if aa_slots != expected_residues:
        raise ValueError(
            f"expected {expected_residues} amino-acid slots, found {aa_slots}"
        )
    if changed == 0 and len(set(native_ids) & amino_acid_ids) > 1:
        raise ValueError("non-degenerate native prompt unexpectedly equals its null")


def expects_position_token(
    generated: Sequence[int], contact_id: int, position_ids: set[int]
) -> bool:
    """Whether the next token is an endpoint of a valid contact statement."""
    if generated and generated[-1] == contact_id:
        return True
    return (
        len(generated) >= 2
        and generated[-2] == contact_id
        and generated[-1] in position_ids
    )


def parse_contacts(
    generated: Sequence[int],
    token_log_ratios: Sequence[float],
    contact_id: int,
    position_to_index: dict[int, int],
    minimum_separation: int = 6,
) -> tuple[list[list[int]], list[float], int]:
    """Parse first-occurrence contacts and their three-token log-ratios."""
    contacts: list[list[int]] = []
    log_ratios: list[float] = []
    seen: set[tuple[int, int]] = set()
    malformed = 0
    index = 0
    while index < len(generated):
        token = generated[index]
        if token != contact_id:
            index += 1
            continue
        if index + 2 >= len(generated):
            malformed += 1
            break
        left = position_to_index.get(generated[index + 1])
        right = position_to_index.get(generated[index + 2])
        if left is None or right is None:
            malformed += 1
            index += 1
            continue
        pair = (min(left, right), max(left, right))
        if left != right and abs(left - right) >= minimum_separation and pair not in seen:
            seen.add(pair)
            contacts.append([pair[0], pair[1]])
            log_ratios.append(float(sum(token_log_ratios[index:index + 3])))
        index += 3
    return contacts, log_ratios, malformed


def native_top_p_mask(logits, top_p: float):
    """Return the ordinary nucleus set for each row of a torch logit tensor."""
    import torch

    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_p == 1:
        return torch.ones_like(logits, dtype=torch.bool)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    remove = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
    keep_sorted = ~remove
    keep = torch.zeros_like(keep_sorted).scatter(-1, sorted_indices, keep_sorted)
    return keep


def guided_sample_logits(
    native_logits,
    null_logits,
    guide_rows,
    *,
    gamma: float,
    top_p: float,
    temperature: float,
    pure_ratio: bool,
):
    """Combine paired logits inside the native stream's plausibility set."""
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    native = native_logits.float() / temperature
    background = null_logits.float() / temperature
    keep = native_top_p_mask(native, top_p)
    contrastive = native - background
    proposed = contrastive if pure_ratio else native + gamma * contrastive
    combined = torch.where(guide_rows[:, None], proposed, native)
    return combined.masked_fill(~keep, float("-inf"))

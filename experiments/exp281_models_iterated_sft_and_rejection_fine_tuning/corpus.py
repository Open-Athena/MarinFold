"""Build weighted examples and select whole trajectories without label leakage."""

import random
from dataclasses import dataclass
from typing import Any, Sequence

from marinfold.document_structures.contacts_v1_multi import BEGIN, END, FINAL, MULTI, parse_history
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class LossProfile:
    """Weights apply to target tokens, before the trainer's causal shift."""

    hypothesis: float = 0.1
    marker: float = 1.0
    final: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.hypothesis <= 1 or self.marker <= 0 or self.final <= 0:
            raise ValueError("invalid loss profile")


def tokenize_exact(tokenizer: PreTrainedTokenizerBase, tokens: Sequence[str]) -> list[int]:
    """Require one vocabulary id per document token, including the final marker."""
    ids = tokenizer.encode(" ".join(tokens), add_special_tokens=False)
    if tokenizer.convert_ids_to_tokens(ids) != list(tokens):
        raise ValueError("document/tokenizer mismatch or unknown token")
    return ids


def build_example(
    *, header: Sequence[str], history: Sequence[str], reference: Sequence[str],
    forced: bool, profile: LossProfile, tokenizer: PreTrainedTokenizerBase,
    context: int, plain: bool = False,
) -> dict[str, Any]:
    """Replace the sampled final answer with reference contacts and build masks."""
    parse_history(history, allow_prefix=True)
    if FINAL in history or END in history:
        raise ValueError("history must contain hypotheses only")
    parse_history([FINAL, *reference, END])
    if not header or header[0] not in (MULTI, "<contacts-v1>"):
        raise ValueError("invalid document header")
    if plain and history:
        raise ValueError("plain control cannot contain hypotheses")
    head = ["<contacts-v1>" if plain else MULTI, *header[1:]]
    marker = BEGIN if plain else FINAL
    tokens = [*head, *history, marker, *reference, END]
    if len(tokens) > context:
        raise ValueError("reference answer does not fit; do not truncate labels")
    marker_weight = profile.marker if not forced or plain else 0.0
    weights = ([0.0] * len(head) + [profile.hypothesis] * len(history)
               + [marker_weight] + [profile.final] * (len(reference) + 1))
    return {"input_ids": tokenize_exact(tokenizer, tokens), "loss_weights": weights,
            "hypothesis_weight": profile.hypothesis * len(history),
            "final_weight": profile.final * (len(reference) + 1),
            "marker_weight": marker_weight, "forced": forced, "plain": plain}


def contact_score(predicted: Sequence[Sequence[int]], reference: Sequence[Sequence[int]]) -> dict[str, float]:
    """Score undirected unique contacts, with zero F1 for an empty prediction."""
    pred = {tuple(sorted(p)) for p in predicted}
    truth = {tuple(sorted(p)) for p in reference}
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (len(pred) + len(truth)) if pred or truth else 0.0}


def select_candidate(candidates: Sequence[dict[str, Any]], strategy: str, seed: int) -> dict[str, Any]:
    """Select by final-answer F1, breaking ties randomly without length preference.

    Invalid candidates have score -1. Random selection samples the same valid
    pool as best selection; invalid-rate statistics are retained independently.
    """
    if strategy not in ("best", "random"):
        raise ValueError("selection must be best or random")
    if not candidates:
        raise ValueError("empty candidate pool")
    keys = {(c["target_id"], c["forced"], c["budget"], tuple(c["header"])) for c in candidates}
    if len(keys) != 1:
        raise ValueError("candidate comparison must share target, prompt, mode, and budget")
    valid = [c for c in candidates if c["valid"]]
    if not valid:
        raise ValueError("all candidate trajectories are invalid")
    if strategy == "best":
        best = max(c["score"]["f1"] for c in valid)
        valid = [c for c in valid if c["score"]["f1"] == best]
    return random.Random(seed).choice(valid)

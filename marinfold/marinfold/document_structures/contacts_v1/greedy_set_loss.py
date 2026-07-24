# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Greedy latent-order contact-set loss prototype.

This module is intentionally small and model-agnostic. It does not run a
transformer. Instead, it defines pair-slot target selection for a training loop
that can supply teacher-forced next-token log probabilities.

The objective is a hard/Viterbi relaxation of contacts-v1's arbitrary contact
ordering and pair orientation. At each contact slot, any remaining unordered
contact pair is valid. The model chooses the easiest remaining pair as a whole,
with pair orientation treated as latent.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactBlockTargets:
    """Parsed contacts-v1 structure block from one tokenized document."""

    begin_position: int
    end_position: int
    slot_positions: np.ndarray
    pairs: np.ndarray


@dataclass(frozen=True)
class MatchedPairChoice:
    """One pair-slot assignment chosen by greedy matching."""

    slot_index: int
    pair_index: int
    pair: tuple[int, int]
    oriented_tokens: tuple[int, int]
    log_prob: float


@dataclass(frozen=True)
class MatchedContactBlockLoss:
    """Loss details for the single-pass pair-block matching prototype."""

    loss: float
    prefix_loss: float
    pair_loss: float
    end_loss: float
    choices: tuple[MatchedPairChoice, ...]


def parse_contact_block_targets(
    token_ids: np.ndarray,
    *,
    begin_statements_token_id: int,
    contact_token_id: int,
    end_token_id: int,
    position_token_ids: np.ndarray,
) -> ContactBlockTargets:
    """Parse contacts-v1 ``<begin_statements>`` triples from one token row."""
    token_ids = np.asarray(token_ids)
    begin_matches = np.flatnonzero(token_ids == begin_statements_token_id)
    if begin_matches.size != 1:
        raise ValueError(f"expected exactly one <begin_statements>, got {begin_matches.size}")
    begin_position = int(begin_matches[0])

    end_matches = np.flatnonzero(token_ids[begin_position + 1 :] == end_token_id)
    if end_matches.size == 0:
        raise ValueError("structure block has no <end> token")
    end_position = begin_position + 1 + int(end_matches[0])

    token_to_position = {int(token_id): pos for pos, token_id in enumerate(position_token_ids)}
    slot_positions: list[tuple[int, int, int]] = []
    pairs: list[tuple[int, int]] = []
    cursor = begin_position + 1
    while cursor < end_position:
        if cursor + 2 >= end_position:
            raise ValueError("structure block ended inside a contact triple")
        if int(token_ids[cursor]) != contact_token_id:
            raise ValueError(f"expected <contact> at token position {cursor}")
        left_token = int(token_ids[cursor + 1])
        right_token = int(token_ids[cursor + 2])
        if left_token not in token_to_position or right_token not in token_to_position:
            raise ValueError(f"contact at token position {cursor} does not use <p*> endpoints")
        left = token_to_position[left_token]
        right = token_to_position[right_token]
        if left == right:
            raise ValueError(f"self-contact at token position {cursor}: <p{left}>")
        slot_positions.append((cursor, cursor + 1, cursor + 2))
        pairs.append(tuple(sorted((left, right))))
        cursor += 3

    if len(set(pairs)) != len(pairs):
        raise ValueError("structure block contains duplicate unordered contact pairs")
    return ContactBlockTargets(
        begin_position=begin_position,
        end_position=end_position,
        slot_positions=np.asarray(slot_positions, dtype=np.int64).reshape((-1, 3)),
        pairs=np.asarray(pairs, dtype=np.int64).reshape((-1, 2)),
    )


def greedy_matched_contact_block_loss(
    log_probs: np.ndarray,
    token_ids: np.ndarray,
    *,
    begin_statements_token_id: int,
    contact_token_id: int,
    end_token_id: int,
    position_token_ids: np.ndarray,
) -> MatchedContactBlockLoss:
    """Single-pass contacts-v1 loss with greedy order/orientation matching.

    ``log_probs[position]`` predicts ``token_ids[position + 1]``. Prefix tokens
    through ``<begin_statements>`` and the final ``<end>`` are exact CE. Contact
    triples are matched greedily to unused expected pairs, ignoring serialized
    order and pair orientation.
    """
    targets = parse_contact_block_targets(
        token_ids,
        begin_statements_token_id=begin_statements_token_id,
        contact_token_id=contact_token_id,
        end_token_id=end_token_id,
        position_token_ids=position_token_ids,
    )
    log_probs = np.asarray(log_probs)
    token_ids = np.asarray(token_ids)
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must have shape [position, vocab], got {log_probs.shape}")
    if log_probs.shape[0] != token_ids.shape[0]:
        raise ValueError("log_probs and token_ids must have the same position length")

    prefix_loss = -float(
        np.sum(log_probs[np.arange(targets.begin_position), token_ids[1 : targets.begin_position + 1]])
    )
    pair_scores = _pair_score_matrix(
        log_probs,
        targets=targets,
        contact_token_id=contact_token_id,
        position_token_ids=position_token_ids,
    )

    remaining = np.ones(targets.pairs.shape[0], dtype=bool)
    pair_loss = 0.0
    choices: list[MatchedPairChoice] = []
    for slot_index in range(targets.slot_positions.shape[0]):
        masked_scores = np.where(remaining, pair_scores[slot_index], -np.inf)
        pair_index = int(np.argmax(masked_scores))
        if not np.isfinite(masked_scores[pair_index]):
            raise ValueError("no remaining contact pair to match")
        remaining[pair_index] = False
        score = float(pair_scores[slot_index, pair_index])
        pair_loss -= score
        left_token, right_token = _best_orientation_tokens(
            log_probs,
            slot_positions=targets.slot_positions[slot_index],
            pair=targets.pairs[pair_index],
            position_token_ids=position_token_ids,
        )
        choices.append(
            MatchedPairChoice(
                slot_index=slot_index,
                pair_index=pair_index,
                pair=tuple(int(x) for x in targets.pairs[pair_index]),
                oriented_tokens=(left_token, right_token),
                log_prob=score,
            )
        )

    end_loss = -float(log_probs[targets.end_position - 1, end_token_id])
    return MatchedContactBlockLoss(
        loss=prefix_loss + pair_loss + end_loss,
        prefix_loss=prefix_loss,
        pair_loss=pair_loss,
        end_loss=end_loss,
        choices=tuple(choices),
    )


def _pair_score_matrix(
    log_probs: np.ndarray,
    *,
    targets: ContactBlockTargets,
    contact_token_id: int,
    position_token_ids: np.ndarray,
) -> np.ndarray:
    """Return ``[slot, pair]`` log-prob scores with best orientation."""
    n_slots = targets.slot_positions.shape[0]
    n_pairs = targets.pairs.shape[0]
    scores = np.empty((n_slots, n_pairs), dtype=np.float64)
    for slot_index, (marker_pos, left_pos, right_pos) in enumerate(targets.slot_positions):
        contact_score = float(log_probs[marker_pos - 1, contact_token_id])
        for pair_index, (first, second) in enumerate(targets.pairs):
            first_token = int(position_token_ids[first])
            second_token = int(position_token_ids[second])
            forward = float(log_probs[left_pos - 1, first_token] + log_probs[right_pos - 1, second_token])
            reverse = float(log_probs[left_pos - 1, second_token] + log_probs[right_pos - 1, first_token])
            scores[slot_index, pair_index] = contact_score + max(forward, reverse)
    return scores


def _best_orientation_tokens(
    log_probs: np.ndarray,
    *,
    slot_positions: np.ndarray,
    pair: np.ndarray,
    position_token_ids: np.ndarray,
) -> tuple[int, int]:
    _, left_pos, right_pos = slot_positions
    first_token = int(position_token_ids[int(pair[0])])
    second_token = int(position_token_ids[int(pair[1])])
    forward = float(log_probs[left_pos - 1, first_token] + log_probs[right_pos - 1, second_token])
    reverse = float(log_probs[left_pos - 1, second_token] + log_probs[right_pos - 1, first_token])
    return (first_token, second_token) if forward >= reverse else (second_token, first_token)


__all__ = [
    "ContactBlockTargets",
    "MatchedContactBlockLoss",
    "MatchedPairChoice",
    "greedy_matched_contact_block_loss",
    "parse_contact_block_targets",
]

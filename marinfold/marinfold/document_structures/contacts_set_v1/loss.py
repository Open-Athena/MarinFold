# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Target-anchored slot-assignment helpers for contacts-set-v1.

These helpers are pure Python/Numpy reference code for tests and experiment
prototyping.  Training code can port the same objective to JAX.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Assignment:
    """Result of target-anchored slot assignment.

    ``pairs`` contains ``(target_index, slot_index)`` entries.  Every target row
    is assigned exactly once; assigned slot indices are unique.  ``extra_slots``
    are predicted slots not used to explain a target and should receive the
    separate empty-slot / extra-contact penalty.
    """

    pairs: tuple[tuple[int, int], ...]
    total_cost: float
    extra_slots: tuple[int, ...]


def target_anchored_slot_assignment(cost: np.ndarray) -> Assignment:
    """Assign each target row to one unique predicted slot with minimum cost.

    Args:
        cost: ``[num_targets, num_predicted_slots]`` matrix.  ``num_targets``
            must be no larger than ``num_predicted_slots``.

    Returns:
        A globally minimum-cost target-anchored assignment.  Extra predicted
        slots are reported but do not affect ``total_cost``.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"cost must be a 2D matrix, got shape {matrix.shape}")
    num_targets, num_slots = matrix.shape
    if num_targets > num_slots:
        raise ValueError(f"cannot assign {num_targets} targets to {num_slots} slots")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cost matrix must be finite")
    if num_targets == 0:
        return Assignment(pairs=(), total_cost=0.0, extra_slots=tuple(range(num_slots)))

    # Dynamic programming over used-slot bitmasks.  K=16 in the format, so this
    # exact solver is tiny and avoids adding scipy just for the reference tests.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for target_idx in range(num_targets):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (prev_cost, prev_slots) in states.items():
            for slot_idx in range(num_slots):
                bit = 1 << slot_idx
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_cost = prev_cost + float(matrix[target_idx, slot_idx])
                current = next_states.get(new_mask)
                if current is None or new_cost < current[0]:
                    next_states[new_mask] = (new_cost, (*prev_slots, slot_idx))
        states = next_states

    best_mask, (best_cost, best_slots) = min(states.items(), key=lambda item: item[1][0])
    pairs = tuple((target_idx, slot_idx) for target_idx, slot_idx in enumerate(best_slots))
    extra_slots = tuple(slot_idx for slot_idx in range(num_slots) if not (best_mask & (1 << slot_idx)))
    return Assignment(pairs=pairs, total_cost=float(best_cost), extra_slots=extra_slots)


def sigmoid_binary_cross_entropy(logit: float, target: bool) -> float:
    """Numerically stable scalar BCE-with-logits."""
    y = 1.0 if target else 0.0
    x = float(logit)
    return max(x, 0.0) - x * y + float(np.log1p(np.exp(-abs(x))))


def softmax_cross_entropy(logits: np.ndarray, target_index: int) -> float:
    """Numerically stable scalar softmax cross-entropy."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"logits must be 1D, got shape {values.shape}")
    if not 0 <= target_index < values.shape[0]:
        raise ValueError(f"target_index out of range: {target_index}")
    shifted = values - np.max(values)
    log_z = float(np.log(np.exp(shifted).sum()))
    return log_z - float(shifted[target_index])

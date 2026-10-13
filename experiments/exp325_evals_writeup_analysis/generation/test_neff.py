"""CUDA Neff must reproduce the established overlap identity definition."""

import numpy as np
import pytest
import torch

from measure_neff import audit_counts, neighbor_counts
from msa_depth import a3m_to_match_array, compute_neff


@pytest.mark.parametrize("sequences", [
    ["AAAAA", "AAAAA", "AAAAC", "CCCCC"],
    ["A----", "AAAAA", "----A", "CCCCC"],
    ["ACdEFG", "ACeEFG", "ACEFG", "AC---", "---FG"],
])
def test_neighbors_match_reference_with_duplicates_gaps_and_threshold(sequences: list[str]) -> None:
    torch.set_float32_matmul_precision("high")
    arr = a3m_to_match_array(sequences)
    counts = neighbor_counts(arr, block_size=2)
    audit_counts(arr, counts)
    assert np.sum(1.0 / counts) == pytest.approx(compute_neff(arr, threshold=0.8), abs=1e-12)


def test_random_alignment_matches_reference() -> None:
    rng = np.random.default_rng(325)
    arr = rng.choice(np.frombuffer(b"AAAAACDEFG-", dtype=np.uint8), (71, 37))
    arr[10:20] = arr[0]
    counts = neighbor_counts(arr, block_size=11)
    assert audit_counts(arr, counts) == 8
    assert np.sum(1.0 / counts) == pytest.approx(compute_neff(arr, threshold=0.8), abs=1e-12)

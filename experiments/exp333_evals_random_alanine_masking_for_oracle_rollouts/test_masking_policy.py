"""Tests for deterministic alanine masking."""

import pytest

from masking_policy import (
    alanine_mask,
    mutation_count,
    validate_nested_masks,
    validate_prompt_mutation,
)


def test_exact_nested_masks_exclude_native_alanine() -> None:
    sequence = "ACDEFGHIKLMNPQRSTVWYA"
    fractions = [0.0, 0.05, 0.10, 0.20, 0.40, 1.0]
    masks = []
    for fraction in fractions:
        mutated, positions = alanine_mask(sequence, "target", 7, fraction)
        assert sum(left != right for left, right in zip(sequence, mutated, strict=True)) == len(
            positions
        )
        assert all(sequence[index] != "A" and mutated[index] == "A" for index in positions)
        masks.append((fraction, positions))
    validate_nested_masks(sequence, masks)


def test_masks_are_deterministic_but_vary_by_rollout() -> None:
    sequence = "CDEFGHIKLMNPQRSTVWY" * 4
    first = alanine_mask(sequence, "target", 3, 0.2)
    assert first == alanine_mask(sequence, "target", 3, 0.2)
    assert first[1] != alanine_mask(sequence, "target", 4, 0.2)[1]


def test_mutation_count_rounding_and_validation() -> None:
    assert mutation_count(19, 0.05) == 1
    assert mutation_count(19, 0.10) == 2
    assert mutation_count(19, 1.0) == 19
    assert mutation_count(0, 1.0) == 0
    with pytest.raises(ValueError):
        mutation_count(10, 1.01)


def test_prompt_validation_rejects_non_amino_acid_changes() -> None:
    native = "<p3> <CYS> <p4> <ASP> <begin_statements>"
    mutated = "<p3> <ALA> <p4> <ASP> <begin_statements>"
    validate_prompt_mutation(native, mutated, 1)
    with pytest.raises(ValueError):
        validate_prompt_mutation(native, mutated.replace("<p4>", "<p5>"), 1)

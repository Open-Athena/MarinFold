"""Scientific invariants for contact-seeded continuation and vote selection."""

import pytest

from seed_completion import complete_pairs, parse_pairs, render_seeds


def test_seed_prompt_round_trip_across_position_wrap() -> None:
    seeds = [[0, 9], [5, 29], [8, 18]]
    mapping = {(1990 + i) % 2000: i for i in range(40)}
    prompts = [render_seeds(seeds, 1990, r) for r in range(10)]
    assert len(set(prompts)) > 1
    assert all(parse_pairs(prompt, mapping) == {tuple(pair) for pair in seeds} for prompt in prompts)
    assert render_seeds([], 1990, 0) == ""


def test_completion_excludes_seed_duplicates_and_invalid_residue_pairs() -> None:
    mapping = {0: 0, 6: 5, 10: 9, 15: 14, 20: 19}
    # Seed repeats, unresolved residues, and separation shortened by missing
    # residues all have higher votes than the valid additions and must lose.
    votes = [[0, 10, 100], [0, 8, 99], [0, 6, 98], [0, 20, 7], [0, 15, 7], [10, 20, 0]]
    assert complete_pairs([[0, 10]], votes, mapping, 3) == [(0, 15, 7), (0, 20, 7)]
    with pytest.raises(ValueError, match="Only 2 voted pairs"):
        complete_pairs([[0, 10]], votes, mapping, 4)


def test_parser_deduplicates_orientation_and_rejects_invalid_tokens() -> None:
    text = "<contact> <p2> <p20> <contact> <p20> <p2> <contact> <p2> <p3> <contact> <p2> <p99>"
    assert parse_pairs(text, {2: 0, 3: 1, 20: 18}) == {(0, 18)}

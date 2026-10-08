"""Guard distinctions that would otherwise mislabel contact failure modes."""

import numpy as np
import pytest

from build_contact_atlas import summarize_target, valid_mask


def test_empty_unresolved_wrong_correct_and_unfinished_are_distinct() -> None:
    target = {'stem': 'synthetic', 'group_id': 'one', 'complex_type': 'heterodimer',
              'chain_lengths': [3, 3], 'resolved_positions_by_chain': [[0, 1], [3, 4]],
              'gt_contacts': [[0, 3]], 'n_resolved_pairs': 4,
              'target_id': 'synthetic', 'dataset': 'test', 'split': 'test', 'L': 6}
    cases = [('stop', [[0, 1]]), ('stop', [[2, 5]]), ('stop', [[1, 4]]),
             ('stop', [[0, 3], [1, 4]]), ('length', [[0, 3]])]
    samples = [{'rollout': i, 'finish_reason': cases[i % 5][0], 'contacts': cases[i % 5][1]}
               for i in range(1000)]
    summary, votes, rows, examples = summarize_target(target, samples)
    for state in ('none', 'unresolved_only', 'false_only', 'some_true', 'unfinished'):
        assert summary[state] == 200
    assert votes[(0, 3)] == 200  # unfinished partial output contributes no vote
    assert summary['top_correct'] == 0  # repeated wrong pair dominates consensus
    assert summary['typical_rollout'] == 0
    assert summary['oracle_rollout'] == 3
    assert summary['oracle_f1'] == pytest.approx(2 / 3)
    assert rows[1]['n_inter'] == 1 and rows[1]['n_scored'] == 0
    assert len(examples) == 2


def test_resolved_mask_keeps_full_chain_coordinates_and_gaps() -> None:
    mask = valid_mask({'chain_lengths': [4, 3], 'resolved_positions_by_chain': [[0, 2], [4, 6]]})
    np.testing.assert_array_equal(mask, [[1, 0, 1], [0, 0, 0], [1, 0, 1], [0, 0, 0]])

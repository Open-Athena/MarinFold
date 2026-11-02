# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A2 driver tests: substitutions, locality, mutant prefixes, and one end-to-end protein."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import a2_contact_jacobian as a2  # noqa: E402


def test_substitutions_skip_the_wild_type_and_flip_charge():
    assert a2.substitutes("A") == [None, "G", "P", "D"]
    assert a2.substitutes("P") == ["A", "G", None, "D"]
    assert a2.substitutes("K") == ["A", "G", "P", "E"]
    assert a2.substitutes("E") == ["A", "G", "P", "K"]
    assert a2.substitutes("D") == ["A", "G", "P", "K"]


def test_locality_is_one_for_effects_at_the_pair_and_null_for_uniform_effects():
    length = 40
    top_i, top_j = np.array([5]), np.array([30])
    dp = np.zeros((length, 4, 1))
    dp[7, 0, 0] = 0.2  # within 3 of i
    dp[28, 3, 0] = -0.1  # within 3 of j
    dp[15, 1, 0] = np.nan  # a skipped substitution contributes nothing
    locality, null = a2.locality_scores(dp, top_i, top_j)
    assert locality[0] == pytest.approx(1.0)
    assert null[0] == pytest.approx(14 / length)  # 7 residues around each of i and j

    uniform = np.full((length, 4, 1), 0.01)
    locality, null = a2.locality_scores(uniform, top_i, top_j)
    assert locality[0] == pytest.approx(null[0])

    locality, _ = a2.locality_scores(np.zeros((length, 4, 1)), top_i, top_j)
    assert np.isnan(locality[0])


def test_mutant_prefix_differs_in_one_token_and_a_double_mutant_is_caught(backend, protein):
    single = a2.mutant(protein, 3, "P")
    a2.check_one_token_change(backend, protein, single, seeds=2)
    double = a2.mutant(single, 8, "G")
    with pytest.raises(AssertionError, match="differs in 2 tokens"):
        a2.check_one_token_change(backend, protein, double, seeds=1)


def test_run_protein_records_every_substitution_and_skips_the_wild_type(backend, protein):
    args = argparse.Namespace(seeds=1)
    arrays, contacts, extra = a2.run_protein(protein, lambda p: backend, args)
    length = protein.length
    skipped = sum(aa is None for res in protein.structure.residues
                  for aa in a2.substitutes(a2.THREE_TO_ONE[res.resname]))
    assert extra["n_mutants"] == 4 * length - skipped
    assert arrays["dp_top"].shape == (length, 4, length)
    assert np.isnan(arrays["dp_top"][:, 0, :]).any()  # the sequence has alanines
    assert (contacts["separation"] >= 6).all()
    assert contacts["locality"].between(0, 1).all()
    # Ranked by clean P(contact), strongest first.
    assert (np.diff(contacts["p_clean"]) <= 0).all()


def test_shards_partition_the_proteins_and_balance_cost():
    table = pd.DataFrame({"key": [f"p{k}" for k in range(10)], "est_l40s_seconds": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]})
    shards = a2.assign_shards(table, 3)
    assert set(shards) == {0, 1, 2}
    loads = table.groupby(shards)["est_l40s_seconds"].sum()
    assert loads.max() - loads.min() <= 20

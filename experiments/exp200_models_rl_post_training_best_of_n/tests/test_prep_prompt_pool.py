# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks the training prompt pool matches what ``contacts_env`` reads.

The pool builder and the environment are written months apart and connected only
by an undocumented parquet schema, so this asserts the contract directly rather
than discovering a mismatch on a TPU. Needs ``marinfold`` importable::

    PYTHONPATH=../../marinfold uv run pytest tests/test_prep_prompt_pool.py
"""

import pyarrow.parquet as pq
import pytest

import contact_rewards as cr

prep = pytest.importorskip(
    "prep_prompt_pool", reason="needs marinfold on PYTHONPATH (see module docstring)"
)

# A real sequence; build_document rejects anything it cannot map to residues.
SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"


@pytest.fixture
def prompt_rows(tmp_path):
    target = {
        "entry_id": "AF-TEST-F1", "L": len(SEQUENCE), "sequence": SEQUENCE,
        "n_gt": 10, "gt_contacts": [[0, 10]], "global_plddt": 95.0, "struct_cluster_id": "c1",
    }
    written = prep.write_prompts([target], str(tmp_path), k=3, workers=2)
    assert written == 1
    return pq.read_table(tmp_path / "AF-TEST-F1.parquet").to_pylist()


def test_schema_is_exactly_what_the_environment_reads(prompt_rows):
    # contacts_env._prompts_for selects these three columns by name.
    assert set(prompt_rows[0]) >= {"r", "prefix", "seq_positions"}
    assert [row["r"] for row in prompt_rows] == [0, 1, 2]


def test_prefix_is_a_contacts_v1_document_ending_on_begin_statements(prompt_rows):
    prefix = prompt_rows[0]["prefix"]
    assert prefix.startswith("<contacts-v1>")
    # The model generates the first section's contents, so the prompt must stop
    # exactly here — this is what makes response index 0 section 0.
    assert prefix.rstrip().endswith("<begin_statements>")


def test_realizations_are_actually_resampled(prompt_rows):
    """A fresh N-terminus and statement order per rollout is exp82's recipe, and
    #163's candidate spread depends on it — identical prefixes would silently
    collapse the diversity the whole experiment is measuring."""
    assert len({row["prefix"] for row in prompt_rows}) == 3


def test_seq_positions_is_a_bijection_onto_the_sequence(prompt_rows):
    for row in prompt_rows:
        positions = row["seq_positions"]
        assert len(positions) == len(SEQUENCE)
        assert len(set(positions)) == len(SEQUENCE)
        pos_to_seq = {int(p): i for i, p in enumerate(positions)}
        assert sorted(pos_to_seq.values()) == list(range(len(SEQUENCE)))


def test_prompt_splices_the_multi_sentinel_over_a_real_prefix(prompt_rows):
    from contacts_env import ContactsV1RLEnv

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return [cr.PLAIN_DOC_ID, 8, 500, 501, cr.BEGIN_STATEMENTS_ID]

    env = ContactsV1RLEnv.__new__(ContactsV1RLEnv)
    env.doc_token_id = cr.MULTI_DOC_ID
    ids = env._build_prompt_ids(FakeTokenizer(), prompt_rows[0]["prefix"])
    assert ids[0] == cr.MULTI_DOC_ID

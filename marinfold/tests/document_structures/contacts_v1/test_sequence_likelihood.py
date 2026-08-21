# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the any-order amino-acid conditional readout — no model download.

The forward pass is replaced by ``_OracleBackend``, a stub that peeks one
token ahead: at slot ``t`` it returns near-all mass on the token that actually
occupies slot ``t + 1``. That turns the readout into an exact identity check —
if :func:`amino_acid_conditionals` maps slots to residues correctly, the
recovered argmax sequence *is* the input sequence, and any off-by-one in the
statement walk, the wrap-around position arithmetic, or the gather shows up as
a scrambled sequence rather than a plausible-looking number.

Run from the marinfold/ dir::

    uv run pytest tests/document_structures/contacts_v1/test_sequence_likelihood.py -v
"""

import numpy as np
import pytest

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1 import sequence_likelihood as sl
from marinfold.document_structures.contacts_v1.parse import residues_from_sequence
from marinfold.document_structures.contacts_v1.vocab import all_domain_tokens


# 40 residues, every canonical amino acid represented at least once.
_SEQ = "MGDIQVQVNIDDNGKAAAAQWCFHRSTYLPEKMNVGWCFH"


def _tokenizer():
    return build_tokenizer(all_domain_tokens())


class _OracleBackend:
    """Stub whose slot-``t`` distribution is one-hot on the token at ``t + 1``.

    Not a model — a probe. It makes the *correct* answer at every amino-acid
    slot exactly knowable, so the test asserts on identity rather than on a
    restatement of the implementation.
    """

    def __init__(self, tokenizer, hit: float = 0.9):
        self._tok = tokenizer
        self._hit = hit

    @property
    def tokenizer(self):
        return self._tok

    def teacher_forced_target_probs(
        self, token_ids_batch, target_token_ids, *, batch_size=None
    ):
        lengths = {len(row) for row in token_ids_batch}
        assert len(lengths) == 1, f"rows must share a length; got {lengths}"
        seq_len = lengths.pop()
        column = {tid: i for i, tid in enumerate(target_token_ids)}
        # Off-target mass is deliberately non-zero: real softmax leaks onto
        # <UNK> and off-grammar tokens, and target_mass must reflect that.
        miss = (1.0 - self._hit) / len(target_token_ids)
        out = np.full(
            (len(token_ids_batch), seq_len, len(target_token_ids)), miss / 2
        )
        for row_index, row in enumerate(token_ids_batch):
            for slot in range(seq_len - 1):
                col = column.get(row[slot + 1])
                if col is not None:
                    out[row_index, slot, col] = self._hit
        return out


def _conditionals(sequence=_SEQ, *, num_orderings=1, entry_id="probe", hit=0.9):
    return sl.amino_acid_conditionals(
        _OracleBackend(_tokenizer(), hit=hit),
        residues_from_sequence(sequence),
        entry_id=entry_id,
        num_orderings=num_orderings,
    )


# ---------------------------------------------------------------------------
# Slot mapping: the load-bearing correctness property
# ---------------------------------------------------------------------------


def test_recovers_the_sequence_from_the_oracle():
    """Every residue's argmax is that residue's own amino acid."""
    cond = _conditionals(num_orderings=4)
    recovered = [
        "".join(sl.AA_ALPHABET[c] for c in cond.logprobs[k].argmax(axis=1))
        for k in range(cond.num_orderings)
    ]
    assert recovered == [_SEQ] * 4


def test_slot_mapping_holds_under_a_different_ordering_seed():
    """A different seed reshuffles statements and start index — mapping still holds."""
    other = _conditionals(entry_id="a-completely-different-stem")
    recovered = "".join(
        sl.AA_ALPHABET[c] for c in other.logprobs[0].argmax(axis=1)
    )
    assert recovered == _SEQ


def test_shapes_and_alphabet():
    cond = _conditionals(num_orderings=3)
    assert cond.seq_len == len(_SEQ)
    assert cond.num_orderings == 3
    assert cond.logprobs.shape == (3, len(_SEQ), 20)
    assert cond.context_sizes.shape == (3, len(_SEQ))
    assert cond.target_mass.shape == (3, len(_SEQ))
    assert sl.AA_TOKENS[0] == "<ALA>" and sl.AA_TOKENS[-1] == "<TYR>"
    assert len(sl.AA_ALPHABET) == 20


# ---------------------------------------------------------------------------
# Context sizes
# ---------------------------------------------------------------------------


def test_context_sizes_are_a_permutation_of_zero_to_l_minus_one():
    """Each residue statement sees a distinct number of earlier residues."""
    cond = _conditionals(num_orderings=3)
    for k in range(cond.num_orderings):
        assert sorted(cond.context_sizes[k].tolist()) == list(range(len(_SEQ)))


def test_context_fraction_spans_zero_to_one():
    cond = _conditionals()
    fractions = cond.context_fractions()
    assert fractions.min() == pytest.approx(0.0)
    assert fractions.max() == pytest.approx(1.0)


def test_orderings_differ_from_each_other():
    """K orderings must actually be K different shuffles, not one repeated."""
    cond = _conditionals(num_orderings=5)
    assert not np.array_equal(cond.context_sizes[0], cond.context_sizes[1])
    distinct = {tuple(row) for row in cond.context_sizes.tolist()}
    assert len(distinct) == 5


def test_same_entry_id_reproduces_exactly():
    a = _conditionals(num_orderings=2, entry_id="stable")
    b = _conditionals(num_orderings=2, entry_id="stable")
    assert np.array_equal(a.context_sizes, b.context_sizes)
    assert np.array_equal(a.logprobs, b.logprobs)


# ---------------------------------------------------------------------------
# Renormalization and mass accounting
# ---------------------------------------------------------------------------


def test_logprobs_are_renormalized_over_the_twenty():
    cond = _conditionals()
    total = np.exp(cond.logprobs).sum(axis=-1)
    assert np.allclose(total, 1.0, atol=1e-5)


def test_target_mass_reports_pre_renormalization_leakage():
    """Mass on the 20 is below 1 exactly as much as the stub leaks."""
    hit = 0.8
    cond = _conditionals(hit=hit)
    expected = hit + 19 * ((1.0 - hit) / 20) / 2
    assert np.allclose(cond.target_mass, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def test_mean_logprobs_threshold_drops_low_context_slots():
    cond = _conditionals(num_orderings=8)
    counts = cond.sample_counts(min_context_fraction=0.5)
    assert (counts <= 8).all()
    # At threshold 1.0 only the statement that landed last in each ordering
    # qualifies, so the total across residues is exactly one per ordering.
    assert cond.sample_counts(min_context_fraction=1.0).sum() == 8


def test_mean_logprobs_is_nan_where_nothing_qualifies():
    cond = _conditionals(num_orderings=1)
    mean = cond.mean_logprobs(min_context_fraction=1.0)
    qualifying = cond.sample_counts(min_context_fraction=1.0) > 0
    assert qualifying.sum() == 1
    assert not np.isnan(mean[qualifying]).any()
    assert np.isnan(mean[~qualifying]).all()


def test_substitution_log_ratios_are_zero_at_wild_type():
    cond = _conditionals(num_orderings=4)
    ratios = sl.substitution_log_ratios(cond, _SEQ)
    assert ratios.shape == (len(_SEQ), 20)
    wt_columns = [sl.AA_ALPHABET.index(c) for c in _SEQ]
    assert np.allclose(ratios[np.arange(len(_SEQ)), wt_columns], 0.0, atol=1e-6)


def test_substitution_log_ratios_favour_the_wild_type_under_the_oracle():
    """The oracle puts its mass on the true residue, so every swap scores < 0."""
    ratios = sl.substitution_log_ratios(_conditionals(num_orderings=2), _SEQ)
    off_wt = ratios[ratios != 0.0]
    assert (off_wt < 0).all()


# ---------------------------------------------------------------------------
# Failure modes — these must be loud
# ---------------------------------------------------------------------------


def test_rejects_zero_orderings():
    with pytest.raises(ValueError, match="num_orderings"):
        _conditionals(num_orderings=0)


def test_rejects_unserializable_chain():
    with pytest.raises(ValueError, match="cannot serialize"):
        _conditionals(sequence="M")


def test_substitution_rejects_length_mismatch():
    cond = _conditionals()
    with pytest.raises(ValueError, match="residues"):
        sl.substitution_log_ratios(cond, _SEQ[:-1])


def test_substitution_is_nan_at_a_non_canonical_wild_type_site():
    """No reference column there, so the row must be nan — not a wrong number."""
    cond = _conditionals()
    ratios = sl.substitution_log_ratios(cond, "X" + _SEQ[1:])
    assert np.isnan(ratios[0]).all()
    assert not np.isnan(ratios[1:]).any()


def test_rejects_a_tokenizer_without_the_contacts_v1_vocab():
    """A wrong tokenizer must fail, not silently return garbage conditionals."""
    plain = build_tokenizer(["<a>", "<b>", "<c>"])
    with pytest.raises(ValueError, match="contacts-v1"):
        sl.amino_acid_conditionals(
            _OracleBackend(plain), residues_from_sequence(_SEQ), entry_id="x"
        )

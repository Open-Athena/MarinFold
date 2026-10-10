# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Harness tests on a tiny random Qwen3 with the real contacts-v1 vocabulary."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1.inference import ContactStructure
from marinfold.document_structures.contacts_v1.parse import RawContact, residues_from_sequence
from marinfold.document_structures.contacts_v1.vocab import all_domain_tokens
from marinfold.inference._transformers import TransformersBackend
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import harness  # noqa: E402

SEQUENCE = "MGDIQVQVNIDDNGKAAAAQ"


@pytest.fixture(scope="module")
def backend():
    tokenizer = build_tokenizer(all_domain_tokens())
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=512,
    )
    return TransformersBackend.from_model(Qwen3ForCausalLM(config).float(), tokenizer)


@pytest.fixture(scope="module")
def protein():
    contacts = (RawContact(0, 10, 0.5), RawContact(2, 15, 0.2), RawContact(4, 19, 0.0005))
    structure = ContactStructure(
        entry_id="toy", residues=residues_from_sequence(SEQUENCE), gt_contacts=contacts
    )
    return harness.EvalProtein(eval_set="eval-val", structure=structure, resolved=np.arange(20))


def test_capture_records_every_layer_boundary_without_changing_the_readout(backend, protein):
    clean = harness.pcontact(backend, protein, seeds=1)
    with harness.capture_residuals(backend.model) as captured:
        hooked = harness.pcontact(backend, protein, seeds=1)
    np.testing.assert_array_equal(hooked, clean)
    assert len(captured) == 3  # embedding + 2 layers
    # fwd_matrix makes two next_token_probs calls (lp1, lp2), each a prefix
    # forward plus one tail chunk (the 20 lp2 tails fit one chunk of 64).
    assert [len(calls) for calls in captured] == [4, 4, 4]
    assert captured[0][0].shape[-1] == 32


def test_empty_ablation_is_the_clean_readout_and_real_ablation_moves_it(backend, protein):
    clean = harness.pcontact(backend, protein, seeds=1)
    with harness.mean_ablation(backend.model, {}):
        np.testing.assert_array_equal(harness.pcontact(backend, protein, seeds=1), clean)
    with harness.mean_ablation(backend.model, {(1, "mlp"): torch.zeros(32), (0, "attn"): torch.zeros(32)}):
        ablated = harness.pcontact(backend, protein, seeds=1)
    assert not np.allclose(ablated, clean)
    # Hooks are removed on exit.
    np.testing.assert_array_equal(harness.pcontact(backend, protein, seeds=1), clean)


def test_r_precision_counts_only_resolved_pairs_above_the_degree_floor(protein):
    score = np.zeros((20, 20))
    score[0, 10] = score[10, 0] = 1.0
    # Two true contacts (the 0.0005 one is below the floor); one ranked first.
    assert harness.r_precision(score, protein) == 0.5
    score[2, 15] = score[15, 2] = 0.9
    assert harness.r_precision(score, protein) == 1.0
    # Dropping residue 15 from the resolved set removes (2, 15) from the universe.
    partial = harness.EvalProtein("eval-val", protein.structure, np.delete(np.arange(20), 15))
    score[2, 15] = score[15, 2] = 0.0
    assert harness.r_precision(score, partial) == 1.0
    assert np.isnan(harness.r_precision(score, protein, min_separation=24))


def test_eval_test_is_refused():
    with pytest.raises(ValueError, match="read budget"):
        harness.load_eval_proteins(["eval-val", "eval-test"])


def test_eval_proteins_load_with_in_range_ground_truth():
    proteins = harness.load_eval_proteins()
    counts = {name: sum(p.eval_set == name for p in proteins) for name in harness.ALLOWED_EVAL_SETS}
    assert counts == {"eval-val": 97, "eval-denovo": 19}
    for p in proteins:
        assert p.resolved.min() >= 0 and p.resolved.max() < p.length
        assert all(0 <= c.seq_i < c.seq_j < p.length for c in p.structure.gt_contacts)

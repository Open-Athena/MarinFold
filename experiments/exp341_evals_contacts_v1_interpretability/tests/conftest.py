# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: a tiny random Qwen3 with the real contacts-v1 vocabulary and a toy protein."""

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

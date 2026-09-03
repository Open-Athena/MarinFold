# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The levanter smear and the PyTorch pilot smear must be the same function.

Two implementations of one idea is how a pilot and the production run it is
supposed to inform quietly stop testing the same thing. This loads both, copies
one set of weights across, and demands they agree.
"""

import sys
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import numpy as np
import pytest
import torch
from haliax import Axis

# ``gpu/`` and ``pilot/`` are siblings inside the experiment directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot"))

from smear import SmearEmbedding  # noqa: E402

from architecture import SmearQwen3Config  # noqa: E402

VOCAB = 37
POSITIONS = 12
HIDDEN = 32


@pytest.mark.parametrize("width", [1, 2, 3])
def test_torch_and_levanter_smears_agree(width: int):
    generator = torch.Generator().manual_seed(5)
    table = torch.nn.Embedding(VOCAB, HIDDEN)
    torch_smear = SmearEmbedding(table, width)
    with torch.no_grad():
        table.weight.normal_(std=0.7, generator=generator)
        torch_smear.weights.normal_(std=0.4, generator=generator)
        torch_smear.gate.weight.normal_(std=0.3, generator=generator)
        torch_smear.gate.bias.normal_(std=0.1, generator=generator)
    ids = torch.randint(0, VOCAB, (1, POSITIONS), generator=generator)
    with torch.no_grad():
        expected = torch_smear(ids)[0].numpy()

    config = SmearQwen3Config(
        max_seq_len=POSITIONS, hidden_dim=HIDDEN, intermediate_dim=64,
        num_heads=4, num_kv_heads=2, num_layers=1, smear_width=width,
    )
    model = config.model_type.init(Axis("vocab", VOCAB), config, key=jax.random.PRNGKey(0))

    def named(array, axes):
        return hax.named(jax.numpy.asarray(np.asarray(array)), axes)

    embeddings = eqx.tree_at(
        lambda e: (e.token_embeddings.weight, e.weights, e.gate.weight, e.gate.bias),
        model.embeddings,
        (
            named(table.weight.detach().numpy(), ("vocab", "embed")),
            named(torch_smear.weights.detach().numpy(), ("smear_offset", "embed")),
            named(torch_smear.gate.weight.detach().numpy(), ("smear_offset", "smear_gate_in")),
            named(torch_smear.gate.bias.detach().numpy(), ("smear_offset",)),
        ),
    )
    actual = np.asarray(
        embeddings.embed(named(ids[0].numpy(), ("position",))).array
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

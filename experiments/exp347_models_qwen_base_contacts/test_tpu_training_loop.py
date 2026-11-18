"""Compare the production logical-rank loop with one complete 32-document update."""

from unittest import skipUnless

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tpu_checkpoint import replicated_adam_step
from tpu_model import abstract_model, configuration, convert_parameters, initial_rest
from tpu_optimizer import learning_rate
from tpu_training import document_batch, training_functions
from train_tpu import update_once
from validate_tpu_math import probe_config


class FixedStream:
    """Provide distinguishable deterministic documents for one logical rank."""

    def __init__(self, rank: int) -> None:
        self.state = {"rank": rank, "world": 8, "row": 0}
        rng = np.random.default_rng(347 + rank)
        self.rows = [
            {"contacts_v1": rng.integers(0, 256, 55 + rank + i).tolist()}
            for i in range(4)
        ]

    def next(self) -> dict:
        row = self.rows[self.state["row"]]
        self.state["row"] += 1
        return row


@skipUnless(
    jax.default_backend() == "cpu" and jax.device_count() == 4,
    "Requires four emulated CPU devices",
)
def test_production_update_preserves_logical_batch() -> None:
    """Check tokens, all cursors, schedule, loss, and nonzero Adam state together."""
    torch.set_num_threads(4)
    torch.manual_seed(347)
    source = probe_config()
    hf = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**source)).eval()
    config = configuration(source, length=128, batch=4, dtype="float32")
    mesh, model = abstract_model(config)
    graph, template, rest = nnx.split(model, nnx.Param, ...)
    rest = initial_rest(rest)
    params = convert_parameters(
        {n: p.detach().numpy() for n, p in hf.named_parameters()},
        source,
        config,
        template,
    ).to_pure_dict()
    state = {
        "params": params,
        "mu": jax.tree.map(lambda p: jnp.full_like(p, 0.001), params),
        "nu": jax.tree.map(lambda p: jnp.full_like(p, 0.002), params),
        "adam_step": replicated_adam_step(59319, mesh),
    }
    metadata = {
        "step": 59319,
        "tokens": 9122000000,
        "format": "contacts_v1",
        "target_tokens": 100000000000,
    }
    streams = [FixedStream(rank) for rank in range(8)]
    rows = [stream.rows[i] for i in range(4) for stream in streams]
    count = sum(len(row["contacts_v1"]) - 1 for row in rows)
    functions = training_functions(graph, template, rest, config)
    sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))
    batch = jax.tree.map(
        lambda a: jax.device_put(a, sharding), document_batch(rows, "contacts_v1", 128)
    )
    loss, grads = functions[0](state["params"], rest, batch, count)
    expected, _ = functions[2](
        jax.tree.map(jnp.copy, state), grads, learning_rate(metadata["tokens"])
    )
    result, next_metadata, metrics = update_once(
        state, metadata, streams, rest, functions, sharding, length=128
    )
    for actual, reference in zip(
        jax.tree.leaves(result), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(metrics["train/loss"], loss, atol=2e-6, rtol=2e-6)
    assert next_metadata["step"] == 59320
    assert next_metadata["adam_step"] == 59320
    assert next_metadata["tokens"] == metadata["tokens"] + count
    assert next_metadata["cursors"] == [
        {"rank": rank, "world": 8, "row": 4} for rank in range(8)
    ]
    assert metrics["train/lr"] == learning_rate(metadata["tokens"])

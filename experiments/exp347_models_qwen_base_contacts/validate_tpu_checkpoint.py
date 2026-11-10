"""Exercise accumulation and exact resumed training through the Orbax adapter."""

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tpu_checkpoint import restore_arrays, save_arrays
from tpu_model import abstract_model, configuration, convert_parameters, initial_rest
from tpu_training import document_batch, training_functions
from validate_tpu_math import probe_config
from validate_tpu_runtime import verify_roundtrip


def validate() -> dict:
    """Require padded accumulation and save/restore to preserve the next update."""
    torch.set_num_threads(4)
    torch.manual_seed(347)
    source = probe_config()
    hf = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**source)).eval()
    config = configuration(source, length=128, batch=2, dtype="float32")
    _, model = abstract_model(config)
    graph, template, rest = nnx.split(model, nnx.Param, ...)
    rest = initial_rest(rest)
    params = convert_parameters(
        {name: value.detach().numpy() for name, value in hf.named_parameters()},
        source,
        config,
        template,
    ).to_pure_dict()
    state = {
        "params": params,
        "mu": jax.tree.map(lambda p: jnp.full_like(p, 0.001), params),
        "nu": jax.tree.map(lambda p: jnp.full_like(p, 0.002), params),
        "adam_step": jnp.asarray(59319),
    }
    rng = np.random.default_rng(914)
    rows = [
        {"contacts_v1": rng.integers(0, 256, length).tolist()}
        for length in (129, 81, 99, 66)
    ]
    count = sum(len(row["contacts_v1"]) - 1 for row in rows)
    gradient_fn, accumulate_fn, update_fn = training_functions(
        graph, template, rest, config
    )

    def step(current: dict, batch_size: int = 2) -> tuple[dict, float]:
        total_loss = 0.0
        accumulated = None
        for i in range(0, len(rows), batch_size):
            batch = document_batch(rows[i : i + batch_size], "contacts_v1", 128)
            loss, gradient = gradient_fn(current["params"], rest, batch, count)
            total_loss += float(loss)
            accumulated = (
                gradient
                if accumulated is None
                else accumulate_fn(accumulated, gradient)
            )
        updated, _ = update_fn(current, accumulated, 1.9e-5)
        return updated, total_loss

    full = document_batch(rows, "contacts_v1", 128)
    targets = (
        torch.tensor(full["targets"])
        .long()
        .masked_fill(torch.tensor(full["mask"]) == 0, -100)
    )
    with torch.no_grad():
        logits = hf(
            torch.tensor(full["tokens"]).long(),
            attention_mask=torch.tensor(full["mask"]),
            use_cache=False,
        ).logits
        expected = float(
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, 256), targets.reshape(-1), reduction="sum"
            )
            / count
        )
    with tempfile.TemporaryDirectory(prefix="exp347-orbax-test-") as folder:
        uri = folder + "/state"
        save_arrays(uri, state)
        restored = restore_arrays(uri, state)
        smaller_batch_state = restore_arrays(uri, state)
        verify_roundtrip(state, restored)
        original_next, loss = step(state)
        restored_next, restored_loss = step(restored)
        verify_roundtrip(original_next, restored_next)
        np.testing.assert_allclose(loss, expected, atol=2e-6, rtol=2e-6)
        if restored_loss != loss or int(restored_next["adam_step"]) != 59320:
            raise ValueError("Restored training did not preserve the next update")
        smaller_next, smaller_loss = step(smaller_batch_state, batch_size=1)
        np.testing.assert_allclose(smaller_loss, loss, atol=2e-6, rtol=2e-6)
        # Regrouping the same variable-length documents must preserve the
        # token-weighted objective and Adam state, up to FP32 reduction order.
        for original, smaller in zip(
            jax.tree.leaves(original_next), jax.tree.leaves(smaller_next), strict=True
        ):
            np.testing.assert_allclose(original, smaller, atol=2e-6, rtol=2e-6)
    return {
        "status": "passed",
        "completed_utc": datetime.now(UTC).isoformat(),
        "backend": jax.default_backend(),
        "devices": jax.device_count(),
        "loss": loss,
        "reference_loss": expected,
        "valid_tokens": count,
        "state_roundtrip": "exact",
        "resumed_update": "exact",
        "microbatch_sizes": [2, 1],
        "microbatch_update_parity": "passed (atol=2e-6, rtol=2e-6)",
        "smaller_microbatch_loss": smaller_loss,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = validate()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)

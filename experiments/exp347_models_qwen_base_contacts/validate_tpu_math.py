"""Compare hybrid Qwen logits, padded token loss, gradients, and resumed AdamW.

Run with the isolated TPU runtime on CPU before allocating a validation slice.
The small model retains the dense 3:1 hybrid architecture, tied embeddings,
partial RoPE, grouped heads, convolution, and multi-chunk DeltaNet recurrence.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tpu_model import (
    abstract_model,
    apply_model,
    configuration,
    convert_parameters,
    initial_rest,
    summed_loss,
)
from tpu_optimizer import adamw_update, learning_rate


def probe_config() -> dict:
    """Return a small model with all production architectural operations present."""
    source = json.loads(Path("data/tpu_forks/source_model_config.json").read_text())
    source.update(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        vocab_size=256,
        eos_token_id=255,
        layer_types=source["layer_types"][:4],
    )
    source["rope_parameters"]["mrope_section"] = [1, 1, 2]
    return source


def compare_trees(
    actual: nnx.State, expected: nnx.State, *, atol: float, rtol: float
) -> float:
    """Require every mapped parameter to agree, including both tied-head gradients."""
    errors = []
    for (pa, a), (pe, e) in zip(
        nnx.to_flat_state(actual), nnx.to_flat_state(expected), strict=True
    ):
        if pa != pe:
            raise ValueError("Parameter paths differ")
        av, ev = np.asarray(a.get_value()), np.asarray(e.get_value())
        np.testing.assert_allclose(av, ev, atol=atol, rtol=rtol, err_msg=str(pa))
        errors.append(float(np.max(np.abs(av - ev))))
    return max(errors)


def validate() -> dict:
    """Check FP32 equations and a nonzero-state optimizer continuation."""
    torch.set_num_threads(4)
    torch.manual_seed(347)
    source = probe_config()
    hf = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**source)).eval()
    config = configuration(source, length=128, batch=2, dtype="float32")
    _, model = abstract_model(config)
    graph, template, rest = nnx.split(model, nnx.Param, ...)
    rest = initial_rest(rest)

    def convert(values: dict) -> nnx.State:
        return convert_parameters(values, source, config, template)

    params = convert({n: p.detach().numpy() for n, p in hf.named_parameters()})
    ids = np.random.default_rng(347).integers(0, 256, (2, 129), dtype=np.int32)
    # The second document ends early, exercising causal padding after a complete
    # document while the first crosses two DeltaNet chunks.
    mask = np.ones((2, 128), np.int32)
    mask[1, 79:] = 0
    x, targets = ids[:, :-1], ids[:, 1:]
    tx = torch.from_numpy(x.astype(np.int64))
    ty = torch.from_numpy(targets.astype(np.int64)).masked_fill(
        torch.from_numpy(mask == 0), -100
    )
    ref = hf(tx, attention_mask=torch.from_numpy(mask), use_cache=False).logits
    logits = jax.jit(lambda p, r, t, m: apply_model(graph, p, r, config, t, m))(
        params, rest, x, mask
    )
    valid = mask.astype(bool)
    np.testing.assert_allclose(
        np.asarray(logits)[valid], ref.detach().numpy()[valid], atol=2e-5, rtol=2e-4
    )
    token_count = int(mask.sum())
    reference_loss = (
        torch.nn.functional.cross_entropy(
            ref.reshape(-1, 256), ty.reshape(-1), reduction="sum"
        )
        / token_count
    )
    reference_loss.backward()

    def loss_fn(p, r, x, y, m):
        return summed_loss(graph, p, r, config, x, y, m) / token_count

    loss, grad = jax.jit(jax.value_and_grad(loss_fn))(params, rest, x, targets, mask)
    np.testing.assert_allclose(
        float(loss), float(reference_loss.detach()), atol=2e-6, rtol=2e-6
    )
    expected_grad = convert({n: p.grad.numpy() for n, p in hf.named_parameters()})
    gradient_error = compare_trees(grad, expected_grad, atol=2e-6, rtol=2e-4)

    optimizer = torch.optim.AdamW(
        hf.parameters(),
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=False,
    )
    # Nonzero saved moments and a large saved count ensure this checks continuation,
    # not a fresh optimizer that can conceal a missing or reset state shard.
    rng = np.random.default_rng(839)
    for param in hf.parameters():
        optimizer.state[param] = {
            "step": torch.tensor(59319.0),
            "exp_avg": torch.from_numpy(
                rng.normal(0, 0.001, tuple(param.shape)).astype(np.float32)
            ),
            "exp_avg_sq": torch.from_numpy(
                rng.uniform(0.001, 0.003, tuple(param.shape)).astype(np.float32)
            ),
        }
    mu = convert(
        {n: optimizer.state[p]["exp_avg"].numpy() for n, p in hf.named_parameters()}
    )
    nu = convert(
        {n: optimizer.state[p]["exp_avg_sq"].numpy() for n, p in hf.named_parameters()}
    )
    lr = float(learning_rate(9_121_552_680))
    optimizer.param_groups[0]["lr"] = lr
    # Exercise the clipping branch independently of the random model's gradient scale.
    for param in hf.parameters():
        param.grad *= 100
    grad = jax.tree.map(lambda g: g * 100, grad)
    norm = torch.nn.utils.clip_grad_norm_(hf.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    updated, mu, nu, count, actual_norm = jax.jit(adamw_update)(
        params, mu, nu, grad, jnp.asarray(59319), lr
    )
    np.testing.assert_allclose(float(actual_norm), float(norm), atol=2e-5, rtol=2e-5)
    parameter_error = compare_trees(
        updated,
        convert({n: p.detach().numpy() for n, p in hf.named_parameters()}),
        atol=2e-7,
        rtol=2e-6,
    )
    for actual, key in ((mu, "exp_avg"), (nu, "exp_avg_sq")):
        compare_trees(
            actual,
            convert(
                {n: optimizer.state[p][key].numpy() for n, p in hf.named_parameters()}
            ),
            atol=2e-7,
            rtol=2e-5,
        )
    if int(count) != 59320:
        raise ValueError("Adam step did not continue")
    return {
        "completed_utc": datetime.now(UTC).isoformat(),
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "torch_version": torch.__version__,
        "valid_tokens": token_count,
        "logits_max_absolute_error": float(
            np.max(np.abs(np.asarray(logits)[valid] - ref.detach().numpy()[valid]))
        ),
        "loss": float(loss),
        "reference_loss": float(reference_loss.detach()),
        "gradient_max_absolute_error": gradient_error,
        "updated_parameter_max_absolute_error": parameter_error,
        "adam_step_before": 59319,
        "adam_step_after": int(count),
        "grad_norm": float(norm),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate()
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

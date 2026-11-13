"""One full-document, token-weighted update of the frozen GPU experiment."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tpu_checkpoint import as_nnx
from tpu_model import summed_loss
from tpu_optimizer import adamw_update


def document_batch(rows: list[dict], document_format: str, length: int) -> dict:
    """Right-pad independent next-token examples without changing their objective."""
    tokens = np.zeros((len(rows), length), np.int32)
    targets = np.zeros_like(tokens)
    mask = np.zeros_like(tokens)
    for i, row in enumerate(rows):
        ids = row[document_format]
        if not 1 < len(ids) <= length + 1:
            raise ValueError("Invalid complete-document length")
        count = len(ids) - 1
        tokens[i, :count], targets[i, :count] = ids[:-1], ids[1:]
        mask[i, :count] = 1
    return {"tokens": tokens, "targets": targets, "mask": mask}


def training_functions(
    graph: Any, template: nnx.State, rest: nnx.State, config: Any
) -> tuple[Callable, Callable, Callable]:
    """Compile microbatch gradients, accumulation, and the exact resumed AdamW step."""

    def loss_fn(params: dict, rest: nnx.State, batch: dict, count: Any) -> Any:
        return (
            summed_loss(
                graph,
                as_nnx(template, params),
                rest,
                config,
                batch["tokens"],
                batch["targets"],
                batch["mask"],
            )
            / count
        )

    value_and_grad = jax.value_and_grad(loss_fn)
    gradient = jax.jit(value_and_grad)

    @jax.jit(donate_argnums=(4,))
    def accumulate(
        params: dict, rest: nnx.State, batch: dict, count: Any, accumulated: dict
    ) -> tuple[Any, dict]:
        # A separate gradient call must materialize a full second gradient tree
        # before accumulation can donate either buffer. Fuse the two operations
        # so XLA can reuse the old accumulator while producing the new one.
        loss, current = value_and_grad(params, rest, batch, count)
        return loss, jax.tree.map(jnp.add, accumulated, current)

    @jax.jit(donate_argnums=(0, 1))
    def update(state: dict, gradient: dict, lr: Any) -> tuple[dict, Any]:
        params, mu, nu, step, norm = adamw_update(
            state["params"], state["mu"], state["nu"], gradient, state["adam_step"], lr
        )
        return {"params": params, "mu": mu, "nu": nu, "adam_step": step}, norm

    return gradient, accumulate, update

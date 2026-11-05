"""The GPU experiment's FP32 AdamW update, clipping, and absolute-token schedule."""

import math
from typing import Any

import jax
import jax.numpy as jnp


def learning_rate(tokens: int, target: int = 100_000_000_000) -> float:
    """Evaluate the unchanged schedule at the number of tokens before this update."""
    ratio = min(tokens / target, 1.0)
    factor = (
        max(0.01, ratio / 0.01)
        if ratio < 0.01
        else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (ratio - 0.01) / 0.99))
    )
    return 2e-5 * factor


def adamw_update(
    params: Any, mu: Any, nu: Any, grad: Any, count: Any, lr: Any
) -> tuple[Any, Any, Any, Any, Any]:
    """Apply global norm clipping and continue all saved FP32 Adam moments."""
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(grad)))
    scale = jnp.minimum(1.0, 1.0 / (norm + 1e-6))
    grad = jax.tree.map(lambda g: g * scale, grad)
    count = count + 1
    mu = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, mu, grad)
    nu = jax.tree.map(lambda v, g: 0.95 * v + 0.05 * g * g, nu, grad)
    correction1 = 1 - jnp.power(0.9, count)
    correction2 = 1 - jnp.power(0.95, count)
    params = jax.tree.map(
        lambda p, m, v: (
            p * (1 - lr * 0.1)
            - (lr / correction1) * m / (jnp.sqrt(v / correction2) + 1e-8)
        ),
        params,
        mu,
        nu,
    )
    return params, mu, nu, count, norm

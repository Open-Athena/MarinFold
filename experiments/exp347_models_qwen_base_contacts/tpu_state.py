"""Recover the GPU run's eight ZeRO-1 optimizer partitions by parameter name.

PyTorch 2.10 stores rank-local integer parameter IDs rather than names. Recreate
its stable descending-size, least-loaded-rank partition from the original model
enumeration. Refuse missing states, unexpected shapes, or changed Adam settings.
No optimizer is reset and no weights are cast while decoding the checkpoint.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def parameter_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """Enumerate unique HF parameters in the original PyTorch module order."""
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
        model.tie_weights()
    return {name: tuple(param.shape) for name, param in model.named_parameters()}


def serialized_parameter_names(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, str]:
    """Map Qwen's saved text-weight names to its in-memory parameter names.

    Transformers 5.3 saves this text-only model under ``model.language_model``
    through its checkpoint conversion mapping, while ``named_parameters`` uses
    ``model``. Optimizer partition reconstruction must retain the latter names
    and their original order. The tied LM head has no separate saved tensor.
    """
    if any(not name.startswith("model.") for name in shapes):
        raise ValueError("Expected the tied Qwen text model's unique parameters")
    return {
        "model.language_model." + name.removeprefix("model."): name for name in shapes
    }


def zero_partitions(
    shapes: Mapping[str, tuple[int, ...]], world: int = 8
) -> list[list[str]]:
    """Reconstruct PyTorch 2.10's deterministic one-group ZeRO-1 assignment."""
    if world < 1:
        raise ValueError("World size must be positive")
    sizes = [0] * world
    partitions: list[list[str]] = [[] for _ in range(world)]
    for name in sorted(shapes, key=lambda n: int(np.prod(shapes[n])), reverse=True):
        rank = min(range(world), key=lambda r: sizes[r])
        partitions[rank].append(name)
        sizes[rank] += int(np.prod(shapes[name]))
    return partitions


def decode_rank(
    state: dict,
    names: list[str],
    shapes: Mapping[str, tuple[int, ...]],
    parent: dict,
    rank: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, dict]:
    """Decode one saved rank, checking scientific and checkpoint invariants."""
    if any(state[key] != parent[key] for key in ("step", "tokens", "world")):
        raise ValueError("Optimizer shard does not belong to the frozen parent")
    if state["stream"]["rank"] != rank or state["stream"]["world"] != 8:
        raise ValueError("Unexpected logical data rank")
    optimizer = state["optimizer"]
    groups = optimizer["param_groups"]
    if len(groups) != 1:
        raise ValueError("Expected one original AdamW parameter group")
    group = groups[0]
    expected = {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
        "amsgrad": False,
        "maximize": False,
    }
    if any(group[key] != value for key, value in expected.items()):
        raise ValueError("AdamW settings differ from the original experiment")
    if len(group["params"]) != len(names) or set(optimizer["state"]) != set(
        group["params"]
    ):
        raise ValueError("Incomplete optimizer partition")
    mu, nu, steps = {}, {}, set()
    for name, index in zip(names, group["params"], strict=True):
        item = optimizer["state"][index]
        if set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"Unexpected optimizer fields for {name}")
        value = float(item["step"])
        if not np.isfinite(value) or value < 1 or value != int(value):
            raise ValueError("Invalid Adam step")
        steps.add(int(value))
        for key, output in (("exp_avg", mu), ("exp_avg_sq", nu)):
            array = item[key].detach().numpy()
            if array.shape != shapes[name] or array.dtype != np.float32:
                raise ValueError(f"Optimizer shape/dtype mismatch for {name}")
            if not np.isfinite(array).all() or (
                key == "exp_avg_sq" and np.any(array < 0)
            ):
                raise ValueError(f"Invalid optimizer values for {name}")
            output[name] = array
    if len(steps) != 1:
        raise ValueError("Parameters disagree on Adam step")
    provenance: dict[str, Any] = {
        key: value for key, value in state.items() if key != "optimizer"
    }
    return mu, nu, steps.pop(), provenance

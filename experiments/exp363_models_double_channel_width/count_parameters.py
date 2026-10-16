"""Count actual model parameter shapes without allocating model weights."""

import csv
from pathlib import Path

import equinox as eqx
import jax
from haliax import Axis
from levanter.models.qwen import Qwen3Config

from experiments.exp232_sweep_cv1_decontam.training_contract import VOCAB_SIZE
from experiments.exp363_models_double_channel_width.config import (
    BASELINE_MODEL_CONFIG,
    EXPECTED_PARAMETERS,
    MODEL_CONFIG,
)


def count_parameters(config: Qwen3Config) -> int:
    """Sum every parameter leaf in the abstract Levanter model tree."""
    model = eqx.filter_eval_shape(
        config.build, Axis("vocab", VOCAB_SIZE), key=jax.random.PRNGKey(0)
    )
    return sum(leaf.size for leaf in jax.tree.leaves(model))


def main() -> None:
    """Write the reproducible architecture audit next to the experiment."""
    rows = []
    for name, config in (("exp277", BASELINE_MODEL_CONFIG), ("exp363", MODEL_CONFIG)):
        count = count_parameters(config)
        if name == "exp363" and count != EXPECTED_PARAMETERS:
            raise ValueError(f"Unexpected model parameter count: {count}")
        row = {
            "model": name,
            "hidden_dim": config.hidden_dim,
            "intermediate_dim": config.intermediate_dim,
            "layers": config.num_layers,
            "query_heads": config.num_heads,
            "kv_heads": config.num_kv_heads,
            "head_dim": config.actual_head_size,
            "vocab_size": VOCAB_SIZE,
            "parameters": count,
        }
        print(row)
        rows.append(row)
    path = Path(__file__).parent / "data/parameter_counts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

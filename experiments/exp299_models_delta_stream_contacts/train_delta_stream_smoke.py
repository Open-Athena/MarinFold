# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny standard next-token CE smoke on delta-stream contact documents."""

import argparse
import os
import socket
from typing import Any, NamedTuple

import fsspec
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import pyarrow.parquet as pq
import wandb

DEFAULT_DATA = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/documents/2026.09.16.1/shard-00000-of-03338.parquet"
)


class Batch(NamedTuple):
    input_ids: jax.Array
    target_ids: jax.Array
    mask: jax.Array


def read_rows(path: str, *, num_rows: int, max_tokens: int) -> list[dict[str, Any]]:
    with fsspec.open(path, "rb") as f:
        table = pq.read_table(f, columns=["entry_id", "token_ids", "token_count", "vocab_size"])
    rows = []
    for row in table.to_pylist():
        if int(row["token_count"]) <= max_tokens:
            rows.append(row)
        if len(rows) >= num_rows:
            break
    if not rows:
        raise ValueError(f"no rows with token_count <= {max_tokens} in {path}")
    return rows


def make_batch(rows: list[dict[str, Any]]) -> tuple[Batch, int, list[int]]:
    seqs = [list(map(int, row["token_ids"])) for row in rows]
    lengths = [len(seq) for seq in seqs]
    max_len = max(lengths)
    vocab_size = max(int(row["vocab_size"]) for row in rows)
    token_ids = np.zeros((len(seqs), max_len), dtype=np.int32)
    for i, seq in enumerate(seqs):
        token_ids[i, : len(seq)] = seq
    input_ids = token_ids[:, :-1]
    target_ids = token_ids[:, 1:]
    mask = np.arange(max_len - 1)[None, :] < (np.asarray(lengths)[:, None] - 1)
    return Batch(jnp.asarray(input_ids), jnp.asarray(target_ids), jnp.asarray(mask)), vocab_size, lengths


def init_params(key: jax.Array, *, vocab_size: int, hidden: int, max_len: int) -> dict[str, jax.Array]:
    k1, k2, k3, k4 = jrandom.split(key, 4)
    scale = 0.02
    return {
        "tok": scale * jrandom.normal(k1, (vocab_size, hidden)),
        "pos": scale * jrandom.normal(k2, (max_len, hidden)),
        "w": scale * jrandom.normal(k3, (hidden, hidden)),
        "out": scale * jrandom.normal(k4, (hidden, vocab_size)),
    }


def forward(params: dict[str, jax.Array], input_ids: jax.Array) -> jax.Array:
    pos = jnp.arange(input_ids.shape[1])
    h = params["tok"][input_ids] + params["pos"][pos][None]
    h = jax.nn.gelu(jnp.einsum("bth,hk->btk", h, params["w"]))
    return jnp.einsum("bth,hv->btv", h, params["out"])


def loss_fn(params: dict[str, jax.Array], batch: Batch):
    logits = forward(params, batch.input_ids)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch.target_ids)
    total = jnp.sum(jnp.where(batch.mask, ce, 0.0))
    count = jnp.sum(batch.mask)
    loss = total / count
    acc = jnp.sum(jnp.where(batch.mask, jnp.argmax(logits, -1) == batch.target_ids, 0.0)) / count
    return loss, {"loss": loss, "loss_total": total, "token_count": count, "accuracy": acc}


def train_step(params, opt_state, optimizer, batch: Batch):
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    metrics = {**metrics, "grad_norm": optax.global_norm(grads), "param_norm": optax.global_norm(params)}
    return params, opt_state, loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("EXP299_DELTA_STREAM_DATA", DEFAULT_DATA))
    parser.add_argument("--rows", type=int, default=int(os.environ.get("EXP299_SMOKE_ROWS", "128")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("EXP299_SMOKE_MAX_TOKENS", "8096")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("EXP299_SMOKE_STEPS", "100")))
    parser.add_argument("--hidden", type=int, default=int(os.environ.get("EXP299_SMOKE_HIDDEN", "512")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("EXP299_SMOKE_LR", "1e-2")))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME", "exp299-delta-stream-gb200-smoke"))
    args = parser.parse_args()

    rows = read_rows(args.data, num_rows=args.rows, max_tokens=args.max_tokens)
    batch, vocab_size, lengths = make_batch(rows)
    print(f"[delta-stream-smoke] host={socket.gethostname()} devices={jax.devices()}", flush=True)
    print(
        f"[delta-stream-smoke] rows={len(rows)} vocab={vocab_size} max_len={max(lengths)} "
        f"tokens={int(jnp.sum(batch.mask))}",
        flush=True,
    )

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "open-athena"),
        project=os.environ.get("WANDB_PROJECT", "MarinFold"),
        name=args.wandb_name,
        group="exp299-delta-stream-smoke",
        tags=["exp299", "delta-stream", "next-token", "gb200", "smoke"],
        config=vars(args) | {"vocab_size": vocab_size, "max_length": max(lengths)},
    )
    print(f"[delta-stream-smoke] wandb_url={run.url}", flush=True)

    params = init_params(jrandom.PRNGKey(0), vocab_size=vocab_size, hidden=args.hidden, max_len=batch.input_ids.shape[1])
    optimizer = optax.adamw(args.learning_rate, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    step = jax.jit(lambda p, s, b: train_step(p, s, optimizer, b))
    for i in range(args.steps):
        params, opt_state, loss, metrics = step(params, opt_state, batch)
        scalars = {k: float(v) for k, v in metrics.items()} | {"step": i}
        print(f"[delta-stream-smoke] step={i} loss={float(loss):.6f} acc={scalars['accuracy']:.4f}", flush=True)
        wandb.log(scalars, step=i)
    wandb.finish()


if __name__ == "__main__":
    main()

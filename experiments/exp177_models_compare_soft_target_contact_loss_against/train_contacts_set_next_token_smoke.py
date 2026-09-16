# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a tiny standard next-token CE smoke on serialized contacts-set-v1 targets."""

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

from marinfold.document_structures.contacts_set_v1.format import CONTACT_SLOTS

DEFAULT_DATA = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp177_contacts_set_v1_targets/full/2026.09.15.1/shard-00000-of-03338.parquet"
)
PAD = 0
BOS = 1
EOS = 2
AA_BASE = 10
EMPTY = 40
PRESENT = 41
COARSE_BASE = 100
FINE_BASE = 400
VOCAB_SIZE = FINE_BASE + 16
AA_TO_ID = {
    aa: AA_BASE + idx
    for idx, aa in enumerate(
        [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ]
    )
}


class Batch(NamedTuple):
    input_ids: jax.Array
    target_ids: jax.Array
    mask: jax.Array


def _read_rows(path: str, *, num_rows: int, max_seq_len: int) -> list[dict[str, Any]]:
    columns = ["entry_id", "seq_len", "residue_resname", "present", "signed_coarse", "fine", "contacts_used_directed"]
    with fsspec.open(path, "rb") as f:
        table = pq.read_table(f, columns=columns)
    rows: list[dict[str, Any]] = []
    for row in table.to_pylist():
        if int(row["seq_len"]) > max_seq_len:
            continue
        rows.append(row)
        if len(rows) >= num_rows:
            break
    if not rows:
        raise ValueError(f"no rows with seq_len <= {max_seq_len} in {path}")
    return rows


def serialize_row(row: dict[str, Any]) -> list[int]:
    """Serialize one target row into ordinary categorical tokens."""
    tokens = [BOS]
    for aa, present_slots, coarse_slots, fine_slots in zip(
        row["residue_resname"], row["present"], row["signed_coarse"], row["fine"]
    ):
        tokens.append(AA_TO_ID.get(str(aa), AA_BASE))
        for is_present, signed_coarse, fine in zip(present_slots, coarse_slots, fine_slots):
            if is_present:
                tokens.extend([PRESENT, COARSE_BASE + int(signed_coarse), FINE_BASE + int(fine)])
            else:
                tokens.append(EMPTY)
    tokens.append(EOS)
    return tokens


def make_batch(rows: list[dict[str, Any]]) -> tuple[Batch, list[int]]:
    sequences = [serialize_row(row) for row in rows]
    max_len = max(len(seq) for seq in sequences)
    batch = len(sequences)
    token_ids = np.full((batch, max_len), PAD, dtype=np.int32)
    for idx, seq in enumerate(sequences):
        token_ids[idx, : len(seq)] = seq
    input_ids = token_ids[:, :-1]
    target_ids = token_ids[:, 1:]
    mask = target_ids != PAD
    return Batch(jnp.asarray(input_ids), jnp.asarray(target_ids), jnp.asarray(mask)), [len(seq) for seq in sequences]


def init_params(key: jax.Array, *, hidden: int, max_len: int) -> dict[str, jax.Array]:
    k1, k2, k3, k4, k5 = jrandom.split(key, 5)
    scale = 0.02
    return {
        "token_embed": scale * jrandom.normal(k1, (VOCAB_SIZE, hidden)),
        "pos_embed": scale * jrandom.normal(k2, (max_len, hidden)),
        "w1": scale * jrandom.normal(k3, (hidden, hidden)),
        "b1": jnp.zeros((hidden,)),
        "w_out": scale * jrandom.normal(k4, (hidden, VOCAB_SIZE)),
        "b_out": jnp.zeros((VOCAB_SIZE,)),
    }


def forward(params: dict[str, jax.Array], input_ids: jax.Array) -> jax.Array:
    positions = jnp.arange(input_ids.shape[1])
    hidden = params["token_embed"][input_ids] + params["pos_embed"][positions][None, :, :]
    hidden = jax.nn.gelu(jnp.einsum("bsh,hk->bsk", hidden, params["w1"]) + params["b1"])
    return jnp.einsum("bsh,hv->bsv", hidden, params["w_out"]) + params["b_out"]


def batch_loss(params: dict[str, jax.Array], batch: Batch) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits = forward(params, batch.input_ids)
    token_ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch.target_ids)
    masked_ce = jnp.where(batch.mask, token_ce, 0.0)
    token_count = jnp.sum(batch.mask)
    loss = jnp.sum(masked_ce) / token_count
    accuracy = jnp.sum(jnp.where(batch.mask, jnp.argmax(logits, axis=-1) == batch.target_ids, 0.0)) / token_count
    return loss, {
        "loss": loss,
        "token_count": token_count,
        "accuracy": accuracy,
        "loss_total": jnp.sum(masked_ce),
    }


def train_step(params, opt_state, optimizer, batch: Batch):
    (loss, metrics), grads = jax.value_and_grad(batch_loss, has_aux=True)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    metrics = {**metrics, "grad_norm": optax.global_norm(grads), "param_norm": optax.global_norm(params)}
    return params, opt_state, loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("EXP177_CONTACTS_SET_DATA", DEFAULT_DATA))
    parser.add_argument("--rows", type=int, default=int(os.environ.get("EXP177_SMOKE_ROWS", "2")))
    parser.add_argument("--max-seq-len", type=int, default=int(os.environ.get("EXP177_SMOKE_MAX_SEQ_LEN", "128")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("EXP177_SMOKE_STEPS", "10")))
    parser.add_argument("--hidden", type=int, default=int(os.environ.get("EXP177_SMOKE_HIDDEN", "128")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("EXP177_SMOKE_LR", "1e-2")))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME", "exp177-contacts-set-v1-next-token-smoke"))
    args = parser.parse_args()

    rows = _read_rows(args.data, num_rows=args.rows, max_seq_len=args.max_seq_len)
    batch, lengths = make_batch(rows)
    print(f"[exp177-next-token-smoke] host={socket.gethostname()} devices={jax.devices()}", flush=True)
    print(
        f"[exp177-next-token-smoke] rows={len(rows)} serialized_lengths={lengths} "
        f"tokens={int(jnp.sum(batch.mask))} vocab={VOCAB_SIZE}",
        flush=True,
    )

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "open-athena"),
        project=os.environ.get("WANDB_PROJECT", "MarinFold"),
        name=args.wandb_name,
        group="exp177-contacts-set-v1-next-token-smoke",
        tags=["exp177", "contacts-set-v1", "next-token", "gb200", "smoke"],
        config=vars(args) | {"vocab_size": VOCAB_SIZE, "serialized_lengths": lengths},
    )
    print(f"[exp177-next-token-smoke] wandb_url={run.url}", flush=True)

    params = init_params(jrandom.PRNGKey(0), hidden=args.hidden, max_len=batch.input_ids.shape[1])
    optimizer = optax.adamw(args.learning_rate, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    step_fn = jax.jit(lambda p, s, b: train_step(p, s, optimizer, b))

    for step in range(args.steps):
        params, opt_state, loss, metrics = step_fn(params, opt_state, batch)
        scalar_metrics = {name: float(value) for name, value in metrics.items()}
        scalar_metrics["step"] = step
        print(
            f"[exp177-next-token-smoke] step={step} loss={float(loss):.6f} "
            f"acc={scalar_metrics['accuracy']:.4f} grad_norm={scalar_metrics['grad_norm']:.6f}",
            flush=True,
        )
        wandb.log(scalar_metrics, step=step)

    wandb.finish()


if __name__ == "__main__":
    main()

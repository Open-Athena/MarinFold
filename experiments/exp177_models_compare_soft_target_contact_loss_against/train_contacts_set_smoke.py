# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a tiny JAX contacts-set-v1 training smoke against translated exp177 data."""

import argparse
import os
import socket
from dataclasses import dataclass
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
from marinfold.document_structures.contacts_set_v1.jax_loss import (
    NUM_FINE_BINS,
    NUM_SIGNED_COARSE_BINS,
    ContactSetLossWeights,
    contacts_set_residue_loss,
)

DEFAULT_DATA = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp177_contacts_set_v1_targets/full/2026.09.15.1/shard-00000-of-03338.parquet"
)
AA_TO_ID = {
    aa: idx
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
        ],
        start=1,
    )
}


class Batch(NamedTuple):
    aa_ids: jax.Array
    residue_mask: jax.Array
    present: jax.Array
    signed_coarse: jax.Array
    fine: jax.Array


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 64
    vocab: int = 21


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


def _make_batch(rows: list[dict[str, Any]]) -> Batch:
    max_len = max(int(row["seq_len"]) for row in rows)
    batch = len(rows)
    aa_ids = np.zeros((batch, max_len), dtype=np.int32)
    residue_mask = np.zeros((batch, max_len), dtype=bool)
    present = np.zeros((batch, max_len, CONTACT_SLOTS), dtype=bool)
    signed_coarse = np.zeros((batch, max_len, CONTACT_SLOTS), dtype=np.int32)
    fine = np.zeros((batch, max_len, CONTACT_SLOTS), dtype=np.int32)
    for b, row in enumerate(rows):
        seq_len = int(row["seq_len"])
        residue_mask[b, :seq_len] = True
        aa_ids[b, :seq_len] = [AA_TO_ID.get(str(aa), 0) for aa in row["residue_resname"]]
        present[b, :seq_len] = np.asarray(row["present"], dtype=bool)
        signed_coarse[b, :seq_len] = np.asarray(row["signed_coarse"], dtype=np.int32)
        fine[b, :seq_len] = np.asarray(row["fine"], dtype=np.int32)
    return Batch(
        aa_ids=jnp.asarray(aa_ids),
        residue_mask=jnp.asarray(residue_mask),
        present=jnp.asarray(present),
        signed_coarse=jnp.asarray(signed_coarse),
        fine=jnp.asarray(fine),
    )


def init_params(key: jax.Array, config: ModelConfig) -> dict[str, jax.Array]:
    keys = jrandom.split(key, 7)
    scale = 0.02
    return {
        "embed": scale * jrandom.normal(keys[0], (config.vocab, config.hidden)),
        "w_present": scale * jrandom.normal(keys[1], (config.hidden, CONTACT_SLOTS)),
        "b_present": jnp.zeros((CONTACT_SLOTS,)),
        "w_coarse": scale * jrandom.normal(keys[2], (config.hidden, CONTACT_SLOTS * NUM_SIGNED_COARSE_BINS)),
        "b_coarse": jnp.zeros((CONTACT_SLOTS * NUM_SIGNED_COARSE_BINS,)),
        "w_fine": scale * jrandom.normal(keys[3], (config.hidden, CONTACT_SLOTS * NUM_FINE_BINS)),
        "b_fine": jnp.zeros((CONTACT_SLOTS * NUM_FINE_BINS,)),
    }


def forward(params: dict[str, jax.Array], aa_ids: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    hidden = params["embed"][aa_ids]
    present_logits = jnp.einsum("blh,hk->blk", hidden, params["w_present"]) + params["b_present"]
    coarse_flat = jnp.einsum("blh,hk->blk", hidden, params["w_coarse"]) + params["b_coarse"]
    fine_flat = jnp.einsum("blh,hk->blk", hidden, params["w_fine"]) + params["b_fine"]
    coarse_logits = coarse_flat.reshape((*aa_ids.shape, CONTACT_SLOTS, NUM_SIGNED_COARSE_BINS))
    fine_logits = fine_flat.reshape((*aa_ids.shape, CONTACT_SLOTS, NUM_FINE_BINS))
    return present_logits, coarse_logits, fine_logits


def batch_loss(params: dict[str, jax.Array], batch: Batch) -> tuple[jax.Array, dict[str, jax.Array]]:
    present_logits, coarse_logits, fine_logits = forward(params, batch.aa_ids)

    def residue_loss(pl, cl, fl, tp, tc, tf):
        return contacts_set_residue_loss(pl, cl, fl, tp, tc, tf, ContactSetLossWeights(coarse=0.5, fine=0.5))

    losses = jax.vmap(jax.vmap(residue_loss))(
        present_logits,
        coarse_logits,
        fine_logits,
        batch.present,
        batch.signed_coarse,
        batch.fine,
    )
    masked_total = jnp.where(batch.residue_mask, losses.total, 0.0)
    residue_count = jnp.sum(batch.residue_mask)
    target_count = jnp.sum(batch.present)
    # Count each residue as 16 occupancy targets plus one geometry target for
    # each present contact.  Coarse/fine are weighted 0.5 each above.
    target_units = CONTACT_SLOTS * residue_count + target_count
    loss = jnp.sum(masked_total) / target_units

    def masked_component(values):
        return jnp.sum(jnp.where(batch.residue_mask, values, 0.0))

    coarse_sum = masked_component(losses.coarse)
    fine_sum = masked_component(losses.fine)
    return loss, {
        "loss_per_target_unit": loss,
        "residue_count": residue_count,
        "target_contacts": target_count,
        "target_units": target_units,
        "loss_total": jnp.sum(masked_total),
        "loss_present_positive_sum": masked_component(losses.present_positive),
        "loss_present_negative_sum": masked_component(losses.present_negative),
        "loss_coarse_sum": coarse_sum,
        "loss_fine_sum": fine_sum,
        "loss_coarse_ce": coarse_sum / jnp.maximum(target_count, 1),
        "loss_fine_ce": fine_sum / jnp.maximum(target_count, 1),
    }


def train_step(params, opt_state, optimizer, batch: Batch):
    (loss, metrics), grads = jax.value_and_grad(batch_loss, has_aux=True)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    grad_norm = optax.global_norm(grads)
    param_norm = optax.global_norm(params)
    metrics = {**metrics, "grad_norm": grad_norm, "param_norm": param_norm}
    return params, opt_state, loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("EXP177_CONTACTS_SET_DATA", DEFAULT_DATA))
    parser.add_argument("--rows", type=int, default=int(os.environ.get("EXP177_SMOKE_ROWS", "2")))
    parser.add_argument("--max-seq-len", type=int, default=int(os.environ.get("EXP177_SMOKE_MAX_SEQ_LEN", "128")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("EXP177_SMOKE_STEPS", "10")))
    parser.add_argument("--hidden", type=int, default=int(os.environ.get("EXP177_SMOKE_HIDDEN", "64")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("EXP177_SMOKE_LR", "1e-2")))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME", "exp177-contacts-set-v1-gb200-smoke"))
    args = parser.parse_args()

    rows = _read_rows(args.data, num_rows=args.rows, max_seq_len=args.max_seq_len)
    batch = _make_batch(rows)
    devices = jax.devices()
    print(f"[exp177-smoke] host={socket.gethostname()} devices={devices}", flush=True)
    print(
        f"[exp177-smoke] rows={len(rows)} max_len={batch.aa_ids.shape[1]} "
        f"residues={int(jnp.sum(batch.residue_mask))} targets={int(jnp.sum(batch.present))}",
        flush=True,
    )

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "open-athena"),
        project=os.environ.get("WANDB_PROJECT", "MarinFold"),
        name=args.wandb_name,
        group="exp177-contacts-set-v1-smoke",
        tags=["exp177", "contacts-set-v1", "gb200", "smoke"],
        config=vars(args) | {"devices": [str(device) for device in devices]},
    )
    print(f"[exp177-smoke] wandb_url={run.url}", flush=True)

    params = init_params(jrandom.PRNGKey(0), ModelConfig(hidden=args.hidden))
    optimizer = optax.adamw(args.learning_rate, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    step_fn = jax.jit(lambda p, s, b: train_step(p, s, optimizer, b))

    for step in range(args.steps):
        params, opt_state, loss, metrics = step_fn(params, opt_state, batch)
        scalar_metrics = {name: float(value) for name, value in metrics.items()}
        scalar_metrics["loss"] = float(loss)
        scalar_metrics["step"] = step
        print(f"[exp177-smoke] step={step} loss={scalar_metrics['loss']:.6f} grad_norm={scalar_metrics['grad_norm']:.6f}", flush=True)
        wandb.log(scalar_metrics, step=step)

    wandb.finish()


if __name__ == "__main__":
    main()

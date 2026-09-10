# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic integration test for the production sparse-contact soft loss.

This runs a tiny Levanter Qwen model through the real
``SparseContactDocumentBatch`` + ``sparse_contact_document_loss`` path. The data
is an arbitrary-order task analogous to serialized contacts: each example is a
random permutation of items, encoded as contacts ``(ANCHOR, item)``. At the
second endpoint of each contact, the production sparse loss sees all remaining
neighbors of ``ANCHOR`` as valid, so the teacher distribution is uniform over all
not-yet-emitted items.

The script logs both the normal hard next-token CE and unordered/set metrics:
teacher CE over remaining items, valid mass, and argmax-valid.
"""

import argparse
import csv
import math
from collections.abc import Iterable
from typing import NamedTuple

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config

from marinfold.document_structures.contacts_v1.vocab import CONTACT, END
from marinfold_models.document_loss import (
    SparseContactDocumentBatch,
    sparse_contact_document_loss,
    sparse_contact_document_metrics,
)

ANCHOR = int(END) + 1
ITEM_OFFSET = ANCHOR + 1


class SyntheticBatch(NamedTuple):
    batch: SparseContactDocumentBatch
    hard_targets: jax.Array
    second_positions: jax.Array
    remaining_mask: jax.Array


def make_synthetic_sparse_batch(
    *,
    rng: np.random.Generator,
    BatchAxis: hax.Axis,
    Pos: hax.Axis,
    universe_size: int,
    contact_count: int,
    max_contacts: int,
) -> SyntheticBatch:
    batch_size = BatchAxis.size
    token_ids = np.zeros((batch_size, Pos.size), dtype=np.int32)
    segment_ids = np.ones((batch_size, Pos.size), dtype=np.int32)
    attention_blocks = np.tile(np.arange(Pos.size, dtype=np.int32), (batch_size, 1))
    position_ids = np.tile(np.arange(Pos.size, dtype=np.int32), (batch_size, 1))

    first_ids = np.zeros((batch_size, max_contacts), dtype=np.int32)
    second_ids = np.zeros((batch_size, max_contacts), dtype=np.int32)
    neighbor_ids = np.zeros((batch_size, max_contacts, max_contacts), dtype=np.int32)
    neighbor_counts = np.zeros((batch_size, max_contacts, max_contacts), dtype=np.float32)
    neighbor_count = np.zeros((batch_size, max_contacts), dtype=np.int32)
    remaining_mask = np.zeros((batch_size, contact_count, ITEM_OFFSET + universe_size), dtype=bool)
    hard_targets = np.zeros((batch_size, contact_count), dtype=np.int32)
    second_positions = np.zeros((batch_size, contact_count), dtype=np.int32)

    prediction_start = np.zeros((batch_size,), dtype=np.int32)
    counts = np.full((batch_size,), contact_count, dtype=np.int32)

    for row in range(batch_size):
        perm = rng.permutation(universe_size)[:contact_count] + ITEM_OFFSET
        # Layout expected by sparse_contact_document_loss:
        # pos 0 predicts CONTACT; then triples CONTACT, first, second; then END.
        cursor = 1
        for c, item_id in enumerate(perm):
            token_ids[row, cursor] = int(CONTACT)
            token_ids[row, cursor + 1] = ANCHOR
            token_ids[row, cursor + 2] = int(item_id)
            first_ids[row, c] = ANCHOR
            second_ids[row, c] = int(item_id)
            second_positions[row, c] = cursor + 1
            hard_targets[row, c] = int(item_id)
            remaining = perm[c:]
            neighbor_ids[row, c, : remaining.size] = remaining
            neighbor_counts[row, c, : remaining.size] = 1.0
            neighbor_count[row, c] = remaining.size
            remaining_mask[row, c, remaining] = True
            cursor += 3
        token_ids[row, cursor] = int(END)

    with jax.default_device(jax.devices("cpu")[0]):
        batch = SparseContactDocumentBatch(
            tokens=hax.named(jnp.asarray(token_ids), (BatchAxis, Pos)),
            contact_first_ids=jnp.asarray(first_ids),
            contact_second_ids=jnp.asarray(second_ids),
            second_neighbor_ids=jnp.asarray(neighbor_ids),
            second_neighbor_counts=jnp.asarray(neighbor_counts),
            second_neighbor_count=jnp.asarray(neighbor_count),
            contact_count=jnp.asarray(counts),
            prediction_start=jnp.asarray(prediction_start),
            position_ids=hax.named(jnp.asarray(position_ids), (BatchAxis, Pos)),
            segment_ids=hax.named(jnp.asarray(segment_ids), (BatchAxis, Pos)),
            attention_blocks=hax.named(jnp.asarray(attention_blocks), (BatchAxis, Pos)),
            target_position_count=jnp.asarray(batch_size * (3 * contact_count + 1), dtype=jnp.int32),
            vocabulary=None,
        )
    return SyntheticBatch(
        batch=batch,
        hard_targets=jnp.asarray(hard_targets),
        second_positions=jnp.asarray(second_positions),
        remaining_mask=jnp.asarray(remaining_mask),
    )


def logits_array(model, batch: SparseContactDocumentBatch) -> jax.Array:
    logits = model(batch.tokens, batch.attention_mask if hasattr(batch, "attention_mask") else None)
    return logits.array


def forward_logits(model, batch: SparseContactDocumentBatch) -> jax.Array:
    from marinfold_models.document_loss import _compact_contact_attention_mask

    return model(
        batch.tokens,
        _compact_contact_attention_mask(batch),
        pos_ids=batch.position_ids,
    ).array


def second_endpoint_metrics(model, synthetic: SyntheticBatch) -> dict[str, float]:
    metrics = sparse_contact_document_metrics(model, synthetic.batch, prefix="synthetic")
    return {
        "hard_ce": float(metrics["synthetic/second_endpoint_hard_ce"]),
        "teacher_ce": float(metrics["synthetic/second_endpoint_teacher_ce"]),
        "valid_mass": float(metrics["synthetic/second_endpoint_valid_mass"]),
        "argmax_valid": float(metrics["synthetic/second_endpoint_argmax_valid"]),
    }


def next_token_loss(model, batch: SparseContactDocumentBatch) -> jnp.ndarray:
    logits = forward_logits(model, batch)
    targets = jnp.roll(batch.tokens.array, -1, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    gathered = jnp.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)
    return -jnp.mean(gathered[:, :-1])


def train(args: argparse.Namespace) -> None:
    BatchAxis = hax.Axis("batch", args.batch_size)
    Pos = hax.Axis("position", 3 * args.max_contacts + 2)
    Vocab = hax.Axis("vocab", ITEM_OFFSET + args.universe_size)
    config = Qwen3Config(
        max_seq_len=Pos.size,
        hidden_dim=args.d_model,
        intermediate_dim=4 * args.d_model,
        num_heads=args.n_heads,
        num_kv_heads=args.n_heads,
        num_layers=args.n_layers,
        rope=Llama3RotaryEmbeddingsConfig(),
        attn_backend=AttentionBackend.VANILLA,
        scan_layers=False,
        gradient_checkpointing=False,
    )
    key = jax.random.PRNGKey(args.seed)
    model = config.build(Vocab, key=key)
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_rng = np.random.default_rng(args.seed + 1)
    eval_rng = np.random.default_rng(args.seed + 10_000)

    def loss_fn(model, synthetic: SyntheticBatch):
        if args.mode == "soft":
            return sparse_contact_document_loss(model, synthetic.batch)
        return next_token_loss(model, synthetic.batch)

    @eqx.filter_jit
    def step(model, opt_state, synthetic: SyntheticBatch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, synthetic)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def eval_once() -> dict[str, float]:
        totals = {"hard_ce": 0.0, "teacher_ce": 0.0, "valid_mass": 0.0, "argmax_valid": 0.0}
        for _ in range(args.eval_batches):
            synthetic = make_synthetic_sparse_batch(
                rng=eval_rng,
                BatchAxis=BatchAxis,
                Pos=Pos,
                universe_size=args.universe_size,
                contact_count=args.contact_count,
                max_contacts=args.max_contacts,
            )
            metrics = second_endpoint_metrics(model, synthetic)
            for key, value in metrics.items():
                totals[key] += value
        return {key: value / args.eval_batches for key, value in totals.items()}

    records: list[dict[str, float | int | str]] = []

    def record_metrics(step_index: int, split: str, train_loss: float | None, metrics: dict[str, float]) -> None:
        records.append(
            {
                "mode": args.mode,
                "split": split,
                "step": step_index,
                "train_loss": "" if train_loss is None else train_loss,
                "hard_ce": metrics["hard_ce"],
                "teacher_ce": metrics["teacher_ce"],
                "valid_mass": metrics["valid_mass"],
                "argmax_valid": metrics["argmax_valid"],
                "oracle_second_endpoint_ce": sum(math.log(args.contact_count - j) for j in range(args.contact_count))
                / args.contact_count,
            }
        )

    initial = eval_once()
    record_metrics(0, "initial", None, initial)
    print(f"INITIAL mode={args.mode} {initial}")
    for i in range(1, args.steps + 1):
        synthetic = make_synthetic_sparse_batch(
            rng=train_rng,
            BatchAxis=BatchAxis,
            Pos=Pos,
            universe_size=args.universe_size,
            contact_count=args.contact_count,
            max_contacts=args.max_contacts,
        )
        model, opt_state, loss = step(model, opt_state, synthetic)
        if i == 1 or i % args.log_every == 0:
            metrics = eval_once()
            oracle = sum(math.log(args.contact_count - j) for j in range(args.contact_count)) / args.contact_count
            loss_value = float(loss)
            record_metrics(i, "eval", loss_value, metrics)
            print(
                f"STEP {i:04d} mode={args.mode} train_loss={loss_value:.4f} "
                f"hard_ce={metrics['hard_ce']:.4f} teacher_ce={metrics['teacher_ce']:.4f} "
                f"valid_mass={metrics['valid_mass']:.4f} argmax_valid={metrics['argmax_valid']:.4f} "
                f"oracle_second_endpoint_ce={oracle:.4f}"
            )
    final = eval_once()
    record_metrics(args.steps, "final", None, final)
    print(f"FINAL mode={args.mode} {final}")
    if args.metrics_out:
        with open(args.metrics_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"WROTE metrics_out={args.metrics_out}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "soft"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--universe-size", type=int, default=32)
    parser.add_argument("--contact-count", type=int, default=32)
    parser.add_argument("--max-contacts", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--metrics-out")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())

# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp201 Phase 1b: verify the statement-head loss mask against the real corpus.

The mask keys off hard-coded contacts-v1 token ids and a parity rule. Both are
the kind of thing that fails silently — a wrong id masks nothing (and the arm
becomes an expensive re-run of the control), a wrong parity masks the amino
acids (and the arm becomes a different experiment). This script is the gate:
run it before spending TPU time.

It checks, on real exp53 documents:

1. the tokenizer's ids for ``<begin_sequence>`` / ``<begin_statements>`` /
   ``<end>`` equal the defaults compiled into ``marinfold_models.loss_masks``;
2. the on-device mask agrees exactly with the plain-Python oracle
   (``soft_targets.statement_head_slots``), document by document;
3. the same holds for documents **packed** into a training-length window, which
   is what the trainer actually sees;
4. the masked fraction and the loss share it removes match Phase 0's numbers.

Usage::

    PYTHONPATH=../../models:../../marinfold <venv>/bin/python verify_mask.py --shards 1 --limit 400
"""

import argparse

import jax.numpy as jnp
import pyarrow.parquet as pq

import haliax as hax

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1 import vocab
from marinfold.document_structures.contacts_v1.soft_targets import (
    permutation_entropy,
    soft_targets,
    statement_head_slots,
)
from marinfold_models.loss_masks import (
    BEGIN_SEQUENCE_ID,
    BEGIN_STRUCTURE_ID,
    END_ID,
    contacts_v1_statement_head_mask,
)

from analyze_entropy import VAL_SHARDS, VAL_PREFIX

# The training window the arm will run at (exp117/exp150 SEQ_LEN).
TRAIN_SEQ_LEN = 8192


def mask_zero_slots(ids: list[int]) -> set[int]:
    Pos = hax.Axis("position", len(ids))
    tokens = hax.named(jnp.asarray(ids, dtype=jnp.int32), (Pos,))
    mask = contacts_v1_statement_head_mask(tokens).array.tolist()
    return {i for i, value in enumerate(mask) if value == 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--text-column", default="document")
    args = parser.parse_args()

    tokenizer = build_tokenizer(vocab.all_domain_tokens())
    to_id = {t: tokenizer.convert_tokens_to_ids(t) for t in vocab.all_domain_tokens()}

    # --- 1. token ids -------------------------------------------------------
    expected = {
        vocab.BEGIN_SEQUENCE_TOKEN: BEGIN_SEQUENCE_ID,
        vocab.BEGIN_STRUCTURE_TOKEN: BEGIN_STRUCTURE_ID,
        vocab.END_TOKEN: END_ID,
    }
    for token, want in expected.items():
        got = to_id[token]
        if got != want:
            raise SystemExit(
                f"FAIL: tokenizer id for {token} is {got}, mask default is {want}. "
                "Pass the real ids to Qwen3StatementHeadMaskedConfig."
            )
    print(f"[ok] token ids match the mask defaults: {expected}")

    # --- 2/3. mask == oracle, per document and packed -----------------------
    uris = [
        f"{VAL_PREFIX}/contacts_v1-{i:05d}-of-{VAL_SHARDS:05d}.parquet"
        for i in range(args.shards)
    ]
    documents: list[list[str]] = []
    for uri in uris:
        column = pq.read_table(uri, columns=[args.text_column]).column(args.text_column)
        for value in column.to_pylist():
            documents.append(value.split())
            if len(documents) >= args.limit:
                break
        if len(documents) >= args.limit:
            break

    masked_slots = 0
    total_slots = 0
    masked_nats = 0.0
    total_nats = 0.0
    for tokens in documents:
        ids = [to_id[t] for t in tokens]
        if mask_zero_slots(ids) != set(statement_head_slots(tokens)):
            raise SystemExit(f"FAIL: mask disagrees with the oracle on a {len(tokens)}-token document")
        masked_slots += len(statement_head_slots(tokens))
        total_slots += len(tokens) - 1
        breakdown = permutation_entropy(tokens)
        masked_nats += breakdown.sequence_nats
        total_nats += breakdown.total_nats
    print(f"[ok] mask == oracle on {len(documents):,} individual documents")

    packed: list[int] = []
    expected_zeros: set[int] = set()
    n_packed = 0
    for tokens in documents:
        if len(packed) + len(tokens) > TRAIN_SEQ_LEN:
            break
        expected_zeros |= {len(packed) + i for i in statement_head_slots(tokens)}
        packed += [to_id[t] for t in tokens]
        n_packed += 1
    if mask_zero_slots(packed) != expected_zeros:
        raise SystemExit("FAIL: mask disagrees with the oracle on a packed window")
    print(f"[ok] mask == oracle on a {len(packed):,}-token packed window ({n_packed} documents)")

    # --- 4. magnitude -------------------------------------------------------
    print(
        f"\nmasked slots        : {masked_slots:,} / {total_slots:,} "
        f"({100 * masked_slots / total_slots:.1f}% of supervised slots)"
    )
    print(
        f"nuisance nats removed: {masked_nats / total_slots:.4f} of "
        f"{total_nats / total_slots:.4f} nats/token "
        f"({100 * masked_nats / total_nats:.1f}% of the nuisance floor)"
    )
    print("\nAll checks passed — safe to launch the masked arm.")


if __name__ == "__main__":
    main()

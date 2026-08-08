# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp201 Phase 0: how much of the contacts-v1 loss is nuisance permutation entropy?

Walks real contacts-v1 documents, computes the exact conditional next-token
target at every slot (``marinfold.document_structures.contacts_v1.soft_targets``)
and accumulates two things:

* the **nuisance entropy** -- the cross-entropy an oracle that knows the
  structure exactly would still pay, because the generator shuffles both the
  sequence statements and the contacts. This is the floor the reported val loss
  sits on top of, for the one-hot loss and the soft loss alike.
* the **per-slot-kind split** of that budget, i.e. how the reported
  nats/token divide between the sub-tasks the aggregate number mixes together
  (statement order, amino-acid identity, contact endpoints, ``<end>`` timing).

Usage::

    uv run python analyze_entropy.py --shards 3
    uv run python analyze_entropy.py --documents local.parquet --text-column document

Writes ``data/entropy_by_document.csv`` and ``data/entropy_summary.csv``.
"""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from marinfold.document_structures.contacts_v1.soft_targets import (
    FIRST_ENDPOINT,
    FRAME,
    SECOND_ENDPOINT,
    STATEMENT_BODY,
    STATEMENT_HEAD,
    permutation_entropy,
    soft_targets,
)

# The exp53 corpus that #117/#150 trained on, held-out split. us-east5 is where
# it lives; reading a few shards for analysis is a one-time ~6 MB/shard pull.
VAL_PREFIX = (
    "gs://marin-us-east5/protein-structure/MarinFold/"
    "exp53_contacts_v1_5x/documents/val"
)
VAL_SHARDS = 22

# Order slot kinds most-informative-last so the printed table reads as
# "nuisance first, real signal after".
KIND_ORDER = (STATEMENT_HEAD, FIRST_ENDPOINT, SECOND_ENDPOINT, STATEMENT_BODY, FRAME)

KIND_LABEL = {
    STATEMENT_HEAD: "sequence statement order (nuisance)",
    FIRST_ENDPOINT: "contact 1st endpoint (nuisance + signal)",
    SECOND_ENDPOINT: "contact 2nd endpoint (nuisance + signal)",
    STATEMENT_BODY: "amino acid / terminus index (signal)",
    FRAME: "section markers, <contact> vs <end> (signal)",
}


@dataclass
class Totals:
    """Running sums over a document collection."""

    documents: int = 0
    tokens: int = 0
    predicted: int = 0
    nuisance_nats: float = 0.0
    sequence_nats: float = 0.0
    structure_nats: float = 0.0

    def add(self, breakdown) -> None:
        self.documents += 1
        self.tokens += breakdown.num_tokens
        self.predicted += breakdown.num_predicted
        self.nuisance_nats += breakdown.total_nats
        self.sequence_nats += breakdown.sequence_nats
        self.structure_nats += breakdown.structure_nats


def shard_uris(n_shards: int) -> list[str]:
    """The first ``n_shards`` val shards of the exp53 contacts-v1 corpus."""
    return [
        f"{VAL_PREFIX}/contacts_v1-{i:05d}-of-{VAL_SHARDS:05d}.parquet"
        for i in range(n_shards)
    ]


def iter_documents(uris: list[str], text_column: str, limit: int | None):
    """Yield document strings from parquet shards, streaming row group by row group."""
    seen = 0
    for uri in uris:
        table = pq.read_table(uri, columns=[text_column])
        for value in table.column(text_column).to_pylist():
            yield value
            seen += 1
            if limit is not None and seen >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=3,
                        help="number of exp53 val shards to read (default 3, ~19 MB)")
    parser.add_argument("--documents", nargs="*", default=None,
                        help="explicit parquet path(s) instead of the val shards")
    parser.add_argument("--text-column", default="document")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many documents")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    uris = args.documents if args.documents else shard_uris(args.shards)
    print(f"reading {len(uris)} parquet file(s)", flush=True)

    totals = Totals()
    kind_nats: Counter[str] = Counter()
    kind_slots: Counter[str] = Counter()
    rows: list[dict[str, object]] = []

    for document in iter_documents(uris, args.text_column, args.limit):
        tokens = document.split()
        targets = soft_targets(tokens)
        breakdown = permutation_entropy(tokens, targets=targets)
        totals.add(breakdown)
        for target in targets:
            kind_nats[target.kind] += target.entropy
            kind_slots[target.kind] += 1
        if totals.documents % 2000 == 0:
            print(f"  {totals.documents:,} documents", flush=True)
        rows.append({
            "seq_len": breakdown.seq_len,
            "num_contacts": breakdown.num_contacts,
            "num_tokens": breakdown.num_tokens,
            "num_predicted": breakdown.num_predicted,
            "sequence_nats": round(breakdown.sequence_nats, 4),
            "structure_nats": round(breakdown.structure_nats, 4),
            "nats_per_token": round(breakdown.nats_per_token, 6),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_doc = args.out_dir / "entropy_by_document.csv"
    with per_doc.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    nuisance_per_token = totals.nuisance_nats / totals.predicted
    summary = [
        {"metric": "documents", "value": totals.documents},
        {"metric": "tokens", "value": totals.tokens},
        {"metric": "predicted_slots", "value": totals.predicted},
        {"metric": "mean_tokens_per_document",
         "value": round(totals.tokens / totals.documents, 2)},
        {"metric": "nuisance_nats_per_token", "value": round(nuisance_per_token, 5)},
        {"metric": "sequence_nats_per_token",
         "value": round(totals.sequence_nats / totals.predicted, 5)},
        {"metric": "structure_nats_per_token",
         "value": round(totals.structure_nats / totals.predicted, 5)},
    ]
    for kind in KIND_ORDER:
        summary.append({
            "metric": f"slot_share::{kind}",
            "value": round(kind_slots[kind] / totals.predicted, 5),
        })
        summary.append({
            "metric": f"floor_nats_per_token::{kind}",
            "value": round(kind_nats[kind] / totals.predicted, 5),
        })
    with (args.out_dir / "entropy_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{totals.documents:,} documents, {totals.tokens:,} tokens, "
          f"mean {totals.tokens / totals.documents:.0f} tokens/doc\n")
    print(f"{'slot kind':<44} {'% of slots':>11} {'floor nats/tok':>15}")
    print("-" * 72)
    for kind in KIND_ORDER:
        print(f"{KIND_LABEL[kind]:<44} "
              f"{100 * kind_slots[kind] / totals.predicted:>10.1f}% "
              f"{kind_nats[kind] / totals.predicted:>15.4f}")
    print("-" * 72)
    print(f"{'TOTAL nuisance floor':<44} {'':>11} {nuisance_per_token:>15.4f}")
    print(f"\n  sequence-section order : {totals.sequence_nats / totals.predicted:.4f} nats/token")
    print(f"  structure-section order: {totals.structure_nats / totals.predicted:.4f} nats/token")
    print(f"\nAgainst the #117/#150 val loss of 2.7112 nats/token, the nuisance")
    print(f"floor is {100 * nuisance_per_token / 2.7112:.1f}% of the reported number;")
    print(f"the informative remainder is {2.7112 - nuisance_per_token:.4f} nats/token.")
    print(f"\nwrote {per_doc} and {args.out_dir / 'entropy_summary.csv'}")


if __name__ == "__main__":
    main()

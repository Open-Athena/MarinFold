# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the chain reconstruction against the published corpus's own metadata.

`complex_sections.label` derives chain membership from a document's `<n-term>` /
`<c-term>` statements alone. The published corpus independently records
`num_chains`, `chain_lengths` and `contacts_emitted_inter_chain` for every
document, computed from the source structures rather than from the document text
-- so comparing the two is a real check, not a restatement of the same code.

Every row is compared and any mismatch fails the script. The inter-chain loss
split is this experiment's headline measurement, and a reconstruction that were
merely right on average would quietly bias it.

    uv run python verify_sections.py --shard <local or s3 shard>
"""

import argparse
import collections
import csv
import json
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

from experiments.exp343_models_complex_corpus_training.complex_sections import (
    Role,
    label,
)

COLUMNS = [
    "document",
    "num_chains",
    "chain_lengths",
    "contacts_emitted",
    "contacts_emitted_inter_chain",
]


def check(shard: str, limit: int | None) -> dict:
    """Compare every document's reconstruction with its recorded metadata."""
    mismatches: collections.Counter[str] = collections.Counter()
    role_tokens: collections.Counter[str] = collections.Counter()
    checked = 0
    intra = inter = unresolved = unresolved_documents = 0
    with fsspec.open(shard, "rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(
            batch_size=512, columns=COLUMNS
        ):
            for row in batch.to_pylist():
                sections = label(row["document"])
                if sections.num_chains != row["num_chains"]:
                    mismatches["num_chains"] += 1
                observed = sorted(
                    collections.Counter(sections.chain_of_index.values()).values()
                )
                if observed != sorted(row["chain_lengths"]):
                    mismatches["chain_lengths"] += 1
                if sections.contacts_inter != row["contacts_emitted_inter_chain"]:
                    mismatches["contacts_inter"] += 1
                total = (
                    sections.contacts_intra
                    + sections.contacts_inter
                    + sections.contacts_unresolved
                )
                if total != row["contacts_emitted"]:
                    mismatches["contacts_total"] += 1
                for role in sections.roles:
                    role_tokens[role.value] += 1
                intra += sections.contacts_intra
                inter += sections.contacts_inter
                unresolved += sections.contacts_unresolved
                unresolved_documents += sections.contacts_unresolved > 0
                checked += 1
                if limit is not None and checked >= limit:
                    break
            if limit is not None and checked >= limit:
                break
    return {
        "shard": shard,
        "documents_checked": checked,
        "documents_exact": checked - sum(mismatches.values()),
        "mismatch_num_chains": mismatches["num_chains"],
        "mismatch_chain_lengths": mismatches["chain_lengths"],
        "mismatch_contacts_inter": mismatches["contacts_inter"],
        "mismatch_contacts_total": mismatches["contacts_total"],
        "contacts_intra": intra,
        "contacts_inter": inter,
        "contacts_unresolved": unresolved,
        "documents_with_unresolved_contacts": unresolved_documents,
        **{f"tokens_{role.value}": role_tokens[role.value] for role in Role},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/sections_check.csv"))
    arguments = parser.parse_args()
    result = check(arguments.shard, arguments.limit)
    failures = {
        key: value
        for key, value in result.items()
        if key.startswith("mismatch_") and value
    }
    if failures:
        raise ValueError(
            f"chain reconstruction disagrees with the published metadata: {failures}"
        )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2))
    print(f"-> {arguments.out}")


if __name__ == "__main__":
    main()

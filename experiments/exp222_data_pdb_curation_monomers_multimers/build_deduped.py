# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a redundancy-controlled corpus that needs no sampling logic to train on.

The monomer and multimer corpora keep every passing chain and assembly, which
is right for a source dataset and wrong for a training loop: PDB redundancy is
extreme (the largest 40%-identity group here holds **4,055** near-duplicate
documents), so a uniform pass over the raw corpus spends most of its gradient
on lysozyme. Protenix/AF3 handle this with cluster-weighted *sampling*; this
script bakes the same intent into the data so a consumer can shuffle and read.

**Grouping.** A document's key is what makes it redundant with another:

* monomer -- its chain's RCSB 40%-identity cluster id.
* multimer -- the **sorted tuple** of its chains' cluster ids, so composition
  and stoichiometry both matter: a homodimer ``(7, 7)``, a homotetramer
  ``(7, 7, 7, 7)`` and a heterodimer ``(7, 12)`` are three different things to
  learn, not three copies of one.
* anything with an unclustered chain (short peptides are absent from the RCSB
  file) -- its exact resolved-sequence hash, so it still dedupes, just by
  identity rather than homology.

A monomer and a multimer of the same protein therefore both survive. That is
deliberate: one teaches the fold, the other the interface.

**Representative.** Best (lowest) resolution first, then most residues, then
``entry_id`` for determinism. Structures with no reported resolution (NMR) sort
after every X-ray structure in the same group -- if they are the only member,
they are still kept.

**Output** is shuffled with a fixed seed before sharding, so a sequential read
is already in random order and is not grouped by PDB id or cluster.

Usage::

    uv run python build_deduped.py --max-per-cluster 1
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from curate_and_generate import DOC_SCHEMA, ShardWriter


# Sorts after every real resolution, so NMR entries lose a tie-break against
# an X-ray structure of the same protein but still win an empty group.
_NO_RESOLUTION = float("inf")

# Only what grouping and ranking need; the documents themselves are re-read
# per shard at write time so the full text is never all in memory at once.
_KEY_COLUMNS = [
    "entry_id", "subset", "cluster_ids", "resolved_seq_sha1",
    "resolution", "seq_len",
]


def group_key(cluster_ids, resolved_seq_sha1: str) -> tuple:
    """The identity a document is deduplicated against."""
    if not cluster_ids or any(c < 0 for c in cluster_ids):
        return ("seq", resolved_seq_sha1)
    return ("cluster", tuple(sorted(cluster_ids)))


def rank(resolution, seq_len: int, entry_id: str) -> tuple:
    """Sort key for choosing a group's representative (lower is better)."""
    return (
        resolution if resolution is not None else _NO_RESOLUTION,
        -seq_len,
        entry_id,
    )


def collect(root: Path, subsets: list[str]) -> dict[tuple, list[tuple]]:
    """Map every group key to its members as ``(rank, shard, row_index)``."""
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for subset in subsets:
        directory = root / "docs" / subset
        if not directory.is_dir():
            continue
        for shard in sorted(directory.glob("*.parquet")):
            table = pq.read_table(shard, columns=_KEY_COLUMNS)
            entry_ids = table.column("entry_id").to_pylist()
            cluster_ids = table.column("cluster_ids").to_pylist()
            hashes = table.column("resolved_seq_sha1").to_pylist()
            resolutions = table.column("resolution").to_pylist()
            seq_lens = table.column("seq_len").to_pylist()
            for i in range(table.num_rows):
                key = group_key(cluster_ids[i], hashes[i])
                groups[key].append(
                    (rank(resolutions[i], seq_lens[i], entry_ids[i]), str(shard), i)
                )
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/exp222_pdb_curation"))
    parser.add_argument(
        "--max-per-cluster", type=int, default=1,
        help="documents to keep per group (1 = strict one-per-cluster)",
    )
    parser.add_argument("--out-subset", default="deduped")
    parser.add_argument("--rows-per-shard", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=222)
    args = parser.parse_args(argv)

    groups = collect(args.root, ["monomers", "multimers"])
    print(f"{len(groups)} distinct groups over "
          f"{sum(len(v) for v in groups.values())} documents", flush=True)

    # Choose each group's representatives, then shuffle globally so the output
    # is trainable by sequential read.
    chosen: list[tuple[str, int]] = []
    for members in groups.values():
        members.sort()
        for _, shard, index in members[: args.max_per_cluster]:
            chosen.append((shard, index))
    random.Random(args.seed).shuffle(chosen)
    print(f"keeping {len(chosen)} documents "
          f"(max {args.max_per_cluster} per group)", flush=True)

    # Re-read shard by shard so no more than one shard's document text is
    # resident, then place each row at its shuffled destination.
    by_shard: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for destination, (shard, index) in enumerate(chosen):
        by_shard[shard].append((index, destination))

    output: list[dict | None] = [None] * len(chosen)
    for shard, pairs in by_shard.items():
        table = pq.read_table(shard)
        rows = table.take(pa.array([i for i, _ in pairs])).to_pylist()
        for row, (_, destination) in zip(rows, pairs):
            output[destination] = row
        del table, rows
    assert all(r is not None for r in output), "a chosen row was not materialised"

    writer = ShardWriter(
        args.root / "docs" / args.out_subset, DOC_SCHEMA, args.rows_per_shard
    )
    for start in range(0, len(output), args.rows_per_shard):
        writer.add(output[start : start + args.rows_per_shard])
    writer.close()

    tokens = sum(r["num_tokens"] for r in output)
    subsets: dict[str, int] = defaultdict(int)
    for r in output:
        subsets[r["subset"]] += 1
    print(
        f"wrote {writer.total} documents / {tokens/1e6:.1f} M tokens to "
        f"{args.root / 'docs' / args.out_subset}  ({dict(subsets)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

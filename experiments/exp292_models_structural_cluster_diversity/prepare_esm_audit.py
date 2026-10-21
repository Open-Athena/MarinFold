"""Sample current ESM training metadata, retaining original clustering lineage.

Physical shard numbers are not a stable join to exp91 materialization plan chunks.
Instead, project cluster IDs, current anchor hashes and quality fields from the
actual decontaminated training documents. The worker subsequently verifies each
anchor and cluster cardinality against the original complete membership TSV.
Only small metadata columns cross regions, never full document shards.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from sample_esm import choose_clusters

COLUMNS = [
    "entry_id",
    "seq_cluster_id",
    "seq_len",
    "global_plddt",
    "ptm",
    "plddt_std",
    "cluster_size",
    "split",
]


def read_metadata(index: int, cache: Path, corpus: Path) -> tuple[pa.Table, dict]:
    """Read current anchor metadata, without assuming physical shard alignment."""
    source = corpus / f"shard-{index:05d}-of-03338.parquet"
    projected = cache / f"current-{index:05d}.parquet"
    if source.exists():
        current = pq.read_table(source, columns=COLUMNS)
        corpus_source = str(source)
    else:
        corpus_source = (
            "buckets/open-athena/MarinFold/data/document_structures/contacts_v1_esm_atlas_decontam/train/"
            + source.name
        )
        if projected.exists():
            current = pq.read_table(projected)
        else:
            with HfFileSystem(token=False).open(
                corpus_source, "rb", block_size=256 << 10
            ) as handle:
                current = pq.read_table(handle, columns=COLUMNS, pre_buffer=False)
    current = current.filter(pc.equal(current["split"], "train"))
    pq.write_table(current, projected)
    plan = pa.table(
        {
            "protein_hash": current["entry_id"],
            "cluster_id": current["seq_cluster_id"],
            "seq_len": current["seq_len"],
            "mean_plddt": pc.divide(current["global_plddt"], 100),
            "ptm": current["ptm"],
            "plddt_std": current["plddt_std"],
            "cluster_size": current["cluster_size"],
        }
    )
    return plan, {
        "corpus_shard": index,
        "source": corpus_source,
        "projected_bytes": projected.stat().st_size,
        "projected_sha256": hashlib.sha256(projected.read_bytes()).hexdigest(),
        "training_ids": plan.num_rows,
    }


def main() -> None:
    """Save stratified clusters and provenance for a fresh roughly 1,000-cluster audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=293)
    parser.add_argument("--corpus-shards", type=int, default=100)
    parser.add_argument("--per-bin", type=int, default=112)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    indices = sorted(
        int(i)
        for i in np.random.default_rng(args.seed).choice(
            np.arange(1, 3338), args.corpus_shards, replace=False
        )
    )
    plans, provenance = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for plan, source in pool.map(
            partial(read_metadata, cache=args.cache, corpus=args.corpus), indices
        ):
            plans.append(plan)
            provenance.append(source)
    all_rows = pa.concat_tables(plans)
    hashes = all_rows["protein_hash"].to_pylist()
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate current training anchor across sampled shards")
    chosen = choose_clusters(all_rows, set(hashes), args.per_bin, args.seed)
    pq.write_table(pa.Table.from_pylist(chosen), args.output / "plan.parquet")
    pq.write_table(
        pa.table({"entry_id": [r["protein_hash"] for r in chosen]}),
        args.output / "retained.parquet",
    )
    record = {
        "seed": args.seed,
        "per_bin": args.per_bin,
        "corpus_shards": args.corpus_shards,
        "shard_sampling": "simple random sample without replacement from current corpus shards 1..3337; development shard 0 excluded",
        "cluster_sampling": "hash-ranked within length/cluster-size strata among eligible current anchors in sampled shards; original membership validated by worker",
        "selected_clusters": len(chosen),
        "sampled_current_anchors": all_rows.num_rows,
        "projected_metadata_bytes": sum(r["projected_bytes"] for r in provenance),
        "provenance": provenance,
    }
    (args.output / "sampling.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()

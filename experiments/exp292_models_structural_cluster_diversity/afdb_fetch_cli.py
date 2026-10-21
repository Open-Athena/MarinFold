"""Run source-local AFDB structure validation with Iris and Zephyr."""

import argparse
import os
from functools import partial

from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from afdb_fetch_rows import fetch_plan_files

PLAN_COLUMNS = (
    "entry_id",
    "gcs_uri",
    "struct_cluster_id",
    "seq_cluster_id",
    "global_plddt",
    "seq_len",
    "split",
    "uniprot_accession",
    "tax_id",
    "organism_name",
    "is_anchor",
    "reservoir_rank",
    "n_anchors",
    "n_candidates",
)


def main() -> None:
    """Map source validation over one parquet input shard per task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-docs", type=int)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--fetch-concurrency", type=int, default=32)
    parser.add_argument("--worker-cpu", type=float, default=1)
    parser.add_argument("--worker-memory", default="4g")
    parser.add_argument("--worker-disk", default="8g")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument(
        "--preemptible", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    rows = Dataset.from_files(args.input)
    if args.num_docs is not None:
        rows = rows.take_per_shard(1)
    output = rows.map_shard(
        partial(
            fetch_plan_files,
            columns=PLAN_COLUMNS,
            fetch_concurrency=args.fetch_concurrency,
            row_limit_per_shard=args.num_docs,
        )
    )
    if args.num_docs is not None:
        output = output.reshard(1).take_per_shard(args.num_docs)
    if "{shard" not in args.output:
        output = output.reshard(1)
    result = output.write_parquet(args.output)
    context = ZephyrContext(
        max_workers=args.max_workers,
        resources=ResourceConfig(
            cpu=args.worker_cpu,
            ram=args.worker_memory,
            disk=args.worker_disk,
            regions=[args.region],
            preemptible=args.preemptible,
        ),
    )
    context.execute(result)
    if os.path.splitext(args.output)[1] != ".parquet":
        raise ValueError("AFDB validated-structure output must be parquet")


if __name__ == "__main__":
    main()

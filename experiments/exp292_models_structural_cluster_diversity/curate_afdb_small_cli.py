"""Curate the untrained AFDB clusters with Iris and Zephyr.

One task per validated shard. Unlike the anchored arm this stage runs no
alignments — each cluster holds at most two members and there is no anchor to
compare them against — so a single-core worker is the right shape.
"""

import argparse
import os
from functools import partial

from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from curate_afdb_small import curate_shard_files


def main() -> None:
    """Map unanchored-cluster curation over one validated shard per task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aux-prefix", required=True)
    parser.add_argument("--mmseqs-archive", required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--work", default="exp292-small-curate-work")
    parser.add_argument("--max-workers", type=int, default=256)
    parser.add_argument("--screen-threads", type=int, default=1)
    parser.add_argument("--worker-cpu", type=float, default=1)
    parser.add_argument("--worker-memory", default="4g")
    parser.add_argument("--worker-disk", default="8g")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument(
        "--preemptible", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if os.path.splitext(args.output)[1] != ".parquet":
        raise ValueError("Untrained-cluster selection output must be parquet")
    shards = Dataset.from_files(args.input)
    selected = shards.map_shard(
        partial(
            curate_shard_files,
            archive_uri=args.mmseqs_archive,
            reference_uris=tuple(args.reference),
            aux_prefix=args.aux_prefix,
            work=args.work,
            threads=args.screen_threads,
        )
    )
    if "{shard" not in args.output:
        selected = selected.reshard(1)
    result = selected.write_parquet(args.output)
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


if __name__ == "__main__":
    main()

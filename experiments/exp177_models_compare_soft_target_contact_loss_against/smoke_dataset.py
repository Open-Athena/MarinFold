# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build one staged pilot epoch and report loader throughput and packing."""

import argparse
import asyncio
import dataclasses
import json
import os
import time

import numpy as np

from premade_contacts_dataset import FixedQuotaPremadeContactsDataset


DEFAULT_DATA_PREFIX = (
    os.environ.get("EXP147_BUCKET", "gs://marin-us-east5").rstrip("/")
    + "/protein-structure/MarinFold/"
    + "exp147_on_the_fly_contacts_v1_pilot/pilot_data/contacts"
)


async def _consume_epoch(
    dataset: FixedQuotaPremadeContactsDataset,
    *,
    batch_size: int,
) -> int:
    num_examples = dataset.num_shards * dataset.examples_per_shard
    loss_tokens = 0
    for start in range(0, num_examples, batch_size):
        stop = min(start + batch_size, num_examples)
        examples = await dataset.get_batch(range(start, stop))
        loss_tokens += sum(
            int(np.asarray(example.loss_weight).sum()) for example in examples
        )
    return loss_tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-prefix", default=DEFAULT_DATA_PREFIX)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--examples-per-shard", type=int, default=2650)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-utilization", type=float)
    args = parser.parse_args(argv)
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.examples_per_shard <= 0:
        parser.error("--examples-per-shard must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.min_utilization is not None and not 0 <= args.min_utilization <= 1:
        parser.error("--min-utilization must be in [0, 1]")

    dataset = FixedQuotaPremadeContactsDataset(
        data_prefix=args.data_prefix,
        num_shards=args.num_shards,
        examples_per_shard=args.examples_per_shard,
        seed=0,
        max_seq_len=8192,
    )
    started = time.perf_counter()
    loss_tokens = asyncio.run(_consume_epoch(dataset, batch_size=args.batch_size))
    elapsed_seconds = time.perf_counter() - started

    stats = dataclasses.asdict(dataset.stats)
    stats["packing_utilization"] = dataset.stats.packing_utilization
    result = {
        "data_prefix": args.data_prefix,
        "num_shards": args.num_shards,
        "examples_per_shard": args.examples_per_shard,
        "examples": args.num_shards * args.examples_per_shard,
        "loss_tokens": loss_tokens,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": (
            args.num_shards * args.examples_per_shard / elapsed_seconds
        ),
        "stats": stats,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if (
        args.min_utilization is not None
        and dataset.stats.packing_utilization < args.min_utilization
    ):
        raise ValueError(
            f"packing utilization {dataset.stats.packing_utilization:.4f} is below "
            f"the required {args.min_utilization:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

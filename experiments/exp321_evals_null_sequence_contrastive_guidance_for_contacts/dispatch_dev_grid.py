#!/usr/bin/env python
"""Submit the preregistered development guidance grid in one controller session."""

import argparse
from types import SimpleNamespace

from fray.iris_backend import FrayIrisClient
from iris.cli.connect import open_iris_client

from dispatch_cw import MODEL, OUT, request

DEV_CONFIGS = (
    ("dev_g0_pa_pos", 0.0, "polyala", "positions", False, 1.0),
    ("dev_g025_pa_pos", 0.25, "polyala", "positions", False, 1.0),
    ("dev_g05_pa_pos", 0.5, "polyala", "positions", False, 1.0),
    ("dev_g1_pa_pos", 1.0, "polyala", "positions", False, 1.0),
    ("dev_g2_pa_pos", 2.0, "polyala", "positions", False, 1.0),
    ("dev_ratio_pa_pos", 0.0, "polyala", "positions", True, 1.0),
    ("dev_g05_pa_all", 0.5, "polyala", "all", False, 1.0),
    ("dev_t11_pa_pos", 0.0, "polyala", "positions", False, 1.1),
)


def main() -> None:
    """Submit every mode with identical targets, realizations, and rollout count."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = []
    for mode, gamma, null_kind, scope, pure_ratio, temperature in DEV_CONFIGS:
        config = SimpleNamespace(
            label=mode.replace("_", "-"),
            mode=mode,
            out=args.out,
            model=args.model,
            selection="dev",
            null=null_kind,
            scope=scope,
            gamma=gamma,
            pure_ratio=pure_ratio,
            temperature=temperature,
            top_p=0.95,
            n_rollouts=args.n_rollouts,
            batch_size=args.batch_size,
            limit=None,
        )
        jobs.extend(request(shard, args.num_shards, config) for shard in range(args.num_shards))
    print(f"[exp321] development grid: {len(DEV_CONFIGS)} modes, {len(jobs)} jobs")
    if args.dry_run:
        for job in jobs:
            print(job.name)
        return
    with open_iris_client(cluster_name="cw-rno2a", workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        for index, job in enumerate(jobs, 1):
            submitted = client.submit(job)
            print(
                f"[exp321] {index}/{len(jobs)} {job.name} -> "
                f"{getattr(submitted, 'id', submitted)}",
                flush=True,
            )


if __name__ == "__main__":
    main()

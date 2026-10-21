"""Retry a failed independent shard without interrupting healthy workers."""

import argparse
import json
import time
from pathlib import Path

import fsspec
from fray.current_client import current_client

from driver import ROOT, collect, payload, put, submit_worker, wait_jobs

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v1")
    parser.add_argument("--checkpoint")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--cap-check", action="store_true")
    args = parser.parse_args()
    root = ROOT + "/" + args.run_id
    checkpoints = json.loads((HERE / "checkpoints.json").read_text())
    if args.cap_check:
        caps = json.loads((HERE / "capped_rollouts.json").read_text())
        for name in (
            "targets.json",
            "checkpoints.json",
            "code_manifest.json",
            "capped_rollouts.json",
        ):
            put(f"{root}/inputs/{name}", (HERE / name).read_bytes())
        jobs = []
        for checkpoint in checkpoints:
            keys = sorted(
                {
                    f"{r['dataset']}__{r['stem']}"
                    for r in caps
                    if r["model"] == checkpoint["label"]
                }
            )
            if not keys:
                continue
            put(f"{root}/inputs/worker-{checkpoint['label']}.zip", payload(checkpoint))
            jobs.append(
                submit_worker(
                    current_client(),
                    checkpoint,
                    root,
                    0,
                    1,
                    smoke=False,
                    extra_args=["--full-context", "--include", *keys],
                )
            )
        wait_jobs(jobs)
        print(
            json.dumps({"event": "cap_check_complete", "units": len(caps)}), flush=True
        )
        return
    if args.collect_only:
        targets = json.loads((HERE / "targets.json").read_text())
        deadline = time.monotonic() + 3600
        while True:
            counts = []
            for checkpoint in checkpoints:
                filesystem, prefix = fsspec.core.url_to_fs(
                    f"{root}/rollout/{checkpoint['label']}/complete"
                )
                filesystem.invalidate_cache()
                counts.append(len(filesystem.glob(prefix + "/*.json")))
            if all(count == len(targets) for count in counts):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Still incomplete after one hour: {counts}")
            print(
                json.dumps({"event": "awaiting_completion", "counts": counts}),
                flush=True,
            )
            time.sleep(15)
        put(
            f"{root}/inputs/collection_code_manifest.json",
            (HERE / "code_manifest.json").read_bytes(),
        )
        collect(root, checkpoints, targets)
        return
    if args.checkpoint is None or args.shard is None:
        raise ValueError("A repair requires --checkpoint and --shard")
    checkpoint = next(c for c in checkpoints if c["job_label"] == args.checkpoint)
    put(
        f"{root}/inputs/repair_code_manifest.json",
        (HERE / "code_manifest.json").read_bytes(),
    )
    wait_jobs(
        [
            submit_worker(
                current_client(), checkpoint, root, args.shard, args.shards, smoke=False
            )
        ]
    )


if __name__ == "__main__":
    main()

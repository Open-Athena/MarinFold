"""Repack the scale run's artifacts into Hub-sized shards and upload them.

The run holds about 1.8M objects totalling 353 GB. The Hub handles a few hundred
large files far better than a million small ones, so each kind is concatenated
into roughly equal shards before upload. The shard plan is computed once and
stored beside the run, so independent workers partition identically and a
restart resumes by asking the Hub which shards already exist rather than by
trusting local state.

Repacking is lossless: parquet kinds keep their columns, and each generated
`.npz` becomes one row carrying its flattened Ca coordinates plus the shape
needed to restore them.
"""

import argparse
import io
import json
import os
import time
from pathlib import Path

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError

KINDS = {
    "candidates": "folded/candidates",
    "sequences": "folded/sequences",
    "generated": "generated",
}
SUFFIX = {"candidates": ".parquet", "sequences": ".parquet", "generated": ".npz"}


def open_fs(root: str, cluster: str | None = None):
    """Resolve the run root to a filesystem and its bucket-relative path.

    In-pod the task environment already carries the object-store credentials, so
    bare fsspec resolves. From the workstation there is no such environment, and
    `--cluster` selects launch.py's kubectl-backed client instead; that import is
    deliberately lazy because the CPU bootstrap does not install iris or fray.
    """
    if cluster:
        from launch import storage_filesystem

        return storage_filesystem(cluster), root.removeprefix("s3://")
    return fsspec.core.url_to_fs(root)


def build_plan(fs, base: str, target_bytes: int) -> dict:
    """Group each kind's objects into shards of roughly target_bytes."""
    listing = fs.find(base + "/cases", detail=True)
    plan = {"target_bytes": target_bytes, "kinds": {}}
    for kind, subdir in KINDS.items():
        entries = sorted(
            (path, info.get("size") or 0)
            for path, info in listing.items()
            if f"/{subdir}/" in path and path.endswith(SUFFIX[kind])
        )
        shards, current, total = [], [], 0
        for path, size in entries:
            current.append(path)
            total += size
            if total >= target_bytes:
                shards.append({"sources": current, "bytes": total})
                current, total = [], 0
        if current:
            shards.append({"sources": current, "bytes": total})
        plan["kinds"][kind] = shards
        print(
            f"{kind:12} {len(entries):>9,} objects  "
            f"{sum(s for _, s in entries) / 1e9:>8.2f} GB  -> {len(shards):>5} shards",
            flush=True,
        )
    return plan


def case_of(path: str) -> str:
    """Recover the case id from any object path under cases/."""
    return path.split("/cases/", 1)[1].split("/", 1)[0]


def write_parquet_shard(fs, sources: list[str], destination: Path) -> int:
    """Concatenate source parquet files, preserving the first file's schema."""
    writer, rows = None, 0
    try:
        for source in sources:
            table = pq.read_table(io.BytesIO(fs.cat(source)))
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows


def write_generated_shard(
    fs, sources: list[str], destination: Path, cases: dict
) -> int:
    """Turn each generated .npz into one lossless row with its Ca coordinates."""
    schema = pa.schema(
        [
            ("case", pa.string()),
            ("batch", pa.int32()),
            ("condition", pa.string()),
            ("length", pa.int32()),
            ("batch_size", pa.int32()),
            ("ca_shape", pa.list_(pa.int32())),
            ("ca_flat", pa.list_(pa.float32())),
            ("timings_json", pa.string()),
        ]
    )
    writer, rows = pq.ParquetWriter(destination, schema, compression="zstd"), 0
    try:
        for source in sources:
            case = case_of(source)
            batch = int(Path(source).stem.split("-")[1])
            with np.load(io.BytesIO(fs.cat(source)), allow_pickle=False) as archive:
                ca = np.asarray(archive["ca"], dtype=np.float32)
                timings = str(archive["timings_json"])
            meta = cases.get(case, {})
            writer.write_table(
                pa.table(
                    {
                        "case": [case],
                        "batch": [batch],
                        "condition": [meta.get("condition")],
                        "length": [meta.get("length")],
                        "batch_size": [int(ca.shape[0])],
                        "ca_shape": [list(int(d) for d in ca.shape)],
                        "ca_flat": [ca.reshape(-1).tolist()],
                        "timings_json": [timings],
                    },
                    schema=schema,
                )
            )
            rows += 1
    finally:
        writer.close()
    return rows


def commit_batch(
    api: HfApi, repo: str, pending: list[tuple[CommitOperationAdd, Path]]
) -> None:
    """Upload a group of shards under a single commit, then drop the local copies.

    The Hub caps repository commits at 320 per hour, so one commit per shard
    fails as soon as several workers run at once: the bytes are not the limit,
    the commits are. Large files pre-upload to LFS outside the commit, so
    batching costs no bandwidth and divides the commit count by the batch size.
    A 429 is a rate limit rather than a transient error and the Hub asks for
    roughly an hour, so back off on that scale instead of in seconds.
    """
    if not pending:
        return
    operations = [operation for operation, _ in pending]
    names = f"{operations[0].path_in_repo} .. {operations[-1].path_in_repo}"
    for attempt in range(8):
        try:
            api.preupload_lfs_files(
                repo_id=repo, additions=operations, repo_type="dataset"
            )
            api.create_commit(
                repo_id=repo,
                operations=operations,
                repo_type="dataset",
                commit_message=f"exp278 upload: {len(operations)} shards",
            )
            break
        except (HfHubHTTPError, OSError) as error:
            if attempt == 7:
                raise
            limited = "429" in str(error) or "rate limit" in str(error).lower()
            delay = min(3600, 900 * (attempt + 1)) if limited else 2**attempt * 15
            print(
                f"  {'rate limited' if limited else type(error).__name__}, "
                f"retry {attempt + 1} in {delay}s ({names})",
                flush=True,
            )
            time.sleep(delay)
    for _, local in pending:
        local.unlink(missing_ok=True)
    print(f"committed {len(operations)} shards: {names}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--mode", choices=["plan", "run"], default="run")
    parser.add_argument("--kind", choices=sorted(KINDS), action="append")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--target-bytes", type=int, default=500_000_000)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/exp278-upload"))
    parser.add_argument("--cluster", default=None)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    if args.cluster:
        from launch import storage_filesystem

        plan_doc = json.loads(
            storage_filesystem(args.cluster).cat(args.manifest.removeprefix("s3://"))
        )
    else:
        with fsspec.open(args.manifest, "rt") as handle:
            plan_doc = json.load(handle)
    root = plan_doc["output"]
    fs, base = open_fs(root, args.cluster)
    plan_uri = f"{base}/uploads/shard-plan.json"

    if args.mode == "plan":
        plan = build_plan(fs, base, args.target_bytes)
        fs.pipe_file(plan_uri, json.dumps(plan).encode())
        print(f"wrote s3://{plan_uri}", flush=True)
        return

    plan = json.loads(fs.cat(plan_uri))
    cases = {
        case["id"]: {"condition": case["condition"], "length": case["length"]}
        for case in plan_doc["cases"]
    }
    api = HfApi(token=os.environ["HF_TOKEN"])
    present = set(api.list_repo_files(repo_id=args.repo, repo_type="dataset"))
    args.scratch.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    pending: list[tuple[CommitOperationAdd, Path]] = []
    for kind in args.kind or sorted(KINDS):
        shards = plan["kinds"][kind]
        for index, shard in enumerate(shards):
            if index % args.workers != args.worker:
                continue
            remote = f"{kind}/shard-{index:05d}.parquet"
            if remote in present:
                skipped += 1
                continue
            local = args.scratch / f"{kind}-{index:05d}.parquet"
            started = time.time()
            if kind == "generated":
                rows = write_generated_shard(fs, shard["sources"], local, cases)
            else:
                rows = write_parquet_shard(fs, shard["sources"], local)
            packed = local.stat().st_size
            pending.append(
                (
                    CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)),
                    local,
                )
            )
            done += 1
            print(
                f"built {remote} rows={rows:,} src={shard['bytes'] / 1e6:.0f}MB "
                f"packed={packed / 1e6:.0f}MB {time.time() - started:.0f}s "
                f"({done} built, {skipped} already present)",
                flush=True,
            )
            if len(pending) >= args.batch:
                commit_batch(api, args.repo, pending)
                pending = []
    commit_batch(api, args.repo, pending)
    print(f"worker {args.worker}: uploaded {done}, skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()

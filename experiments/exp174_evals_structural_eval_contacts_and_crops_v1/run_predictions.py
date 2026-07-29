# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one inference plan over the eval set and write canonical prediction files.

The same entry point runs locally against a directory and on a CoreWeave H100
shard against object storage — everything that differs is a URI. Outputs land in
the ``canonical_pdb`` contract, so ``score_structures.py`` consumes them without
knowing which plan produced them.

Usage (local)::

    uv run python run_predictions.py --model _scratch/models/cc1mix5-step50000 \\
        --gt-dir _scratch/gt --plan F --out-dir _scratch/pred/f-cc1mix5

Usage (CoreWeave shard, S3 in and out)::

    python run_predictions.py --model s3://…/models/cc1mix5-step50000 \\
        --gt-tar s3://…/gt/gt_bundle.tar.gz --plan F \\
        --out s3://…/pred/f-cc1mix5 --shard 3/12

**Sharding interleaves a length-sorted work list** (``idx % n``), not contiguous
blocks: a shard that collected all the long proteins would set the wall clock
for the whole fan-out (root ``AGENTS.md``, exp82's fan-out lessons).

Per-input timings are captured here rather than reconstructed later, in the
schema the root ``AGENTS.md`` pins, so this experiment's numbers join with
exp12/exp20's on ``(stem, n_residues)``.
"""

import argparse
import json
import os
import platform
import socket
import tarfile
import time
from pathlib import Path

import fsspec
import torch

import canonical_pdb
from document_codec import estimate_to_atom_array
from plans import PLANS
from sampler import SamplingConfig, load_sampler

# CoreWeave object storage rejects path-style addressing; iris injects the
# endpoint + credentials as an FSSPEC_S3 blob that only fsspec/s3fs reads.
S3_CONFIG_KWARGS = {"s3": {"addressing_style": "virtual"}}


def _storage_options(uri: str) -> dict:
    if not str(uri).startswith("s3://"):
        return {}
    options = {"config_kwargs": S3_CONFIG_KWARGS}
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        options.update(
            key=os.environ["AWS_ACCESS_KEY_ID"],
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        )
    return options


def fetch_dir(uri: str, local: Path) -> Path:
    """Mirror a (possibly remote) directory locally; a local path passes through."""
    if not str(uri).startswith("s3://"):
        return Path(uri)
    local.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("s3", **_storage_options(uri))
    for remote in fs.ls(str(uri).rstrip("/"), detail=False):
        name = remote.rsplit("/", 1)[-1]
        if not name:
            continue
        started = time.time()
        fs.get_file(remote, str(local / name))
        print(f"  fetched {name} in {time.time() - started:.1f}s", flush=True)
    return local


def fetch_gt(gt_dir: str | None, gt_tar: str | None, local: Path) -> Path:
    """Get the ground-truth bundle in place, from a directory or a tarball."""
    if gt_dir:
        return Path(gt_dir)
    if not gt_tar:
        raise ValueError("one of --gt-dir / --gt-tar is required")
    local.mkdir(parents=True, exist_ok=True)
    archive = local / "gt_bundle.tar.gz"
    with fsspec.open(gt_tar, "rb", **_storage_options(gt_tar)) as src:
        archive.write_bytes(src.read())
    with tarfile.open(archive) as tar:
        tar.extractall(local)
    return local


def worker_metadata(model_name: str) -> dict:
    """Predictor + worker provenance for the timings CSV."""
    meta = {
        "model_nickname": model_name,
        "runner_tag": os.environ.get("EXP174_RUNNER_TAG", "local"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "gpu_name": "",
        "gpu_total_memory_gb": float("nan"),
        "gpu_compute_capability": "",
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        meta.update(
            gpu_name=properties.name,
            gpu_total_memory_gb=round(properties.total_memory / 1e9, 2),
            gpu_compute_capability=f"{properties.major}.{properties.minor}",
        )
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="local dir or s3:// prefix")
    ap.add_argument("--model-name", default=None, help="nickname (default: dir name)")
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--gt-tar", default=None, help="s3:// or local gt_bundle.tar.gz")
    ap.add_argument("--plan", required=True, choices=sorted(PLANS))
    ap.add_argument("--out-dir", default=None, help="local output root")
    ap.add_argument("--out", default=None, help="s3:// output prefix")
    ap.add_argument("--shard", default="0/1", metavar="I/N")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coord-temperature", type=float, default=1.0)
    ap.add_argument("--struct-temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--n-sweeps", type=int, default=3, help="plan F")
    ap.add_argument("--n-samples", type=int, default=4, help="plan F samples per crop")
    ap.add_argument("--n-neighbor-crops", type=int, default=6, help="plan F")
    ap.add_argument(
        "--pass1-feedback-sigma",
        type=float,
        default=None,
        help="plan F: Å noise on the re-synthesized Pass-1 section (default: the "
        "format's own σ=2; 0 for the clean-feedback ablation)",
    )
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/exp174"))
    args = ap.parse_args(argv)

    shard_i, num_shards = (int(v) for v in args.shard.split("/"))
    model_dir = fetch_dir(args.model, args.scratch / "model")
    gt_root = fetch_gt(args.gt_dir, args.gt_tar, args.scratch / "gt")
    model_name = args.model_name or Path(str(args.model).rstrip("/")).name

    records = [json.loads(line) for line in (gt_root / "gt_index.jsonl").open()]
    # Interleave a length-sorted list so no shard collects all the long chains.
    records.sort(key=lambda r: (-int(r["L"]), r["record_id"]))
    records = [r for i, r in enumerate(records) if i % num_shards == shard_i]
    if args.limit is not None:
        records = records[: args.limit]

    config = SamplingConfig(
        coord_temperature=args.coord_temperature,
        struct_temperature=args.struct_temperature,
        top_k=args.top_k,
    )
    plan = PLANS[args.plan]
    plan_kwargs = {}
    if args.plan == "F":
        plan_kwargs = dict(
            n_sweeps=args.n_sweeps,
            n_samples=args.n_samples,
            n_neighbor_crops=args.n_neighbor_crops,
            pass1_feedback_sigma=args.pass1_feedback_sigma,
        )

    sampler = load_sampler(model_dir)
    meta = worker_metadata(model_name)
    meta["model_load_seconds"] = round(sampler.model_load_seconds, 2)
    print(
        f"[predict] plan={args.plan} model={model_name} shard={shard_i}/{num_shards} "
        f"records={len(records)} gpu={meta['gpu_name']}",
        flush=True,
    )

    out_local = Path(args.out_dir) if args.out_dir else args.scratch / "pred"
    out_local.mkdir(parents=True, exist_ok=True)
    stats_path = out_local / f"stats_shard{shard_i:03d}of{num_shards:03d}.jsonl"
    timings_path = out_local / f"timings_shard{shard_i:03d}of{num_shards:03d}.jsonl"

    started = time.time()
    n_ok = 0
    with stats_path.open("w") as stats_fh, timings_path.open("w") as timings_fh:
        for i, record in enumerate(records, start=1):
            gt_path = (
                gt_root / "gt_structures" / record["dataset"] / f"{record['stem']}.pdb"
            )
            gt = canonical_pdb.read_structure(gt_path)
            total_started = time.time()
            result = plan(
                sampler,
                record,
                config=config,
                gt=gt,
                seed=args.seed,
                **plan_kwargs,
            )
            total_seconds = time.time() - total_started

            stats = dict(result.stats)
            stats["record_id"] = record["record_id"]
            stats_fh.write(json.dumps(stats) + "\n")

            timings_fh.write(
                json.dumps(
                    {
                        "stem": record["stem"],
                        "record_id": record["record_id"],
                        "n_residues": record["L"],
                        "n_pairs": record["L"] * (record["L"] - 1) // 2,
                        "mode": f"plan-{args.plan}",
                        "elapsed_seconds": round(
                            float(stats.get("elapsed_seconds", total_seconds)), 3
                        ),
                        "model_load_seconds": meta["model_load_seconds"],
                        "total_seconds": round(total_seconds, 3),
                        "n_samples": args.n_samples if args.plan == "F" else 1,
                        "n_sweeps": stats.get("n_sweeps_run", 1),
                        "timestamp_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        **{
                            k: meta[k]
                            for k in (
                                "model_nickname", "runner_tag", "gpu_name",
                                "gpu_total_memory_gb", "gpu_compute_capability",
                                "hostname", "platform", "torch_version",
                            )
                        },
                    }
                )
                + "\n"
            )

            array = estimate_to_atom_array(result.estimate, record["input_seq"])
            if array is not None:
                out_path = out_local / record["dataset"] / f"{record['stem']}.pdb"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                canonical_pdb.write_structure(array, out_path)
                n_ok += 1
            if i % 10 == 0:
                rate = (time.time() - started) / i
                print(
                    f"  ...{i}/{len(records)} ({rate:.1f}s/protein, "
                    f"eta {rate * (len(records) - i) / 60:.0f} min)",
                    flush=True,
                )

    print(
        f"[predict] {n_ok}/{len(records)} structures in "
        f"{(time.time() - started) / 60:.1f} min -> {out_local}",
        flush=True,
    )

    if args.out:
        fs = fsspec.filesystem("s3", **_storage_options(args.out))
        prefix = str(args.out).rstrip("/")
        for path in sorted(out_local.rglob("*")):
            if path.is_file():
                fs.put_file(str(path), f"{prefix}/{path.relative_to(out_local)}")
        print(f"[predict] uploaded -> {prefix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
